"""Task -> model mapping.

This is the SLM architecture made explicit. A "specialised model" here means a specific
(model, device, sampling, prompt, output schema) combination bound to one narrow job.
Reassigning a job to a different model is a one-line edit, and DONNA_MODEL_<TASK>
overrides any entry at runtime without touching code.

Device assignment matters as much as model size on an 8 GB card. Measured on an
RTX 3060 Ti with qwen3.5:9b resident:

    baseline desktop         ~1.3 GB
    + qwen3.5:9b @ 8k ctx    ~7.8 GB total   (weights dominate; KV cache is ~84 MiB)

That leaves no room for a second model on the GPU. So the high-volume workers are pinned
to the CPU with num_gpu=0, where a 2B emitting 40 tokens of JSON costs ~1.4 s on a
12700K. No VRAM contention, and crucially no model swapping: Ollama would otherwise
evict the 9b every time a background job ran, costing ~30 s to reload it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

Device = Literal["gpu", "cpu"]

# Task names. Used as registry keys, trace labels and eval suite names.
ROUTE = "route"
CLASSIFY_EMAIL = "classify_email"
EXTRACT_COMMITMENT = "extract_commitment"
EXTRACT_COMMITMENT_BIG = "extract_commitment_big"
SUMMARIZE = "summarize"
EXTRACT_FACTS = "extract_facts"
CHAT = "chat"
SCHEDULE = "schedule"
PLAN = "plan"
EMBED = "embed"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    task: str
    model: str
    device: Device = "cpu"
    num_ctx: int = 0  # filled from _MODEL_CTX by get_spec(); see the note there
    temperature: float = 0.0
    num_predict: int = -1
    # Hybrid-reasoning models emit a separate thinking stream. Off for workers: it
    # multiplies latency for jobs whose output is a three-field JSON object.
    think: bool = False
    keep_alive: str | None = None

    @property
    def num_gpu(self) -> int | None:
        """Ollama layer-offload count. 0 forces CPU; None leaves it to Ollama."""
        return 0 if self.device == "cpu" else None

    def options(self) -> dict[str, object]:
        opts: dict[str, object] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
        }
        if self.num_predict > 0:
            opts["num_predict"] = self.num_predict
        if self.num_gpu is not None:
            opts["num_gpu"] = self.num_gpu
        return opts


# --- Tiers -------------------------------------------------------------------
# Two tiers, not three. The original design had a 0.8b "reflex" tier for routing, on the
# reasoning that a smaller model routes faster. Measured, it does not:
#
#   route on qwen3.5:0.8b   3134-3509 ms   (gen ~1.05 s + load ~2.2 s, every single call)
#   route on qwen3.5:2b     1383-1547 ms   (gen ~1.38 s + load ~2 ms)
#
# The 0.8b generates slightly faster and still loses by 2.4x, because a third model does
# not stay resident here: with 16 GB of system RAM and Ollama's default
# OLLAMA_MAX_LOADED_MODELS=3 (the embedding model takes the third slot), the
# least-recently-used CPU runner gets evicted, and the router is always the longest-idle
# one — it runs first in a request, then the worker and the brain run after it.
#
# So the lesson is the opposite of the intuition: on a memory-constrained box the fastest
# model for a job is usually the one already loaded. Routing rides the worker runner.
# qwen3.5:0.8b is still a valid experiment via DONNA_MODEL_ROUTE if this ever changes.
_WORKER_MODEL = "qwen3.5:2b"    # routing, classification, extraction — CPU, always hot
_BRAIN_MODEL = "qwen3.5:9b"     # conversation and planning — GPU-resident, ~59 tok/s
_EMBED_MODEL = "mxbai-embed-large"

# Context size is a property of the MODEL, not of the task, and this is not a
# simplification — it is a hard requirement. Ollama keys its runner process by
# (model, num_ctx), so serving one model tag at two context sizes makes every switch a
# full reload. Measured cost of getting this wrong: the 2b reloaded at 3490 ms going
# 4096 -> 8192 and 3008 ms coming back, on every single alternation. Four runners also
# exceeded Ollama's default OLLAMA_MAX_LOADED_MODELS=3 and evicted the router.
#
# So: one entry per model, and assert_consistent() refuses to let this drift.
_MODEL_CTX: dict[str, int] = {
    _WORKER_MODEL: 8192,   # sized for the longest worker input (a full email thread)
    _BRAIN_MODEL: 8192,    # KV cache at 8k costs ~84 MiB of VRAM; not the lever to pull
    _EMBED_MODEL: 512,
}

_SPECS: dict[str, ModelSpec] = {
    ROUTE: ModelSpec(ROUTE, _WORKER_MODEL, "cpu", num_predict=64),
    CLASSIFY_EMAIL: ModelSpec(CLASSIFY_EMAIL, _WORKER_MODEL, "cpu", num_predict=200),
    EXTRACT_COMMITMENT: ModelSpec(EXTRACT_COMMITMENT, _WORKER_MODEL, "cpu", num_predict=400),
    # Escalation path for low-confidence extractions. Same prompt, bigger model.
    EXTRACT_COMMITMENT_BIG: ModelSpec(
        EXTRACT_COMMITMENT_BIG, _BRAIN_MODEL, "gpu", num_predict=400
    ),
    SUMMARIZE: ModelSpec(SUMMARIZE, _WORKER_MODEL, "cpu", temperature=0.2),
    EXTRACT_FACTS: ModelSpec(EXTRACT_FACTS, _WORKER_MODEL, "cpu", num_predict=400),
    CHAT: ModelSpec(CHAT, _BRAIN_MODEL, "gpu", temperature=0.45),
    SCHEDULE: ModelSpec(SCHEDULE, _BRAIN_MODEL, "gpu", temperature=0.2),
    PLAN: ModelSpec(PLAN, _BRAIN_MODEL, "gpu", temperature=0.1),
    EMBED: ModelSpec(EMBED, _EMBED_MODEL, "cpu"),
}

# Tasks that carry a conversation rather than filling a schema; they get the longer
# keep-alive so the GPU model is not evicted between turns.
_INTERACTIVE = {CHAT, SCHEDULE, PLAN, EXTRACT_COMMITMENT_BIG}


DEFAULT_NUM_CTX = 4096


def get_spec(task: str) -> ModelSpec:
    try:
        spec = _SPECS[task]
    except KeyError as exc:
        raise KeyError(
            f"Unknown LLM task {task!r}. Known tasks: {', '.join(sorted(_SPECS))}"
        ) from exc

    override = os.getenv(f"DONNA_MODEL_{task.upper()}")
    if override:
        spec = replace(spec, model=override)

    if spec.num_ctx <= 0:
        spec = replace(spec, num_ctx=_MODEL_CTX.get(spec.model, DEFAULT_NUM_CTX))

    if spec.keep_alive is None:
        from donna.config import get_settings

        settings = get_settings()
        spec = replace(
            spec,
            keep_alive=settings.chat_keep_alive if task in _INTERACTIVE else settings.worker_keep_alive,
        )
    return spec


def all_tasks() -> list[str]:
    return list(_SPECS)


def assert_consistent() -> None:
    """Refuse a registry that would make Ollama thrash.

    Two rules, both learned the hard way:
      * one num_ctx per model tag, or every task switch reloads the model;
      * no more distinct models than Ollama will keep loaded at once, or the least
        recently used gets evicted and pays a cold load on its next call.
    """
    by_model: dict[str, dict[int, list[str]]] = {}
    for task in _SPECS:
        spec = get_spec(task)
        by_model.setdefault(spec.model, {}).setdefault(spec.num_ctx, []).append(task)

    conflicts = {m: sizes for m, sizes in by_model.items() if len(sizes) > 1}
    if conflicts:
        detail = "; ".join(
            f"{model}: " + ", ".join(f"ctx={ctx} ({', '.join(tasks)})" for ctx, tasks in sizes.items())
            for model, sizes in conflicts.items()
        )
        raise ValueError(
            "Registry inconsistente — stesso modello con num_ctx diversi, "
            f"Ollama ricaricherebbe a ogni switch: {detail}"
        )

    limit = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "3"))
    if len(by_model) > limit:
        raise ValueError(
            f"Il registry usa {len(by_model)} modelli distinti ma OLLAMA_MAX_LOADED_MODELS={limit}. "
            "Alza la variabile o riduci i modelli, altrimenti Ollama ne scarica uno a ogni giro."
        )


def model_context_sizes() -> dict[str, int]:
    return {get_spec(t).model: get_spec(t).num_ctx for t in _SPECS}


def required_models() -> list[str]:
    """Distinct model tags that must be present in Ollama, after env overrides."""
    seen: list[str] = []
    for task in _SPECS:
        tag = get_spec(task).model
        if tag not in seen:
            seen.append(tag)
    return seen
