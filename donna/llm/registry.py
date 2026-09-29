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


# --- One model, not a tier list ----------------------------------------------
# This started as three tiers (0.8b router, 2b workers, 9b brain) and measurement collapsed
# it to one. The evidence, all on this machine (RTX 3060 Ti 8 GB, 16 GB system RAM):
#
# 1. The 0.8b router lost to the 2b, 3134-3509 ms against 1383-1547 ms, because it was the
#    least-recently-used runner and reloaded (~2.2 s) on essentially every call.
#
# 2. Then the decisive one. Ollama silently moves a model to the CPU when it decides it no
#    longer fits, and reports it only in `ollama ps`:
#
#       9b alone resident            -> 100% GPU, 5.6 GB, 57.8 tok/s
#       9b with the 2b also resident -> 100% CPU,          ~10 tok/s
#
#    Holding a second model resident cost a 6x slowdown on the model that matters. Every
#    early triage measurement (10-16 s per email) was the 9b running on the CPU without
#    saying so. The "worker on CPU to avoid VRAM contention" idea backfired: the contention
#    was not for VRAM, it was for Ollama's willingness to keep the big model on the GPU.
#
# So: one model serves every generative task. It is always resident, always on the GPU,
# never swapped, and it is also the most accurate of the three (12/12 on a hand-labelled
# triage set where the 2b managed 0/12 on the same prompt).
#
# The SLM architecture is unaffected, because specialisation here was never about separate
# weights: each task still has its own prompt, its own output schema and its own sampling.
# That is where a per-task LoRA adapter would attach later, over this same base.
_BRAIN_MODEL = "qwen3.5:9b"
# Kept as a name so a task can be pointed at something cheaper via DONNA_MODEL_<TASK>, but
# nothing uses it by default — see above for why a second resident model is not free.
_WORKER_MODEL = "qwen3.5:2b"
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
    _WORKER_MODEL: 8192,  # sized for the longest worker input (a full email thread)
    _BRAIN_MODEL: 8192,  # KV cache at 8k costs ~84 MiB of VRAM; not the lever to pull
    _EMBED_MODEL: 512,
}

# Specialisation is per task: prompt, schema, sampling and output budget. The model is the
# same, and that is a deliberate consequence of the note above, not an oversight.
_SPECS: dict[str, ModelSpec] = {
    ROUTE: ModelSpec(ROUTE, _BRAIN_MODEL, "gpu", num_predict=64),
    CLASSIFY_EMAIL: ModelSpec(CLASSIFY_EMAIL, _BRAIN_MODEL, "gpu", num_predict=400),
    EXTRACT_COMMITMENT: ModelSpec(EXTRACT_COMMITMENT, _BRAIN_MODEL, "gpu", num_predict=400),
    # Retained as a distinct task so an escalation path stays available (and so the eval
    # harness can score a second configuration), but it is the same base model today.
    EXTRACT_COMMITMENT_BIG: ModelSpec(EXTRACT_COMMITMENT_BIG, _BRAIN_MODEL, "gpu", num_predict=400),
    SUMMARIZE: ModelSpec(SUMMARIZE, _BRAIN_MODEL, "gpu", temperature=0.2),
    EXTRACT_FACTS: ModelSpec(EXTRACT_FACTS, _BRAIN_MODEL, "gpu", num_predict=400),
    CHAT: ModelSpec(CHAT, _BRAIN_MODEL, "gpu", temperature=0.45),
    SCHEDULE: ModelSpec(SCHEDULE, _BRAIN_MODEL, "gpu", temperature=0.2),
    PLAN: ModelSpec(PLAN, _BRAIN_MODEL, "gpu", temperature=0.1),
    # The one genuine exception: embeddings need a different architecture, not a different
    # size. Pinned to the CPU so it cannot be the second model that pushes the 9b off the
    # GPU — it is 669 MB and a batch of three vectors costs under a second there.
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
            keep_alive=settings.chat_keep_alive
            if task in _INTERACTIVE
            else settings.worker_keep_alive,
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
            f"{model}: "
            + ", ".join(f"ctx={ctx} ({', '.join(tasks)})" for ctx, tasks in sizes.items())
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
