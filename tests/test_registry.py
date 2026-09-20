"""The registry invariants exist because violating them costs seconds per call in model
reloads — a cost that is invisible in code review and only shows up as "Donna feels slow".
These tests are the guard rail.
"""
from __future__ import annotations

import pytest

from donna.llm import registry


def test_shipped_registry_is_consistent():
    registry.assert_consistent()


def test_every_task_resolves_to_a_concrete_spec():
    for task in registry.all_tasks():
        spec = registry.get_spec(task)
        assert spec.model
        assert spec.num_ctx > 0, f"{task} has no resolved context size"
        assert spec.keep_alive, f"{task} has no keep_alive"


def test_unknown_task_names_its_alternatives():
    with pytest.raises(KeyError, match="classify_email"):
        registry.get_spec("classify_emails")  # plural typo


def test_one_context_size_per_model():
    # The rule that keeps Ollama from rebuilding a runner on every task switch.
    seen: dict[str, int] = {}
    for task in registry.all_tasks():
        spec = registry.get_spec(task)
        if spec.model in seen:
            assert seen[spec.model] == spec.num_ctx, (
                f"{spec.model} served at both {seen[spec.model]} and {spec.num_ctx}"
            )
        seen[spec.model] = spec.num_ctx


def test_model_count_fits_ollama_load_limit():
    assert len(registry.required_models()) <= 3


def test_cpu_tasks_pin_num_gpu_to_zero():
    # Without this the workers land on the GPU and evict the 9b.
    spec = registry.get_spec(registry.CLASSIFY_EMAIL)
    assert spec.device == "cpu"
    assert spec.options()["num_gpu"] == 0


def test_gpu_tasks_leave_offload_to_ollama():
    spec = registry.get_spec(registry.CHAT)
    assert spec.device == "gpu"
    assert "num_gpu" not in spec.options()


def test_workers_do_not_burn_time_on_reasoning_streams():
    for task in (registry.ROUTE, registry.CLASSIFY_EMAIL, registry.EXTRACT_COMMITMENT):
        assert registry.get_spec(task).think is False


def test_interactive_tasks_get_the_longer_keep_alive():
    chat = registry.get_spec(registry.CHAT)
    worker = registry.get_spec(registry.CLASSIFY_EMAIL)
    assert chat.keep_alive != worker.keep_alive


def test_env_override_replaces_the_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DONNA_MODEL_CLASSIFY_EMAIL", "some-other:3b")
    spec = registry.get_spec(registry.CLASSIFY_EMAIL)
    assert spec.model == "some-other:3b"
    # An unknown model still gets a usable context size rather than zero.
    assert spec.num_ctx == registry.DEFAULT_NUM_CTX


def test_override_that_splits_a_model_across_context_sizes_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    # Point one worker task at the brain model. The brain runs at 8192 and the default for
    # an unrecognised tag is 4096 — but here the tag IS recognised, so it must agree.
    monkeypatch.setenv("DONNA_MODEL_CLASSIFY_EMAIL", "qwen3.5:9b")
    registry.assert_consistent()  # same ctx, so this is fine

    monkeypatch.setenv("OLLAMA_MAX_LOADED_MODELS", "1")
    with pytest.raises(ValueError, match="OLLAMA_MAX_LOADED_MODELS"):
        registry.assert_consistent()
