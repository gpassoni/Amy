"""Ollama client.

Deliberately talks to the HTTP API with httpx rather than using an SDK: every call needs
per-request control over num_gpu, num_ctx, keep_alive and the JSON schema in `format`, and
we want the raw request and response bodies handed to the tracer verbatim.

Nothing in this module can reach a network service other than the local Ollama host.

Synchronous by design. Async callers (Telegram, FastAPI) use the a* wrappers, which hand
off to a worker thread — inference blocks on the Ollama side, so there is nothing to gain
from an async HTTP client here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from amy.config import get_settings
from amy.llm import registry
from amy.llm.tracing import Trace, dump_messages

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for inference failures."""


class LLMUnavailable(LLMError):
    """Ollama is not reachable, or the requested model is not pulled."""


class LLMInvalidOutput(LLMError):
    """The model did not produce output matching the requested schema."""

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class StructuredResult(BaseModel, Generic[T]):
    value: T
    trace_id: str
    model: str
    latency_ms: int
    attempts: int = 1


class ChatResult(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = []
    trace_id: str
    model: str
    latency_ms: int


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.timeout = timeout or settings.ollama_timeout
        self._client = httpx.Client(base_url=self.host, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ plumbing
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise LLMUnavailable(
                f"Ollama non raggiungibile su {self.host}. Avvialo con: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            model = payload.get("model")
            if exc.response.status_code == 404:
                raise LLMUnavailable(
                    f"Modello non presente in Ollama: {model!r}. "
                    f"Scaricalo con: ollama pull {model} ({detail})"
                ) from exc
            raise LLMError(f"Ollama ha risposto {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Timeout dopo {self.timeout:.0f}s su {path}") from exc

    def _build_payload(
        self,
        spec: registry.ModelSpec,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "stream": False,
            "think": spec.think,
            "keep_alive": spec.keep_alive,
            "options": spec.options(),
        }
        if schema is not None:
            payload["format"] = schema
        if tools:
            payload["tools"] = tools
        return payload

    # ------------------------------------------------------------------ health
    def version(self) -> str:
        try:
            return self._client.get("/api/version").json().get("version", "?")
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama non raggiungibile su {self.host}") from exc

    def installed_models(self) -> set[str]:
        try:
            data = self._client.get("/api/tags").json()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama non raggiungibile su {self.host}") from exc
        return {m["model"] for m in data.get("models", [])}

    def missing_models(self) -> list[str]:
        """Registry models not present locally. Ollama reports name:tag; bare means :latest."""
        normalised = {m if ":" in m else f"{m}:latest" for m in self.installed_models()}
        missing = []
        for tag in registry.required_models():
            wanted = tag if ":" in tag else f"{tag}:latest"
            if wanted not in normalised:
                missing.append(tag)
        return missing

    def loaded(self) -> list[dict[str, Any]]:
        """What is currently resident, from /api/ps — used to report VRAM decisions."""
        try:
            return self._client.get("/api/ps").json().get("models", [])
        except httpx.HTTPError:
            return []

    # ------------------------------------------------------------------ structured
    def structured(
        self,
        task: str,
        schema: type[T],
        user: str,
        *,
        system: str | None = None,
        parent_trace_id: str | None = None,
        retries: int = 1,
    ) -> StructuredResult[T]:
        """Call the task's model and validate the reply against a Pydantic model.

        Small models occasionally emit schema-shaped-but-invalid JSON. On failure the
        validation error is fed back once as a corrective turn, which recovers most cases
        far more cheaply than escalating to a bigger model.
        """
        spec = registry.get_spec(task)
        json_schema = schema.model_json_schema()

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        trace = Trace(
            task=task,
            model=spec.model,
            device=spec.device,
            parent_id=parent_trace_id,
            system_prompt=system,
            schema_name=schema.__name__,
        )

        last_error: Exception | None = None
        raw = ""
        started = time.perf_counter()
        load_ms: int | None = None

        for attempt in range(1, retries + 2):
            payload = self._build_payload(spec, messages, schema=json_schema)
            try:
                data = self._post("/api/chat", payload)
            except LLMError as exc:
                trace.prompt = dump_messages(messages)
                trace.ok = False
                trace.error = str(exc)
                trace.latency_ms = int((time.perf_counter() - started) * 1000)
                trace.save()
                raise

            raw = (data.get("message") or {}).get("content") or ""
            load_ms = int((data.get("load_duration") or 0) / 1e6) or load_ms
            trace.tokens_in = data.get("prompt_eval_count")
            trace.tokens_out = data.get("eval_count")

            try:
                value = schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Task %s attempt %d: output did not validate against %s (%s)",
                    task,
                    attempt,
                    schema.__name__,
                    _short(exc),
                )
                if attempt > retries:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Il JSON non rispetta lo schema richiesto. Errori:\n"
                                f"{_short(exc)}\n"
                                "Rispondi di nuovo con SOLO il JSON corretto."
                            ),
                        },
                    ]
                )
                continue

            trace.prompt = dump_messages(messages)
            trace.output = raw
            trace.latency_ms = int((time.perf_counter() - started) * 1000)
            trace.load_ms = load_ms
            trace.save()
            return StructuredResult[schema](
                value=value,
                trace_id=trace.id,
                model=spec.model,
                latency_ms=trace.latency_ms,
                attempts=attempt,
            )

        trace.prompt = dump_messages(messages)
        trace.output = raw
        trace.ok = False
        trace.error = f"schema validation failed: {_short(last_error)}"
        trace.latency_ms = int((time.perf_counter() - started) * 1000)
        trace.save()
        raise LLMInvalidOutput(
            f"{task}: il modello {spec.model} non ha prodotto un {schema.__name__} valido "
            f"dopo {retries + 1} tentativi",
            raw=raw,
        )

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        parent_trace_id: str | None = None,
    ) -> ChatResult:
        """Free-form turn, optionally with tools. Used by the agent loop."""
        spec = registry.get_spec(task)
        # The first system message is the stable persona; everything else — including the
        # world-state block, which is also a system message — has to stay in `prompt`.
        # Filtering out every system role dropped the state from the trace entirely, so the
        # dashboard could not show what Amy had actually been told. That is precisely the
        # question the trace view exists to answer.
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        trace = Trace(
            task=task,
            model=spec.model,
            device=spec.device,
            parent_id=parent_trace_id,
            system_prompt=system,
            prompt=dump_messages(messages[1:] if system else messages),
        )

        started = time.perf_counter()
        try:
            data = self._post("/api/chat", self._build_payload(spec, messages, tools=tools))
        except LLMError as exc:
            trace.ok = False
            trace.error = str(exc)
            trace.latency_ms = int((time.perf_counter() - started) * 1000)
            trace.save()
            raise

        message = data.get("message") or {}
        trace.output = message.get("content") or ""
        trace.tokens_in = data.get("prompt_eval_count")
        trace.tokens_out = data.get("eval_count")
        trace.latency_ms = int((time.perf_counter() - started) * 1000)
        trace.load_ms = int((data.get("load_duration") or 0) / 1e6)
        trace.save()

        return ChatResult(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            trace_id=trace.id,
            model=spec.model,
            latency_ms=trace.latency_ms,
        )

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], *, task: str = registry.EMBED) -> list[list[float]]:
        if not texts:
            return []
        spec = registry.get_spec(task)
        payload: dict[str, Any] = {
            "model": spec.model,
            "input": texts,
            "keep_alive": spec.keep_alive,
            "options": spec.options(),
        }
        data = self._post("/api/embed", payload)
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise LLMError(
                f"Attesi {len(texts)} embedding, ricevuti {len(embeddings)} da {spec.model}"
            )
        return embeddings

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # ------------------------------------------------------------------ async wrappers
    async def astructured(self, *args: Any, **kwargs: Any) -> StructuredResult[Any]:
        return await asyncio.to_thread(self.structured, *args, **kwargs)

    async def achat(self, *args: Any, **kwargs: Any) -> ChatResult:
        return await asyncio.to_thread(self.chat, *args, **kwargs)

    async def aembed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, *args, **kwargs)


def _short(exc: Exception | None, limit: int = 400) -> str:
    if exc is None:
        return "unknown"
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()[:4]
        ]
        return "; ".join(parts)[:limit]
    return str(exc)[:limit]


_client: OllamaClient | None = None


def get_llm() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
