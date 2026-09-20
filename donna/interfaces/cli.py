"""Development CLI.

Exists so every layer can be exercised without Telegram or a browser in the loop. The
`doctor` and `smoke` commands are the Phase 0 acceptance check: they prove the local model
tiers work, report what each one costs, and show what is resident on the GPU.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel

from donna.config import get_settings
from donna.llm import registry
from donna.llm.client import LLMError, LLMUnavailable, get_llm
from donna.logging_setup import setup_logging
from donna.store.db import get_db
from donna.timeutil import format_it, now_local


def _init_console() -> bool:
    """Make the Windows console behave: UTF-8 out, colours only when attached to a TTY."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


_COLOR = _init_console()


def _c(code: str) -> str:
    return code if _COLOR else ""


OK = _c("\x1b[32m") + "+" + _c("\x1b[0m")
BAD = _c("\x1b[31m") + "x" + _c("\x1b[0m")
WARN = _c("\x1b[33m") + "!" + _c("\x1b[0m")
DIM = _c("\x1b[2m")
BOLD = _c("\x1b[1m")
RESET = _c("\x1b[0m")


# --- Probe schemas used by `smoke`. Deliberately shaped like the real ones. -------
class _Intent(BaseModel):
    # Literal, not str: Ollama compiles the schema into a sampling grammar, so an enum
    # makes an out-of-vocabulary answer physically unrepresentable. With a plain str and
    # a description, the 0.8b invented "questioning_about_future_activities".
    intent: Literal["schedule_query", "inbox_query", "task_mutate", "smalltalk"]


class _Category(BaseModel):
    category: str
    confidence: float
    reason: str


class _Commitment(BaseModel):
    has_commitment: bool
    title: str | None = None
    date_phrase: str | None = None
    confidence: float = 0.0


class _Summary(BaseModel):
    summary: str


class _Facts(BaseModel):
    facts: list[str]


# (task, schema, system, user) — one representative call per registry task.
_PROBES: list[tuple[str, type[BaseModel] | None, str, str]] = [
    (
        registry.ROUTE,
        _Intent,
        "Classifica l'intento del messaggio. Rispondi solo JSON.",
        "cosa ho in programma domani pomeriggio?",
    ),
    (
        registry.CLASSIFY_EMAIL,
        _Category,
        "Classifica l'email in importante, da_leggere o inutile. Solo JSON.",
        "DA: dentista@studio.it\nOGGETTO: Conferma appuntamento\nCORPO: Le confermiamo "
        "l'appuntamento di martedi 24 settembre alle 15:00.",
    ),
    (
        registry.EXTRACT_COMMITMENT,
        _Commitment,
        "Estrai un eventuale impegno con data dall'email. Solo JSON.",
        "DA: dentista@studio.it\nOGGETTO: Conferma appuntamento\nCORPO: Le confermiamo "
        "l'appuntamento di martedi 24 settembre alle 15:00.",
    ),
    (
        registry.EXTRACT_COMMITMENT_BIG,
        _Commitment,
        "Estrai un eventuale impegno con data dall'email. Solo JSON.",
        "DA: studio@legale.it\nOGGETTO: Udienza\nCORPO: L'udienza e' fissata per il "
        "3 ottobre alle 9:30 presso il tribunale di Milano.",
    ),
    (
        registry.SUMMARIZE,
        _Summary,
        "Riassumi in una frase. Solo JSON.",
        "Il progetto e' in ritardo di due settimane. Il cliente ha chiesto un incontro "
        "per rivedere le scadenze e ha proposto lunedi prossimo.",
    ),
    (
        registry.EXTRACT_FACTS,
        _Facts,
        "Estrai fatti durevoli sull'utente. Solo JSON.",
        "Vado in palestra il martedi e il giovedi sera, e odio le riunioni prima delle 10.",
    ),
    (
        registry.SCHEDULE,
        None,
        "Sei un assistente conciso.",
        "In una frase: perche' e' utile bloccare del tempo in calendario?",
    ),
    (
        registry.CHAT,
        None,
        "Sei Donna, assistente personale. Sintetica e sicura di te.",
        "Presentati in una frase.",
    ),
]


def _gpu_memory() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _tokens_for(trace_id: str) -> str:
    """in/out token counts, read back from the trace the call just wrote."""
    row = get_db().query_one(
        "SELECT tokens_in, tokens_out, load_ms FROM traces WHERE id = ?", (trace_id,)
    )
    if row is None:
        return ""
    marker = "*" if (row["load_ms"] or 0) > 500 else ""
    return f"{row['tokens_in'] or 0}/{row['tokens_out'] or 0}{marker}"


# ---------------------------------------------------------------- commands
def cmd_init_db(_: argparse.Namespace) -> int:
    db = get_db()
    applied = db.migrate()
    if applied:
        print(f"{OK} migrazioni applicate: {', '.join(applied)}")
    else:
        print(f"{OK} schema già aggiornato")
    print(f"{DIM}  db: {db.path}{RESET}")
    print(f"{DIM}  tabelle: {', '.join(db.table_names())}{RESET}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    settings = get_settings()
    problems = 0

    print(f"\n{BOLD}Donna doctor{RESET}  {DIM}{format_it(now_local())}{RESET}\n")

    # --- storage
    db = get_db()
    try:
        applied = db.migrate()
        tables = db.table_names()
        note = f" (applicate: {', '.join(applied)})" if applied else ""
        print(f"{OK} database      {db.path.name}, {len(tables)} tabelle{note}")
    except Exception as exc:
        problems += 1
        print(f"{BAD} database      {exc}")

    # --- ollama
    llm = get_llm()
    try:
        version = llm.version()
        print(f"{OK} ollama        v{version} su {llm.host}")
    except LLMUnavailable as exc:
        print(f"{BAD} ollama        {exc}")
        return 1

    try:
        registry.assert_consistent()
        print(f"{OK} registry      coerente (un num_ctx per modello, entro il limite di Ollama)")
    except ValueError as exc:
        problems += 1
        print(f"{BAD} registry      {exc}")

    missing = llm.missing_models()
    if missing:
        problems += 1
        print(f"{BAD} modelli       mancanti: {', '.join(missing)}")
        for tag in missing:
            print(f"{DIM}                ollama pull {tag}{RESET}")
    else:
        print(f"{OK} modelli       tutti presenti: {', '.join(registry.required_models())}")

    # --- what the registry decided
    print(f"\n  {DIM}task                     modello              dev    ctx  keep_alive{RESET}")
    for task in registry.all_tasks():
        spec = registry.get_spec(task)
        print(
            f"  {task:<24} {spec.model:<20} {spec.device:<6} {spec.num_ctx:>5}  {spec.keep_alive}"
        )

    # --- residency
    loaded = llm.loaded()
    print()
    if loaded:
        for entry in loaded:
            size_gb = (entry.get("size") or 0) / 1e9
            vram_gb = (entry.get("size_vram") or 0) / 1e9
            where = "GPU" if vram_gb > 0.1 else "CPU"
            print(
                f"{OK} residente     {entry.get('model')} — {size_gb:.1f} GB "
                f"({vram_gb:.1f} GB VRAM, {where})"
            )
    else:
        print(f"{WARN} residente     nessun modello caricato (si caricherà alla prima chiamata)")

    gpu = _gpu_memory()
    if gpu:
        print(f"{DIM}  GPU: {gpu}{RESET}")

    # --- google credentials (presence only; no auth attempted here)
    for label, path in (
        ("credentials", settings.google_credentials_file),
        ("token", settings.google_token_file),
    ):
        if path.exists():
            print(f"{OK} google {label:<7} {path.name}")
        else:
            marker = WARN if label == "token" else BAD
            if label != "token":
                problems += 1
            print(f"{marker} google {label:<7} manca: {path}")

    if not settings.telegram_bot_token:
        print(f"{WARN} telegram      TELEGRAM_BOT_TOKEN non impostato")

    print()
    print(f"{OK} nessun problema bloccante" if problems == 0 else f"{BAD} {problems} problemi")
    return 1 if problems else 0


def cmd_smoke(args: argparse.Namespace) -> int:
    get_db().migrate()
    llm = get_llm()

    missing = llm.missing_models()
    if missing:
        print(f"{BAD} modelli mancanti: {', '.join(missing)}")
        return 1

    print(f"\n{BOLD}Smoke test{RESET} — una chiamata reale per ogni task del registry\n")
    header = (
        f"  {'task':<24} {'modello':<16} {'dev':<4} {'try':>4} "
        f"{'tok in/out':>14} {'latency':>10}"
    )
    print(f"{DIM}{header}{RESET}")

    failures = 0
    total_ms = 0
    for task, schema, system, user in _PROBES:
        spec = registry.get_spec(task)
        try:
            if schema is None:
                result = llm.chat(
                    task,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                attempts, latency = 1, result.latency_ms
                payload: Any = result.content.strip().replace("\n", " ")
                tokens = _tokens_for(result.trace_id)
            else:
                result = llm.structured(task, schema, user, system=system)
                attempts, latency = result.attempts, result.latency_ms
                payload = result.value.model_dump()
                tokens = _tokens_for(result.trace_id)
        except LLMError as exc:
            failures += 1
            print(f"  {BAD} {task:<22} {spec.model:<16} {spec.device:<4} {exc}")
            continue

        total_ms += latency
        flag = OK if attempts == 1 else WARN
        print(
            f"  {flag} {task:<22} {spec.model:<16} {spec.device:<4} {attempts:>4} "
            f"{tokens:>14} {latency:>7} ms"
        )
        if args.verbose:
            text = payload if isinstance(payload, str) else repr(payload)
            print(f"{DIM}      -> {text[:160]}{RESET}")

    # Embeddings are not a chat call, so they get their own probe. Semantic memory depends
    # on related text scoring above unrelated text, so assert that rather than just "ran".
    import time as _time

    started = _time.perf_counter()
    try:
        vectors = llm.embed(
            [
                "vado in palestra il martedì e il giovedì sera",
                "il martedì sera sono impegnato con lo sport",
                "riunione di lavoro con il team lunedì mattina",
            ]
        )
        elapsed = int((_time.perf_counter() - started) * 1000)
        near = _cosine(vectors[0], vectors[1])
        far = _cosine(vectors[0], vectors[2])
        spec = registry.get_spec(registry.EMBED)
        flag = OK if near > far else BAD
        if near <= far:
            failures += 1
        total_ms += elapsed
        print(
            f"  {flag} {'embed':<22} {spec.model:<16} {spec.device:<4} "
            f"{len(vectors):>4} {f'dim {len(vectors[0])}':>14} {elapsed:>7} ms"
        )
        if args.verbose:
            print(f"{DIM}      -> simile={near:.3f} diverso={far:.3f} (simile deve vincere){RESET}")
    except LLMError as exc:
        failures += 1
        print(f"  {BAD} {'embed':<22} {exc}")

    gpu = _gpu_memory()
    print(f"\n  totale {total_ms / 1000:.1f}s" + (f"   GPU: {gpu}" if gpu else ""))

    trace_count = get_db().scalar("SELECT count(*) FROM traces", default=0)
    print(f"  {OK} {trace_count} trace registrate in tabella traces")

    if failures:
        print(f"  {BAD} {failures} task falliti")
    return 1 if failures else 0


def cmd_traces(args: argparse.Namespace) -> int:
    rows = get_db().query(
        "SELECT created_at, task, model, device, tokens_in, tokens_out, latency_ms, ok"
        " FROM traces ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (args.limit,),
    )
    if not rows:
        print("Nessuna trace registrata.")
        return 0
    print(f"{DIM}  {'quando':<20} {'task':<22} {'modello':<16} {'dev':<4} {'in/out':>10} {'ms':>7}{RESET}")
    for r in rows:
        mark = OK if r["ok"] else BAD
        io = f"{r['tokens_in'] or 0}/{r['tokens_out'] or 0}"
        print(
            f"  {mark} {r['created_at'][:19]:<18} {r['task']:<22} {r['model']:<16} "
            f"{r['device']:<4} {io:>10} {r['latency_ms'] or 0:>7}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="donna", description="Donna — CLI di sviluppo")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING...")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="applica le migrazioni").set_defaults(func=cmd_init_db)
    sub.add_parser("doctor", help="verifica ambiente, modelli e configurazione").set_defaults(
        func=cmd_doctor
    )

    smoke = sub.add_parser("smoke", help="una chiamata reale per ogni task del registry")
    smoke.add_argument("-v", "--verbose", action="store_true", help="mostra anche gli output")
    smoke.set_defaults(func=cmd_smoke)

    traces = sub.add_parser("traces", help="ultime chiamate LLM registrate")
    traces.add_argument("-n", "--limit", type=int, default=20)
    traces.set_defaults(func=cmd_traces)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level or "WARNING")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrotto")
        return 130


if __name__ == "__main__":
    sys.exit(main())
