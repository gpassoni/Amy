"""Entry point for the eval suites: python -m donna.eval.cli run"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from donna.eval import run as runner
from donna.logging_setup import setup_logging
from donna.store.db import get_db


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="donna.eval")
    parser.add_argument("command", choices=["run", "export"])
    parser.add_argument("-s", "--suite", choices=["triage", "extraction", "all"], default="all")
    parser.add_argument("-o", "--out", default="datasets/eval.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    get_db().migrate()

    suites = []
    if args.suite in ("triage", "all"):
        suites.append(runner.run_triage_suite())
    if args.suite in ("extraction", "all"):
        suites.append(runner.run_extraction_suite())

    failed_total = 0
    for suite in suites:
        print(f"\n  {suite.suite}: {suite.passed}/{suite.total}  (media {suite.mean_latency_ms} ms)")
        for outcome in suite.outcomes:
            mark = "  ok " if outcome.passed else "  BAD"
            if outcome.passed and not args.verbose:
                print(f"{mark} {outcome.name:<26} {outcome.detail}")
            else:
                print(f"{mark} {outcome.name:<26} {outcome.detail}")
                if args.verbose:
                    print(f"       {outcome.produced}")
        failed_total += suite.total - suite.passed

    if args.command == "export":
        written = runner.export_jsonl(suites, Path(args.out))
        print(f"\n  {written} righe scritte in {args.out}")

    print(f"\n  totale: {sum(s.passed for s in suites)}/{sum(s.total for s in suites)}")
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
