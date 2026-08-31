"""Compare NeoSyntropy vs LangGraph on Tier 1 BMAD cases.

Metrics: task accuracy (route + path + schema + required tools), wall-clock
latency, and USD (NeoSyntropy = committed transitions; LangGraph = tokens).

    python Benchmarks/tier1/compare.py
    python Benchmarks/tier1/compare.py --systems neosyntropy
    python Benchmarks/tier1/compare.py --limit 4 --json results.json

Live NeoSyntropy needs ``NEOSYNTROPY_API_KEY`` + ``NEOSYNTROPY_PROJECT_ID``.
Live LangGraph needs ``OPENAI_API_KEY`` (or ``OPENAI_BASE_URL``) and:

    pip install langgraph langchain-openai langchain-core
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol import (  # noqa: E402
    DEFAULT_LG_MODEL,
    Case,
    ScoredCase,
    SystemSummary,
    format_table,
    load_cases,
    score_trace,
    summarize,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default="neosyntropy,langgraph",
        help="Comma-separated: neosyntropy,langgraph",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    parser.add_argument("--model", default=os.getenv("LANGGRAPH_MODEL") or DEFAULT_LG_MODEL)
    parser.add_argument("--json", dest="json_path", default="", help="Write full results JSON")
    parser.add_argument(
        "--cases",
        default="",
        help="Path to cases.jsonl (default: datasets/cases.jsonl)",
    )
    return parser.parse_args(argv)


def _wanted(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _run_system(name: str, cases: list[Case], model_name: str) -> list[ScoredCase]:
    if name == "neosyntropy":
        from engines import NeoSyntropyEngine, neosyntropy_client_from_env

        client = neosyntropy_client_from_env()
        if client is None:
            print(
                "skip neosyntropy: set NEOSYNTROPY_API_KEY and NEOSYNTROPY_PROJECT_ID",
                file=sys.stderr,
            )
            return []
        engine: Any = NeoSyntropyEngine(client)
    elif name == "langgraph":
        try:
            from engines import LangGraphEngine, langgraph_llm_from_env
        except ImportError as exc:
            print(
                f"skip langgraph: {exc} (pip install langgraph langchain-openai)",
                file=sys.stderr,
            )
            return []
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
            print("skip langgraph: set OPENAI_API_KEY or OPENAI_BASE_URL", file=sys.stderr)
            return []
        engine = LangGraphEngine(langgraph_llm_from_env(model_name), model_name=model_name)
    else:
        raise SystemExit(f"unknown system {name!r}")

    scored: list[ScoredCase] = []
    for case in cases:
        print(f"[{name}] {case.id} …", flush=True)
        scored.append(score_trace(case, engine.run(case)))
    return scored


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases_path = Path(args.cases) if args.cases else None
    cases = load_cases(cases_path)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases loaded", file=sys.stderr)
        return 2

    wanted = _wanted(args.systems)
    by_system: dict[str, list[ScoredCase]] = {}
    summaries: list[SystemSummary] = []
    for name in wanted:
        rows = _run_system(name, cases, args.model)
        if not rows:
            continue
        by_system[name] = rows
        summaries.append(summarize(name, rows))

    if not summaries:
        print(
            "nothing to compare. For NeoSyntropy set API credentials; "
            "for LangGraph install extras and set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    print()
    print(format_table(summaries))
    print()
    print(
        "USD: NeoSyntropy = $0.005 tuned router hop + $0.01 per glm node hop. "
        f"LangGraph = {args.model} token list price. "
        "Accuracy = exact route/path/schema/tools; illegal hops are LangGraph labels "
        "outside the six BMAD phases."
    )
    for name, rows in by_system.items():
        failed = [row for row in rows if not row.passed]
        if failed:
            print(f"\n{name} failures:")
            for row in failed:
                print(f"  {row.case_id}: {row.reason}")

    if args.json_path:
        payload = {
            "summaries": [row.to_dict() for row in summaries],
            "cases": {
                name: [row.to_dict() for row in rows] for name, rows in by_system.items()
            },
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
