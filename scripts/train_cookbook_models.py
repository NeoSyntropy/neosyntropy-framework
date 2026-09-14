"""Run every API cookbook and train per-node models from those agent runs.

Each cookbook ``main()`` is treated like an Agno ``agent.run()`` session: the
FSM trace is captured, split into per-node eval samples, criticized, accepted,
and (when the backend has enough gold labels) sent to ``POST .../tune``.

Usage::

    NEOSYNTROPY_API_KEY=nsk_... \\
    NEOSYNTROPY_API_URL=https://api.neosyntropy.com \\
    NEOSYNTROPY_PROVIDER=gemini-2.5-flash \\
    python scripts/train_cookbook_models.py

    python scripts/train_cookbook_models.py cookbook/fsm/schema_node_example.py
    python scripts/train_cookbook_models.py --synthetic 24 --until-eligible
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neosyntropy.benchmark.cookbook_train import (  # noqa: E402
    REMOTE_COOKBOOKS,
    CookbookTrainReport,
    train_cookbooks,
)


def _load_tests_env() -> None:
    env_path = ROOT / "tests" / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _report_dict(report: CookbookTrainReport) -> dict[str, object]:
    return {
        "cookbook": report.cookbook,
        "project_id": report.project_id,
        "runs": report.runs,
        "error": report.error,
        "nodes": [
            {
                "node_id": node.node_id,
                "uploaded": node.uploaded,
                "synthesized": node.synthesized,
                "criticized": node.criticized,
                "accepted": node.accepted,
                "sample_count": node.sample_count,
                "required": node.required,
                "eligible": node.eligible,
                "tuned": node.tuned,
                "tune_job_id": node.tune_job_id,
                "skipped": node.skipped,
            }
            for node in report.nodes
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train NeoSyntropy nodes from cookbook FSM / agent runs."
    )
    parser.add_argument(
        "cookbooks",
        nargs="*",
        help="Cookbook paths relative to the repo root. Default: all remote API cookbooks.",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=int(os.environ.get("NEOSYNTROPY_TRAIN_SYNTHETIC", "8")),
        help="Extra labeled pairs to synthesize per node from the real-run seeds.",
    )
    parser.add_argument(
        "--until-eligible",
        action="store_true",
        help="Keep synthesizing until tune-status.required is reached (often 200).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Upload and label samples without starting a tune job.",
    )
    parser.add_argument(
        "--critic-model",
        default=os.environ.get("NEOSYNTROPY_PROVIDER", "gemini-2.5-flash"),
        help="Foundation model used for critic + teacher synthesis.",
    )
    args = parser.parse_args()
    _load_tests_env()
    os.environ.setdefault("NEO_REMOTE_EXECUTION", "TRUE")

    cookbooks = tuple(args.cookbooks) or REMOTE_COOKBOOKS
    reports = asyncio.run(
        train_cookbooks(
            cookbooks,
            synthetic=max(0, args.synthetic),
            until_eligible=args.until_eligible,
            tune=not args.no_tune,
            critic_model=args.critic_model,
        )
    )

    print("\n=== Train summary ===")
    for report in reports:
        status = report.error or f"{report.runs} runs, {len(report.nodes)} nodes"
        print(f"{report.cookbook}: {status}")
        for node in report.nodes:
            extra = node.skipped or node.tune_job_id or ""
            print(
                f"  {node.node_id}: accepted={node.accepted} "
                f"count={node.sample_count}/{node.required} "
                f"eligible={node.eligible} {extra}".rstrip()
            )
    print(json.dumps([_report_dict(report) for report in reports], indent=2))
    return 1 if any(report.error for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
