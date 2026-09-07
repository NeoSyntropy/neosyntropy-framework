"""Run every FSM cookbook and summarize locally mirrored remote graphs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / ".neosyntropy"
COOKBOOKS = (
    "cookbook/fsm/python_node_example.py",
    "cookbook/fsm/schema_node_example.py",
    "cookbook/fsm/reasoning_node_prompt_tools_example.py",
    "cookbook/fsm/reasoning_node_steps_example.py",
    "cookbook/fsm/semantic_router_parallel_example.py",
    "cookbook/fsm/semantic_router_sequential_example.py",
    "cookbook/kpi/node_kpi_example.py",
    "cookbook/kpi/group_kpi_example.py",
    "cookbook/kpi/fsm_path_kpi_example.py",
    "cookbook/validation/node_validation_example.py",
    "cookbook/validation/group_path_validation_example.py",
    "cookbook/validation/fsm_path_validation_example.py",
    "cookbook/decorators/function_calling_example.py",
    "cookbook/decorators/workflow_reasoning_example.py",
)


def _snapshots() -> set[Path]:
    if not SNAPSHOT_ROOT.is_dir():
        return set()
    return {
        graph_dir
        for project_dir in SNAPSHOT_ROOT.iterdir()
        if project_dir.is_dir()
        for graph_dir in (project_dir / "graphs").glob("*")
        if graph_dir.is_dir()
    }


def main() -> int:
    env = dict(os.environ)
    env["NEO_REMOTE_EXECUTION"] = "TRUE"
    timeout = float(env.get("NEOSYNTROPY_COOKBOOK_TIMEOUT", "300"))
    before = _snapshots()
    failures: list[str] = []

    for relative in COOKBOOKS:
        print(f"\n=== {relative} ===", flush=True)
        try:
            completed = subprocess.run(
                [sys.executable, str(ROOT / relative)],
                cwd=ROOT,
                env=env,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"Timed out after {timeout:g}s.", flush=True)
            failures.append(relative)
            continue
        if completed.returncode:
            failures.append(relative)

    after = _snapshots()
    created = sorted(after - before)
    print("\n=== Remote graph snapshots ===")
    for graph_dir in sorted(after):
        relative = graph_dir.relative_to(SNAPSHOT_ROOT)
        marker = "new" if graph_dir in created else "existing"
        artifacts = len(tuple((graph_dir / "artifacts").glob("*.json.gz")))
        print(f"{relative} [{marker}, {artifacts} artifacts]")

    print(
        f"\nCompleted {len(COOKBOOKS) - len(failures)}/{len(COOKBOOKS)} cookbooks; "
        f"{len(after)} snapshots available."
    )
    if failures:
        print("Failed cookbooks:")
        for relative in failures:
            print(f"- {relative}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
