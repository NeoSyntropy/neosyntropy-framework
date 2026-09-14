"""Collect cookbook ``FSM.run`` traces and train per-node models.

Cookbook scripts are ordinary agent runs: they create a project, execute the
graph, and return per-node outputs. This module intercepts those runs (the
same way an Agno ``agent.run()`` session is captured), writes eval samples,
criticizes them, accepts gold labels, and starts a tune job when eligible.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from ..backend import BackendClient, BackendError, Client
from ..core.graph import FSM
from .from_run import samples_from_run, trainable_node_ids
from .synthesizer import FSMSynthesizer

ROOT = Path(__file__).resolve().parents[2]

REMOTE_COOKBOOKS = (
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

_DEFAULT_CRITIC_MODEL = "gemini-2.5-flash"


@dataclass
class CapturedRun:
    fsm: FSM
    result: Any
    client: Client | None = None
    project_id: str | None = None


@dataclass
class NodeTrainReport:
    node_id: str
    uploaded: int = 0
    synthesized: int = 0
    criticized: int = 0
    accepted: int = 0
    sample_count: int = 0
    required: int = 0
    eligible: bool = False
    tuned: bool = False
    tune_job_id: str | None = None
    skipped: str | None = None


@dataclass
class CookbookTrainReport:
    cookbook: str
    project_id: str | None = None
    runs: int = 0
    nodes: list[NodeTrainReport] = field(default_factory=list)
    error: str | None = None


def load_cookbook_module(relative: str, *, root: Path = ROOT) -> ModuleType:
    path = root / relative
    spec = importlib.util.spec_from_file_location(
        f"neosyntropy_cookbook_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load cookbook {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def capture_cookbook_runs() -> Iterator[list[CapturedRun]]:
    """Patch ``FSM._run_loop`` and ``Client.create_project`` while a cookbook runs."""
    captured: list[CapturedRun] = []
    clients: list[Client] = []
    original_run = FSM._run_loop
    original_create = Client.create_project

    def wrapped_run(self: FSM, *args: Any, **kwargs: Any) -> Any:
        result = original_run(self, *args, **kwargs)
        client = clients[-1] if clients else None
        captured.append(
            CapturedRun(
                fsm=self,
                result=result,
                client=client,
                project_id=getattr(client, "project_id", None),
            )
        )
        return result

    def wrapped_create(self: Client, *args: Any, **kwargs: Any) -> dict[str, Any]:
        project = original_create(self, *args, **kwargs)
        clients.append(self)
        return project

    FSM._run_loop = wrapped_run  # type: ignore[method-assign]
    Client.create_project = wrapped_create  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        FSM._run_loop = original_run  # type: ignore[method-assign]
        Client.create_project = original_create  # type: ignore[method-assign]


def execute_cookbook(relative: str, *, root: Path = ROOT) -> list[CapturedRun]:
    """Import and run a cookbook ``main()``, capturing every FSM execution."""
    module = load_cookbook_module(relative, root=root)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(f"{relative} has no main()")
    with capture_cookbook_runs() as captured:
        main()
    return captured


def _backend_for(captured: list[CapturedRun]) -> tuple[BackendClient, str]:
    for item in captured:
        client = item.client
        project_id = item.project_id or getattr(client, "project_id", None)
        if client is not None and project_id:
            backend = getattr(client, "_backend", None) or getattr(
                client, "_as_backend", lambda: None
            )()
            if backend is None:
                continue
            backend.project_id = str(project_id)
            return backend, str(project_id)
    raise RuntimeError("cookbook run did not bind a project")


def _chunk(items: list[dict[str, Any]], size: int = 100) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


async def _generate_labeled_pairs(
    backend: BackendClient,
    fsm: FSM,
    node_id: str,
    seeds: list[dict[str, Any]],
    count: int,
    *,
    model: str,
    scenario: str,
) -> list[dict[str, Any]]:
    """Teacher-generate schema-aware (input, output) pairs from real-run seeds."""
    if count <= 0 or not seeds:
        return []
    node = fsm.nodes.get(node_id)
    if node is None:
        return []
    synthesizer = FSMSynthesizer(fsm=fsm, client=backend, model=model)
    input_schema = node.input_schema or {"type": "object"}
    output_schema = node.output_schema or {"type": "object"}
    pair_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "output"],
        "properties": {
            "input": input_schema,
            "output": output_schema,
        },
    }
    prompt_text = getattr(node, "prompt", "") or node_id
    pairs: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(4)

    async def one(index: int) -> dict[str, Any] | None:
        seed = seeds[index % len(seeds)]
        seed_json = json.dumps(seed, indent=2, default=str)
        prompt = (
            f"Generate a realistic training pair for this AI workflow node.\n"
            f"Node task: {prompt_text!r}\n\n"
            f"Here is a real historical interaction (seed):\n"
            f"```json\n{seed_json}\n```\n\n"
            "Return NEW diverse input/output JSON. Keep the schema identical "
            "but change names, values, and scenario details.\n"
            "Return ONLY JSON with keys input and output."
        )
        async with sem:
            try:
                raw = await synthesizer._generate(prompt, schema=pair_schema)
            except Exception:
                return None
        payload = raw if isinstance(raw, dict) else None
        if payload is None:
            try:
                payload = json.loads(str(raw))
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        input_json = payload.get("input")
        output_json = payload.get("output")
        if not isinstance(input_json, dict) or output_json is None:
            return None
        if not isinstance(output_json, dict):
            output_json = {"value": output_json}
        return {
            "split": "train",
            "source": "synthetic",
            "scenario": scenario,
            "input_json": input_json,
            "ground_truth_json": output_json,
            "external_key": f"synth:{node_id}:{uuid.uuid4()}",
        }

    generated = await asyncio.gather(*[one(index) for index in range(count)])
    for item in generated:
        if item is not None:
            pairs.append(item)
    return pairs


async def _critic_and_accept(
    backend: BackendClient,
    project_id: str,
    node_id: str,
    *,
    model: str,
) -> tuple[int, int]:
    samples = await backend.pull_eval_samples(project_id, node_id)
    unlabeled = [
        str(sample.get("id"))
        for sample in samples
        if sample.get("id")
        and sample.get("status") != "discarded"
        and not isinstance(sample.get("critic_json"), dict)
    ]
    criticized = 0
    if unlabeled:
        result = await backend.critic_eval_samples(
            project_id,
            node_id,
            unlabeled,
            model=model,
            delete_bad=True,
        )
        criticized = int(result.get("reviewed") or 0)

    remaining = await backend.pull_eval_samples(project_id, node_id)
    accepted = 0
    for sample in remaining:
        sample_id = sample.get("id")
        status = sample.get("status")
        critic = sample.get("critic_json") if isinstance(sample.get("critic_json"), dict) else {}
        if not sample_id or status == "discarded":
            continue
        if status == "accepted":
            accepted += 1
            continue
        good = bool(
            critic.get("good")
            or critic.get("labeled")
            or critic.get("match")
            or critic.get("score") in (1, "1")
        )
        if not good and critic:
            continue
        try:
            await backend.accept_eval_sample(project_id, node_id, str(sample_id))
            accepted += 1
        except BackendError:
            continue
    return criticized, accepted


async def train_captured_runs(
    captured: list[CapturedRun],
    *,
    cookbook: str,
    synthetic: int = 8,
    until_eligible: bool = False,
    tune: bool = True,
    critic_model: str = _DEFAULT_CRITIC_MODEL,
    log: Callable[[str], None] | None = None,
) -> CookbookTrainReport:
    """Upload, label, and optionally tune from already-captured cookbook runs."""
    emit = log or (lambda message: print(message, flush=True))
    scenario = Path(cookbook).stem
    report = CookbookTrainReport(cookbook=cookbook, runs=len(captured))
    if not captured:
        report.error = "no FSM runs captured"
        return report

    try:
        backend, project_id = _backend_for(captured)
    except RuntimeError as exc:
        report.error = str(exc)
        return report
    report.project_id = project_id

    samples_by_node: dict[str, list[dict[str, Any]]] = {}
    fsm_by_node: dict[str, FSM] = {}
    for item in captured:
        converted = samples_from_run(
            item.result, fsm=item.fsm, scenario=scenario, source="real_run"
        )
        for node_id, samples in converted.items():
            samples_by_node.setdefault(node_id, []).extend(samples)
            fsm_by_node[node_id] = item.fsm

    trainable: set[str] = set()
    for item in captured:
        trainable |= trainable_node_ids(item.fsm)

    if not trainable:
        report.nodes.append(
            NodeTrainReport(node_id="-", skipped="no provider-backed nodes")
        )
        return report

    for node_id in sorted(trainable):
        node_report = NodeTrainReport(node_id=node_id)
        real_samples = samples_by_node.get(node_id, [])
        fsm = fsm_by_node.get(node_id) or captured[0].fsm
        extra = 0
        target_synthetic = synthetic
        if until_eligible:
            status = await backend.get_tune_status(project_id, node_id)
            required = int(status.get("required") or 200)
            have = int(status.get("sample_count") or 0) + len(real_samples)
            target_synthetic = max(target_synthetic, max(0, required - have))

        if target_synthetic:
            seeds = [
                {
                    "input": sample.get("input_json"),
                    "output": sample.get("ground_truth_json"),
                }
                for sample in real_samples
            ] or [{"input": {}, "output": {}}]
            generated = await _generate_labeled_pairs(
                backend,
                fsm,
                node_id,
                seeds,
                target_synthetic,
                model=critic_model,
                scenario=scenario,
            )
            real_samples = [*real_samples, *generated]
            extra = len(generated)

        uploaded = 0
        for batch in _chunk(real_samples):
            created = await backend.create_eval_samples(project_id, node_id, batch)
            uploaded += len(created)
        node_report.uploaded = uploaded
        node_report.synthesized = extra
        emit(
            f"  {node_id}: uploaded {uploaded} samples "
            f"({uploaded - extra} from runs, {extra} synthetic)"
        )

        criticized, accepted = await _critic_and_accept(
            backend, project_id, node_id, model=critic_model
        )
        node_report.criticized = criticized
        node_report.accepted = accepted

        status = await backend.get_tune_status(project_id, node_id)
        node_report.sample_count = int(status.get("sample_count") or 0)
        node_report.required = int(status.get("required") or 0)
        node_report.eligible = bool(status.get("eligible"))
        node_report.tuned = bool(status.get("tuned"))
        if tune and node_report.eligible and not node_report.tuned:
            try:
                job_id = await backend.start_tune_job(project_id, node_id)
                node_report.tune_job_id = job_id or None
                emit(f"  {node_id}: started tune job {job_id}")
            except BackendError as exc:
                node_report.skipped = str(exc)
                emit(f"  {node_id}: tune skipped ({exc})")
        elif tune and not node_report.eligible:
            node_report.skipped = (
                f"need {node_report.required} accepted samples, "
                f"have {node_report.sample_count}"
            )
        report.nodes.append(node_report)
    return report


async def train_cookbook(
    relative: str,
    *,
    synthetic: int = 8,
    until_eligible: bool = False,
    tune: bool = True,
    critic_model: str = _DEFAULT_CRITIC_MODEL,
    root: Path = ROOT,
    log: Callable[[str], None] | None = None,
) -> CookbookTrainReport:
    """Run one cookbook, then train every model-backed node from those traces."""
    emit = log or (lambda message: print(message, flush=True))
    emit(f"\n=== train {relative} ===")
    os.environ.setdefault("NEO_REMOTE_EXECUTION", "TRUE")
    try:
        captured = await asyncio.to_thread(execute_cookbook, relative, root=root)
    except SystemExit as exc:
        return CookbookTrainReport(
            cookbook=relative, error=f"cookbook exited: {exc}"
        )
    except Exception as exc:
        return CookbookTrainReport(cookbook=relative, error=str(exc))
    return await train_captured_runs(
        captured,
        cookbook=relative,
        synthetic=synthetic,
        until_eligible=until_eligible,
        tune=tune,
        critic_model=critic_model,
        log=emit,
    )


async def train_cookbooks(
    cookbooks: list[str] | tuple[str, ...] = REMOTE_COOKBOOKS,
    **kwargs: Any,
) -> list[CookbookTrainReport]:
    reports: list[CookbookTrainReport] = []
    for relative in cookbooks:
        reports.append(await train_cookbook(relative, **kwargs))
    return reports
