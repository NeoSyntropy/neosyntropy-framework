"""Turn an FSM / cookbook run into per-node eval samples.

This is the Agno ``agent.run()`` → FSM training step: one captured control
run becomes labeled ``(input, ground_truth)`` rows for every model-backed
node that actually executed.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.graph import FSM
from ..core.node.base import Node

TRAINABLE_KINDS = frozenset({"schema", "reasoning", "combine_part"})


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_object(value: Any) -> dict[str, Any]:
    payload = _jsonable(value)
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    return {"value": payload}


def node_is_trainable(node: Node | None) -> bool:
    """True when the node is provider-backed (not a pure Python handler)."""
    if node is None:
        return False
    kind = getattr(node, "kind", None)
    if kind == "handler":
        return False
    if kind in TRAINABLE_KINDS:
        return True
    return getattr(node, "handler", None) is None and bool(getattr(node, "prompt", ""))


def trainable_node_ids(fsm: FSM) -> set[str]:
    return {
        node.id for node in fsm.nodes.values() if node_is_trainable(node)
    }


def samples_from_run(
    result: Any,
    *,
    fsm: FSM | None = None,
    scenario: str | None = None,
    source: str = "real_run",
) -> dict[str, list[dict[str, Any]]]:
    """Map one ``FSM.run`` / ``agent.run`` result to eval-sample payloads.

    Returns ``{node_id: [DatasetSampleCreate, ...]}``. Handler-only nodes are
    skipped. Input is the run's entry payload (the user turn); ground truth is
    that node's structured output.
    """
    trainable = trainable_node_ids(fsm) if fsm is not None else None
    audit = getattr(result, "audit", None)
    run_input = _as_object(getattr(audit, "input", None) or getattr(result, "state", {}) or {})
    request_id = str(getattr(result, "request_id", "") or "")
    steps = list(getattr(result, "steps", None) or [])
    by_node: dict[str, list[dict[str, Any]]] = {}

    for step in steps:
        step_index = getattr(step, "step", 0)
        for item in getattr(step, "results", None) or []:
            node_id = str(getattr(item, "node_id", "") or "")
            if not node_id:
                continue
            if trainable is not None and node_id not in trainable:
                continue
            status = str(getattr(item, "status", "") or "")
            if status == "failed":
                continue
            output = getattr(item, "output", None)
            if output is None:
                continue
            sample: dict[str, Any] = {
                "split": "train",
                "source": source,
                "input_json": run_input,
                "ground_truth_json": _as_object(output),
                "external_key": f"{request_id}:{node_id}:{step_index}",
            }
            if scenario:
                sample["scenario"] = scenario
            if request_id:
                sample["run_id"] = request_id[:64]
            by_node.setdefault(node_id, []).append(sample)

    if fsm is not None:
        by_node = {
            node_id: samples
            for node_id, samples in by_node.items()
            if node_id in trainable_node_ids(fsm)
        }
    return by_node
