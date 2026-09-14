"""Turn an FSM / cookbook run into per-node eval samples.

This is the Agno ``agent.run()`` → FSM training step: one captured control
run becomes labeled ``(input, ground_truth)`` rows for every model-backed
node or semantic router that actually executed.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.graph import FSM
from ..core.node.base import Node
from ..core.routing.semantic import SemanticRouter

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


def _target_id(target: Any) -> str:
    return str(getattr(target, "id", target) or "")


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


def semantic_router_ids(fsm: FSM) -> set[str]:
    return {
        router.router_state_id
        for router in fsm.routers.values()
        if isinstance(router, SemanticRouter)
    }


def trainable_node_ids(fsm: FSM) -> set[str]:
    ids = {node.id for node in fsm.nodes.values() if node_is_trainable(node)}
    ids |= semantic_router_ids(fsm)
    return ids


def _executed_node_ids(result: Any) -> list[str]:
    executed: list[str] = []
    for step in list(getattr(result, "steps", None) or []):
        for item in getattr(step, "results", None) or []:
            node_id = str(getattr(item, "node_id", "") or "")
            if node_id:
                executed.append(node_id)
    return executed


def _router_choice(fsm: FSM, result: Any) -> dict[str, dict[str, str]]:
    executed = set(_executed_node_ids(result))
    choices: dict[str, dict[str, str]] = {}
    for router in fsm.routers.values():
        if not isinstance(router, SemanticRouter):
            continue
        chosen_id = ""
        route = ""
        for label, target in router.routes.items():
            target_id = _target_id(target)
            if target_id and target_id in executed:
                chosen_id = target_id
                route = str(label)
                break
        if not chosen_id:
            fallback_id = _target_id(router.fallback_node)
            if fallback_id and fallback_id in executed:
                chosen_id = fallback_id
                route = "fallback"
        if chosen_id:
            choices[router.router_state_id] = {
                "chosen_next_node": chosen_id,
                "route": route,
            }
    return choices


def samples_from_run(
    result: Any,
    *,
    fsm: FSM | None = None,
    scenario: str | None = None,
    source: str = "real_run",
) -> dict[str, list[dict[str, Any]]]:
    """Map one ``FSM.run`` / ``agent.run`` result to eval-sample payloads.

    Returns ``{node_id: [DatasetSampleCreate, ...]}``. Handler-only nodes are
    skipped. Semantic routers are labeled with the route that actually fired.
    """
    trainable = trainable_node_ids(fsm) if fsm is not None else None
    audit = getattr(result, "audit", None)
    run_input = _as_object(
        getattr(audit, "input", None) or getattr(result, "state", {}) or {}
    )
    request_id = str(getattr(result, "request_id", "") or "")
    steps = list(getattr(result, "steps", None) or [])
    by_node: dict[str, list[dict[str, Any]]] = {}

    def _append(node_id: str, ground_truth: dict[str, Any], suffix: str) -> None:
        if trainable is not None and node_id not in trainable:
            return
        sample: dict[str, Any] = {
            "split": "train",
            "source": source,
            "input_json": run_input,
            "ground_truth_json": ground_truth,
            "external_key": f"{request_id}:{node_id}:{suffix}",
        }
        if scenario:
            sample["scenario"] = scenario
        if request_id:
            sample["run_id"] = request_id[:64]
        by_node.setdefault(node_id, []).append(sample)

    for step in steps:
        step_index = getattr(step, "step", 0)
        for item in getattr(step, "results", None) or []:
            node_id = str(getattr(item, "node_id", "") or "")
            if not node_id:
                continue
            status = str(getattr(item, "status", "") or "")
            if status == "failed":
                continue
            output = getattr(item, "output", None)
            if output is None:
                continue
            _append(node_id, _as_object(output), str(step_index))

    if fsm is not None:
        for router_id, choice in _router_choice(fsm, result).items():
            _append(router_id, choice, "route")

    if fsm is not None:
        by_node = {
            node_id: samples
            for node_id, samples in by_node.items()
            if node_id in trainable_node_ids(fsm)
        }
    return by_node
