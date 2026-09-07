"""Graph manifest generators for UI visualization and telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from neosyntropy.core.graph import FSM
from neosyntropy.cloud.monitor._manifest import structure_hash
from neosyntropy.cloud.monitor.node.manifest import _node_structure

if TYPE_CHECKING:
    from neosyntropy.tools.core.registry import ToolRegistry


def tool_catalog(tools: ToolRegistry | Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Serialize registered tools for console inspection (no handlers)."""
    if tools is None:
        return []
    registered = getattr(tools, "tools", tools)
    if not isinstance(registered, Mapping):
        return []
    catalog: list[dict[str, Any]] = []
    for key, item in registered.items():
        name = getattr(item, "name", None) or (key if isinstance(key, str) else None)
        if not name:
            continue
        catalog.append(
            {
                "name": name,
                "description": getattr(item, "description", "") or "",
                "input_schema": getattr(item, "json_schema", None)
                or getattr(item, "input_schema", None)
                or {},
                "output_schema": getattr(item, "return_schema", None)
                or getattr(item, "output_schema", None),
            }
        )
    return catalog


def _graph_decorator(graph: FSM) -> str | None:
    decorator = getattr(graph, "decorator", None)
    if isinstance(decorator, str) and decorator.strip():
        return decorator.strip()
    return None


def _graph_node_entry(
    item: Any,
    *,
    control: bool = False,
) -> dict[str, Any]:
    """Compose a graph node entry from the node's own manifest.

    Graph-only fields such as ``prerequisites`` are added for the control wire.
    """
    entry = _node_structure(item)
    entry.pop("schema_version", None)
    if control:
        entry["prerequisites"] = list(item.prerequisites)
    return entry


def _router_providers(graph: FSM) -> dict[str, str]:
    """Map semantic router state id → backend provider id for control inference."""
    providers: dict[str, str] = {}
    for router in graph.routers.values():
        provider = getattr(router, "provider", None)
        if isinstance(provider, str) and provider.strip():
            state_id = getattr(router, "router_state_id", None) or router.id
            providers[state_id] = provider.strip()
    return providers


def _target_detail(target: Any) -> dict[str, str]:
    type_name = type(target).__name__
    if isinstance(target, str):
        return {"target": target, "target_kind": "node"}
    if type_name == "Group":
        return {"target": str(target.name), "target_kind": "group"}
    if type_name in {"SemanticRouter", "DeterministicRouter"}:
        return {"target": str(target.id), "target_kind": "router"}
    return {"target": str(getattr(target, "id", target)), "target_kind": "node"}


def _router_detail(router: Any) -> dict[str, Any]:
    """Serialize an authored router declaration without executable objects."""
    state_id = str(getattr(router, "router_state_id", router.id))
    if type(router).__name__ == "SemanticRouter":
        fallback = getattr(router, "fallback_node", None)
        return {
            "id": router.id,
            "type": "semantic",
            "state_id": state_id,
            "description": getattr(router, "description", "") or "",
            "group": getattr(router, "group", None),
            "input_schema": getattr(router, "json_schema", None),
            "reasoning": getattr(router, "reasoning", "low"),
            "category": getattr(router, "category", "general"),
            "provider": getattr(router, "provider", "neosyntropy/base"),
            "prompt": getattr(router, "prompt", "") or "",
            "tools": list(getattr(router, "tools", ()) or ()),
            "routes": {
                label: _target_detail(target)
                for label, target in getattr(router, "routes", {}).items()
            },
            "fallback": _target_detail(fallback) if fallback is not None else None,
        }
    return {
        "id": router.id,
        "type": "deterministic",
        "state_id": state_id,
        "description": getattr(router, "description", "") or "",
        "group": getattr(router, "group", None),
        "input_schema": getattr(router, "json_schema", None),
        "rules": [
            {"index": index, **_target_detail(target)}
            for index, (_, target) in enumerate(getattr(router, "rules", ()) or ())
        ],
    }


def _group_detail(group: Any) -> dict[str, Any]:
    return {
        "name": group.name,
        "description": getattr(group, "description", "") or "",
        "metadata": dict(getattr(group, "metadata", {}) or {}),
        "entry": group.entry_id() if hasattr(group, "entry_id") else None,
        "parent": getattr(group, "parent", None),
        "namespace": bool(getattr(group, "_namespace", False)),
    }


def _graph_manifest_structure(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the visualization structure before attaching code references."""
    return {
        "schema_version": 3,
        "entry": graph.entry_id,
        "input_schema": graph.input_schema,
        "nodes": [
            *[_graph_node_entry(item) for item in graph.nodes.values()],
            *[
                {
                    "id": getattr(router, "router_state_id", router.id),
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": getattr(router, "prompt", None) or None,
                    "provider": getattr(router, "provider", "neosyntropy/base"),
                    "prerequisites": [],
                    "mode": None,
                    "kind": "router",
                    "tools": list(getattr(router, "tools", ()) or ()),
                    "input_schema": getattr(router, "json_schema", None),
                    "output_schema": None,
                    "group": getattr(router, "group", None),
                    "is_fallback": False,
                    "metadata": {},
                }
                for router in graph.routers.values()
            ],
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "target_kind": edge.target_kind,
                "description": edge.description,
            }
            for edge in graph.edges
        ],
        "groups": [_group_detail(group) for group in graph.groups.values()],
        "routers": sorted(graph.router_ids),
        "routers_detail": [_router_detail(router) for router in graph.routers.values()],
        "router_providers": _router_providers(graph),
        "tools": tool_catalog(tools),
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
        "decorator": _graph_decorator(graph),
    }


def _control_graph_manifest_structure(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Graph definition for backend-owned control runs (+ console display)."""
    groups = [_group_detail(group) for group in graph.groups.values()]
    return {
        "schema_version": 3,
        "entry": graph.entry_id,
        "input_schema": graph.input_schema,
        "nodes": [
            *[_graph_node_entry(item, control=True) for item in graph.nodes.values()],
            *[
                {
                    "id": getattr(router, "router_state_id", router.id),
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": getattr(router, "prompt", None) or None,
                    "provider": getattr(router, "provider", "neosyntropy/base"),
                    "mode": None,
                    "kind": "router",
                    "tools": list(getattr(router, "tools", ()) or ()),
                    "prerequisites": [],
                    "is_fallback": False,
                    "group": getattr(router, "group", None),
                    "input_schema": getattr(router, "json_schema", None),
                    "output_schema": None,
                    "metadata": {},
                }
                for router in graph.routers.values()
            ],
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "target_kind": edge.target_kind,
                "description": edge.description,
            }
            for edge in graph.edges
        ],
        "groups": groups,
        "routers": sorted(graph.router_ids),
        "routers_detail": [_router_detail(router) for router in graph.routers.values()],
        "router_providers": _router_providers(graph),
        "tools": tool_catalog(tools),
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
        "decorator": _graph_decorator(graph),
    }


def graph_manifest(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a structure-only visualization/telemetry manifest."""
    payload = _graph_manifest_structure(graph, tools)
    payload["structure_hash"] = structure_hash(payload)
    return payload


def control_graph_manifest(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a structure-only control manifest."""
    payload = _control_graph_manifest_structure(graph, tools)
    payload["structure_hash"] = structure_hash(payload)
    return payload


__all__ = [
    "control_graph_manifest",
    "graph_manifest",
    "tool_catalog",
]
