"""Graph manifest generators for UI visualization and telemetry."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from neosyntropy.core.graph import FSM

if TYPE_CHECKING:
    from neosyntropy.tools.registry import ToolRegistry


def tool_catalog(tools: "ToolRegistry | Mapping[str, Any] | None" = None) -> list[dict[str, Any]]:
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


def _router_providers(graph: FSM) -> dict[str, str]:
    """Map semantic router id → backend provider id for control inference."""
    providers: dict[str, str] = {}
    for router in graph.routers.values():
        provider = getattr(router, "provider", None)
        if isinstance(provider, str) and provider.strip():
            providers[router.id] = provider.strip()
    return providers


def graph_manifest(
    graph: FSM,
    tools: "ToolRegistry | Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """Return the visualization graph shape safe to send off-process."""
    return {
        "schema_version": 1,
        "entry": graph.entry_id,
        "input_schema": graph.input_schema,
        "nodes": [
            *[
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "prompt": item.prompt,
                    "mode": item.mode,
                    "kind": item.kind,
                    "tools": list(item.tools),
                    "input_schema": item.input_schema,
                    "output_schema": item.output_schema,
                    "group": item.group,
                    "is_fallback": item.is_fallback,
                }
                for item in graph.nodes.values()
            ],
            *[
                {
                    "id": router.id,
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": None,
                    "mode": None,
                    "kind": "router",
                    "tools": [],
                    "input_schema": getattr(router, "json_schema", None),
                    "output_schema": None,
                    "group": getattr(router, "group", None),
                    "is_fallback": False,
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
            }
            for edge in graph.edges
        ],
        "groups": [
            {
                "name": group.name,
                **(
                    {"entry": entry}
                    if (
                        entry := (
                            group.entry_id() if hasattr(group, "entry_id") else None
                        )
                    )
                    else {}
                ),
                **({"parent": group.parent} if getattr(group, "parent", None) else {}),
            }
            for group in graph.groups.values()
        ],
        "routers": sorted(graph.router_ids),
        "router_providers": _router_providers(graph),
        "tools": tool_catalog(tools),
    }


def control_graph_manifest(
    graph: FSM,
    tools: "ToolRegistry | Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """Graph definition for backend-owned control runs (+ console display)."""
    groups: list[dict[str, Any]] = []
    for group in graph.groups.values():
        payload: dict[str, Any] = {"name": group.name}
        entry = group.entry_id() if hasattr(group, "entry_id") else None
        if entry:
            payload["entry"] = entry
        parent = getattr(group, "parent", None)
        if parent:
            payload["parent"] = parent
        groups.append(payload)
    return {
        "schema_version": 1,
        "entry": graph.entry_id,
        "input_schema": graph.input_schema,
        "nodes": [
            *[
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "prompt": item.prompt,
                    "mode": item.mode,
                    "kind": item.kind,
                    "tools": list(item.tools),
                    "prerequisites": list(item.prerequisites),
                    "is_fallback": item.is_fallback,
                    "group": item.group,
                    "input_schema": item.input_schema,
                    "output_schema": item.output_schema,
                }
                for item in graph.nodes.values()
            ],
            *[
                {
                    "id": router.id,
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": None,
                    "mode": None,
                    "kind": "router",
                    "tools": [],
                    "prerequisites": [],
                    "is_fallback": False,
                    "group": getattr(router, "group", None),
                    "input_schema": getattr(router, "json_schema", None),
                    "output_schema": None,
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
            }
            for edge in graph.edges
        ],
        "groups": groups,
        "routers": sorted(graph.router_ids),
        "router_providers": _router_providers(graph),
        "tools": tool_catalog(tools),
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
    }
