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


def _normalize_function_source(raw: Any) -> dict[str, Any] | None:
    """Keep only the fields the console needs to render / redeploy a function."""
    if not isinstance(raw, Mapping):
        return None
    name = raw.get("function_name") or raw.get("name")
    source = raw.get("source_code")
    if not name and not source:
        return None
    payload: dict[str, Any] = {}
    node_id = raw.get("node_id")
    if node_id:
        payload["node_id"] = node_id
    if name:
        payload["function_name"] = name
    module = raw.get("function_module")
    if module:
        payload["function_module"] = module
    if source:
        payload["source_code"] = source
    return payload


def _collect_function_sources(graph: FSM) -> list[dict[str, Any]]:
    """Promote function source from the FSM and node metadata for console + deploy."""
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    def add(raw: Any) -> None:
        item = _normalize_function_source(raw)
        if not item:
            return
        key = (item.get("function_name"), item.get("source_code"))
        if key in seen:
            return
        seen.add(key)
        sources.append(item)

    fsm_source = getattr(graph, "function_source", None)
    if isinstance(fsm_source, Mapping):
        add(fsm_source)
    elif isinstance(fsm_source, list):
        for item in fsm_source:
            add(item)

    for item in graph.nodes.values():
        meta = item.metadata or {}
        if "source_code" not in meta and "function_name" not in meta:
            continue
        add(
            {
                "node_id": item.id,
                "function_name": meta.get("function_name"),
                "function_module": meta.get("function_module"),
                "source_code": meta.get("source_code"),
            }
        )
    return sources


def _graph_decorator(graph: FSM) -> str | None:
    decorator = getattr(graph, "decorator", None)
    if isinstance(decorator, str) and decorator.strip():
        return decorator.strip()
    return None


def _node_metadata_without_source(item: Any) -> dict[str, Any]:
    """Return node metadata with source_code stripped (promoted to top-level instead)."""
    return {k: v for k, v in (item.metadata or {}).items() if k != "source_code"}


def _router_providers(graph: FSM) -> dict[str, str]:
    """Map semantic router state id → backend provider id for control inference."""
    providers: dict[str, str] = {}
    for router in graph.routers.values():
        provider = getattr(router, "provider", None)
        if isinstance(provider, str) and provider.strip():
            state_id = getattr(router, "router_state_id", None) or router.id
            providers[state_id] = provider.strip()
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
                    "metadata": _node_metadata_without_source(item),
                }
                for item in graph.nodes.values()
            ],
            *[
                {
                    "id": getattr(router, "router_state_id", router.id),
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": getattr(router, "prompt", None) or None,
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
        **(
            {"decorator": decorator}
            if (decorator := _graph_decorator(graph))
            else {}
        ),
        "function_source": _collect_function_sources(graph),
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
                    "metadata": _node_metadata_without_source(item),
                }
                for item in graph.nodes.values()
            ],
            *[
                {
                    "id": getattr(router, "router_state_id", router.id),
                    "name": router.id,
                    "description": getattr(router, "description", "") or "",
                    "prompt": getattr(router, "prompt", None) or None,
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
            }
            for edge in graph.edges
        ],
        "groups": groups,
        "routers": sorted(graph.router_ids),
        "router_providers": _router_providers(graph),
        "tools": tool_catalog(tools),
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
        **(
            {"decorator": decorator}
            if (decorator := _graph_decorator(graph))
            else {}
        ),
        "function_source": _collect_function_sources(graph),
    }
