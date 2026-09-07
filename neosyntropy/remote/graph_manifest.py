"""Remote graph manifests with executable artifact references."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from neosyntropy.core.graph import FSM
from neosyntropy.monitor.graph.manifest import (
    control_graph_manifest,
    graph_manifest,
)

from .bundles import dependency_lock, recovery_revision, runtime_compatibility
from .extractor import extract_graph_code
from .schemas import ManifestBundle

if TYPE_CHECKING:
    from neosyntropy.tools.core.registry import ToolRegistry


def _attach_code_refs(payload: dict[str, Any], extraction: dict[str, Any]) -> None:
    """Attach implementation, guard, artifact, and runtime metadata in place."""
    node_links = extraction["node_implementations"]
    router_links = extraction["router_implementations"]
    for node in payload["nodes"]:
        node_id = node.get("id")
        link = (
            router_links.get(node.get("name") or node_id)
            if node.get("kind") == "router"
            else node_links.get(node_id)
        )
        if link:
            node["implementation"] = link
            if link.get("artifact_ref"):
                node["implementation_ref"] = link["artifact_ref"]

    for router in payload.get("routers_detail", []):
        link = router_links.get(router["id"])
        if link:
            router["implementation"] = link
        predicates = (link or {}).get("predicates", [])
        for rule, predicate_ref in zip(router.get("rules", []), predicates, strict=False):
            rule["predicate_ref"] = predicate_ref

    occurrences: dict[tuple[str, str, str], int] = {}
    for edge in payload["edges"]:
        key = (edge["source"], edge["target"], edge["kind"])
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        stable_key = f"{key[0]}->{key[1]}:{key[2]}:{occurrence}"
        ref = extraction["edge_guards"].get(stable_key)
        if ref:
            edge["guard_ref"] = ref

    payload["code_artifacts"] = extraction["code_artifacts"]
    recovery_issues: list[dict[str, Any]] = []

    def collect_issues(value: Any, owner: str = "graph") -> None:
        if isinstance(value, Mapping):
            current_owner = str(value.get("owner_role") or owner)
            if value.get("required") is True and value.get("publishable") is False:
                issue = {
                    "owner_role": current_owner,
                    "reason": str(value.get("reason") or "required implementation unavailable"),
                }
                recoverability = value.get("recoverability")
                if isinstance(recoverability, Mapping) and recoverability.get("action"):
                    issue["action"] = str(recoverability["action"])
                if issue not in recovery_issues:
                    recovery_issues.append(issue)
            for child in value.values():
                collect_issues(child, current_owner)
        elif isinstance(value, list):
            for child in value:
                collect_issues(child, owner)

    collect_issues(extraction["node_implementations"])
    collect_issues(extraction["router_implementations"])
    collect_issues(extraction["code_artifacts"])
    payload["recoverable"] = not recovery_issues
    if recovery_issues:
        payload["recovery_issues"] = recovery_issues

    external_imports = sorted(
        {
            imported
            for artifact in extraction["code_artifacts"]
            for imported in artifact.get("external_imports", [])
        }
    )
    payload["runtime_compat"] = runtime_compatibility()
    payload["dependency_lock"] = dependency_lock(external_imports)
    payload["revision"] = recovery_revision(payload)


def graph_manifest_with_bundles(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> ManifestBundle:
    """Build a graph remote manifest and local content-addressed bundles."""
    extraction = extract_graph_code(graph, tools)
    payload = graph_manifest(graph, tools)
    _attach_code_refs(payload, extraction)
    return ManifestBundle(payload, extraction["bundles"])


def control_graph_manifest_with_bundles(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> ManifestBundle:
    """Build a control-graph remote manifest and local bundles."""
    extraction = extract_graph_code(graph, tools)
    payload = control_graph_manifest(graph, tools)
    _attach_code_refs(payload, extraction)
    return ManifestBundle(payload, extraction["bundles"])


__all__ = [
    "control_graph_manifest_with_bundles",
    "graph_manifest_with_bundles",
]
