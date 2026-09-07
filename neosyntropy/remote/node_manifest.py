"""Remote node manifests with executable artifact references."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from neosyntropy.monitor.node.manifest import node_manifest

from .bundles import dependency_lock, recovery_revision, runtime_compatibility
from .extractor import extract_graph_code
from .schemas import ManifestBundle

if TYPE_CHECKING:
    from neosyntropy.core.node import Node
    from neosyntropy.tools.core.registry import ToolRegistry


def node_manifest_with_bundles(
    node: Node, *, tool_registry: ToolRegistry | None = None
) -> ManifestBundle:
    """Build a node remote manifest and local content-addressed bundles."""
    extraction = extract_graph_code(
        SimpleNamespace(nodes={node.id: node}, routers={}, edges=[]),
        tool_registry,
    )
    payload = node_manifest(node)
    implementation = extraction["node_implementations"].get(node.id)
    if implementation:
        payload["implementation"] = implementation
        if implementation.get("artifact_ref"):
            payload["implementation_ref"] = implementation["artifact_ref"]
    payload["code_artifacts"] = extraction["code_artifacts"]
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
    return ManifestBundle(payload, extraction["bundles"])


__all__ = ["node_manifest_with_bundles"]
