"""Structure-only node manifests for UI visualisation and telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neosyntropy.monitor._manifest import structure_hash

if TYPE_CHECKING:
    from neosyntropy.core.node import Node


def _node_structure(node: Node) -> dict[str, Any]:
    """Serialize a node without any code or bundle bytes."""
    return {
        "schema_version": 3,
        "id": node.id,
        "name": node.name,
        "description": node.description,
        "kind": node.kind,
        "mode": node.mode,
        "prompt": node.prompt,
        "provider": node.provider,
        "prerequisites": list(node.prerequisites),
        "tools": list(node.tools) if node.tools else [],
        "input_schema": node.input_schema,
        "output_schema": node.output_schema,
        "group": node.group,
        "is_fallback": node.is_fallback,
        "metadata": {k: v for k, v in (node.metadata or {}).items() if k != "source_code"},
    }


def node_manifest(node: Node) -> dict[str, Any]:
    """Return the structure-only telemetry manifest for a single node."""
    payload = _node_structure(node)
    payload["structure_hash"] = structure_hash(payload)
    return payload


__all__ = ["node_manifest"]
