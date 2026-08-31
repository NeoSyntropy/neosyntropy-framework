"""Node manifest generator for UI visualisation and telemetry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neosyntropy.core.node import Node


def node_manifest(node: "Node") -> dict[str, Any]:
    """Return the serialisable description of a single Node.

    Mirrors the per-node entries inside ``graph_manifest`` but is emitted
    independently so that a node can be observed on its own (e.g. for
    fine-tuning status, eval samples, or standalone function nodes).
    """
    return {
        "schema_version": 1,
        "id": node.id,
        "name": node.name,
        "description": node.description,
        "kind": node.kind,
        "mode": node.mode,
        "prompt": node.prompt,
        "tools": list(node.tools) if node.tools else [],
        "input_schema": node.input_schema,
        "output_schema": node.output_schema,
        "group": node.group,
        "is_fallback": node.is_fallback,
        "metadata": {
            k: v
            for k, v in (node.metadata or {}).items()
            if k != "source_code"
        },
    }
