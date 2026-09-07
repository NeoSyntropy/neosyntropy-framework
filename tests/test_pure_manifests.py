from __future__ import annotations

from typing import Any

from neosyntropy import FSM, SchemaNode, edge_deterministic, edge_fallback
from neosyntropy.core.node.base import Node
from neosyntropy.cloud.monitor.function.manifest import function_manifest
from neosyntropy.cloud.monitor.graph.manifest import control_graph_manifest, graph_manifest
from neosyntropy.cloud.monitor.node.manifest import node_manifest
from neosyntropy.cloud.remote import graph_manifest_with_bundles, node_manifest_with_bundles


def _handler(ctx: Any) -> Any:
    return ctx.result(output={"ok": True})


def _graph() -> tuple[FSM, Node]:
    node = Node(
        id="Handler",
        handler=_handler,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        metadata={"source_code": "must-not-leak", "label": "safe"},
    )
    fallback = SchemaNode(
        id="Fallback",
        prompt="Fallback.",
        is_fallback=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    return (
        FSM(
            entry=node,
            nodes=[node, fallback],
            edges=[
                edge_deterministic("Handler", "End"),
                edge_fallback("Handler", "Fallback"),
            ],
        ),
        node,
    )


def _assert_structure_only(manifest: dict[str, Any]) -> None:
    forbidden = {
        "code_artifacts",
        "dependency_lock",
        "guard_ref",
        "implementation",
        "implementation_ref",
        "runtime_compat",
        "source_code",
        "vfs",
    }
    assert manifest["structure_hash"]
    assert not forbidden.intersection(str(key) for key in manifest)
    assert not any(f'"{key}"' in str(manifest) for key in forbidden)


def test_monitor_manifests_are_structure_only() -> None:
    graph, node = _graph()

    _assert_structure_only(graph_manifest(graph))
    _assert_structure_only(control_graph_manifest(graph))
    _assert_structure_only(node_manifest(node))


def test_remote_manifests_own_artifact_composition() -> None:
    graph, node = _graph()

    graph_remote = graph_manifest_with_bundles(graph)
    node_remote = node_manifest_with_bundles(node)

    assert graph_remote.manifest["code_artifacts"]
    assert graph_remote.manifest["nodes"][0]["implementation_ref"]
    assert graph_remote.bundles
    assert node_remote.manifest["code_artifacts"]
    assert node_remote.manifest["implementation_ref"]
    assert node_remote.bundles


def test_function_manifest_omits_source_code() -> None:
    manifest = function_manifest(_handler)

    assert "source_code" not in manifest
    assert "def _handler" not in str(manifest)
