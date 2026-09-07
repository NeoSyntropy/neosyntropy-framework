from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neosyntropy import FSM, SchemaNode, ToolRegistry, edge_deterministic, edge_fallback
from neosyntropy.control.manager import ControlManager
from neosyntropy.core.node.base import Node


def _remote_handler(ctx: Any) -> Any:
    return ctx.result(output={"ok": True})


@pytest.mark.asyncio
async def test_structure_then_code_then_recovery_publication(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    node = Node(
        id="Handler",
        handler=_remote_handler,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    fallback = SchemaNode(
        id="Fallback",
        prompt="Fallback.",
        is_fallback=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    graph = FSM(
        entry=node,
        nodes=[node, fallback],
        edges=[
            edge_deterministic("Handler", "End"),
            edge_fallback("Handler", "Fallback"),
        ],
    )
    order: list[str] = []
    monitored: dict[str, Any] = {}
    recovered: dict[str, Any] = {}
    uploaded: dict[str, bytes] = {}

    class Backend:
        async def register_graph_structure(
            self, manifest: dict[str, Any]
        ) -> dict[str, str]:
            order.append("structure")
            monitored.update(manifest)
            return {"id": "graph-1"}

        async def upload_code_bundles(self, bundles: dict[str, bytes]) -> None:
            assert bundles
            uploaded.update(bundles)
            order.append("upload")

        async def publish_graph_recovery(
            self, graph_id: str, manifest: dict[str, Any]
        ) -> dict[str, str]:
            assert graph_id == "graph-1"
            order.append("recovery")
            recovered.update(manifest)
            return {"id": graph_id}

        async def get_graph_snapshot(self, graph_id: str) -> dict[str, Any]:
            assert graph_id == "graph-1"
            order.append("snapshot")
            return {
                "graph": {
                    "id": graph_id,
                    "project_id": "project-1",
                    "manifest": monitored,
                    "recovery_manifest": recovered,
                },
                "artifacts": [
                    {"artifact": {"id": digest, "sha256": digest}}
                    for digest in uploaded
                ],
                "bundles": uploaded,
            }

    class Observer:
        async def run_started(self, **kwargs: Any) -> str:
            order.append("observer")
            assert kwargs["manifest"] == monitored
            return "run-1"

    manager = ControlManager.__new__(ControlManager)
    manager.graph = graph
    manager.tools = ToolRegistry()
    manager._backend = Backend()
    manager._monitor_backend = manager._backend
    manager._monitor_enabled = True
    manager._remote_execution_enabled = True
    manager.observer = Observer()
    manager.telemetry_timeout = 1.0
    manager._run_input = lambda context: {}

    run_id = await manager._observation_started(
        SimpleNamespace(request_id="request-1", current_state="Handler")
    )

    assert run_id == "run-1"
    assert order == ["structure", "upload", "recovery", "snapshot", "observer"]
    assert "code_artifacts" not in monitored
    assert "implementation_ref" not in str(monitored)
    assert recovered["code_artifacts"]
    assert recovered["structure_hash"] == monitored["structure_hash"]
    assert recovered["revision"] != recovered["structure_hash"]
    assert "def _remote_handler" not in str(monitored)
    assert (tmp_path / ".neosyntropy/project-1/graphs/graph-1/recovery.json").is_file()


@pytest.mark.asyncio
async def test_unrecoverable_graph_is_not_registered() -> None:
    node = Node(
        id="Builtin",
        handler=len,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    fallback = SchemaNode(
        id="Fallback",
        prompt="Fallback.",
        is_fallback=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    graph = FSM(
        entry=node,
        nodes=[node, fallback],
        edges=[
            edge_deterministic("Builtin", "End"),
            edge_fallback("Builtin", "Fallback"),
        ],
    )

    class Backend:
        async def register_graph_structure(
            self, manifest: dict[str, Any]
        ) -> dict[str, str]:
            return {"id": "graph-1"}

        async def upload_code_bundles(self, bundles: dict[str, bytes]) -> None:
            raise AssertionError("unrecoverable bundles must not be uploaded")

        async def publish_graph_recovery(
            self, graph_id: str, manifest: dict[str, Any]
        ) -> dict[str, str]:
            raise AssertionError("unrecoverable graph must not be published")

    class Observer:
        async def run_started(self, **kwargs: Any) -> str:
            raise AssertionError("unrecoverable graph must not be registered")

    manager = ControlManager.__new__(ControlManager)
    manager.graph = graph
    manager.tools = ToolRegistry()
    manager._backend = Backend()
    manager._monitor_backend = manager._backend
    manager._monitor_enabled = True
    manager._remote_execution_enabled = True
    manager.observer = Observer()
    manager.telemetry_timeout = 1.0
    manager._run_input = lambda context: {}

    with pytest.raises(RuntimeError, match="cannot be registered for remote recovery"):
        await manager._observation_started(
            SimpleNamespace(request_id="request-1", current_state="Builtin")
        )


@pytest.mark.asyncio
async def test_monitor_only_registers_pure_structure() -> None:
    node = SchemaNode(
        id="Only",
        prompt="Only.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    fallback = SchemaNode(
        id="Fallback",
        prompt="Fallback.",
        is_fallback=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    graph = FSM(
        entry=node,
        nodes=[node, fallback],
        edges=[
            edge_deterministic("Only", "End"),
            edge_fallback("Only", "Fallback"),
        ],
    )
    captured: dict[str, Any] = {}

    class Backend:
        async def register_graph_structure(
            self, manifest: dict[str, Any]
        ) -> dict[str, str]:
            captured.update(manifest)
            return {"id": "graph-1"}

    manager = ControlManager.__new__(ControlManager)
    manager.graph = graph
    manager.tools = ToolRegistry()
    manager._backend = None
    manager._monitor_backend = Backend()
    manager._monitor_enabled = True
    manager._remote_execution_enabled = False
    manager.observer = None
    manager.telemetry_timeout = 1.0
    manager._run_input = lambda context: {}

    run_id = await manager._observation_started(
        SimpleNamespace(request_id="request-1", current_state="Only")
    )

    assert run_id is None
    assert captured["structure_hash"]
    assert "code_artifacts" not in captured
    assert "runtime_compat" not in captured
