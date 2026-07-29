from __future__ import annotations

from typing import Any

import pytest

from neosyntropy import ControlManager, Topology

from .conftest import build_graph


class FakeControlBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._step = 0

    async def start_control_run(self, graph_manifest, request, *, category="general"):
        self.calls.append(("start", {"graph": graph_manifest, "request": request}))
        assert "handlers" not in str(graph_manifest)
        assert all("prompt" not in node for node in graph_manifest["nodes"])
        return {
            "run_id": "run-1",
            "status": "awaiting_execution",
            "current_state": "Start",
            "state": dict(request.get("state") or {}),
            "step": {"step": 0, "nodes": ["VerifyIdentity"]},
            "committed_transitions": [],
            "rejection": None,
            "completed": False,
        }

    async def submit_control_results(
        self, run_id, *, results=None, client_rejection=None
    ):
        self.calls.append(
            ("results", {"run_id": run_id, "results": results, "reject": client_rejection})
        )
        assert client_rejection is None
        assert results is not None
        assert "tool_calls" not in results[0]
        if self._step == 0:
            self._step = 1
            return {
                "run_id": run_id,
                "status": "awaiting_execution",
                "current_state": "VerifyIdentity",
                "state": {"verified": True},
                "step": {"step": 1, "nodes": ["CalculateRefund"]},
                "committed_transitions": ["Start->VerifyIdentity"],
                "rejection": None,
                "completed": False,
            }
        return {
            "run_id": run_id,
            "status": "completed",
            "current_state": "CalculateRefund",
            "state": {"verified": True, "refund_amount": 50.0},
            "step": None,
            "committed_transitions": [
                "Start->VerifyIdentity",
                "VerifyIdentity->CalculateRefund",
            ],
            "rejection": None,
            "completed": True,
        }

    async def generate(self, prompt, *, schema=None, purpose="node"):
        raise AssertionError("control path should not call generate for handlers")

    async def route(self, *args, **kwargs):
        raise AssertionError("control path must not call legacy route")

    async def select(self, *args, **kwargs):
        raise AssertionError("control path must not call legacy select")


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def run_started(self, *, request_id, initial_state, manifest):
        self.events.append("run_started")
        return "observed-run-1"

    async def event(self, run_id, event_type, payload):
        self.events.append(event_type)

    async def run_finished(self, run_id, *, status, final_state):
        self.events.append(f"run_finished:{status}:{final_state}")


def test_control_manager_uses_opaque_control_api() -> None:
    backend = FakeControlBackend()
    observer = RecordingObserver()
    manager = ControlManager(
        build_graph(), backend=backend, observer=observer  # type: ignore[arg-type]
    )
    result = manager.run(
        {
            "intent": "refund",
            "current_state": "Start",
            "state": {"requested_amount": 50.0},
        }
    )

    assert result.completed is True
    assert result.plan is None
    assert result.candidates == []
    assert result.audit.committed_transitions == [
        "Start->VerifyIdentity",
        "VerifyIdentity->CalculateRefund",
    ]
    assert result.state["verified"] is True
    assert [name for name, _ in backend.calls] == ["start", "results", "results"]
    start_payload = backend.calls[0][1]
    assert start_payload["graph"]["edges"][0]["source"] == "Start"
    assert observer.events == [
        "run_started",
        "plan_proposed",
        "step_started",
        "transition_committed",
        "step_completed",
        "step_started",
        "transition_committed",
        "step_completed",
        "run_finished:completed:CalculateRefund",
    ]


def test_offline_control_still_exposes_local_plan() -> None:
    result = ControlManager(build_graph()).run({"intent": "refund order"})
    assert result.plan is not None
    assert result.plan.topology in {
        Topology.SEQUENTIAL,
        Topology.PARALLEL,
        Topology.HYBRID,
        Topology.FALLBACK,
    }
