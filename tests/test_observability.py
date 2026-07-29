from __future__ import annotations

import asyncio
import json
from typing import Any

from neosyntropy import (
    BackendClient,
    ControlManager,
    Edge,
    Graph,
    Group,
    RoutingPlan,
    Topology,
    graph_manifest,
    node,
)
from neosyntropy import backend as backend_module
from neosyntropy.observability import BackendTelemetryReporter

from .conftest import build_graph


class RecordingObserver:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    async def run_started(self, *, request_id, initial_state, manifest):
        self.records.append(
            (
                "run_started",
                {
                    "request_id": request_id,
                    "initial_state": initial_state,
                    "manifest": manifest,
                },
            )
        )
        return "telemetry-run-1"

    async def event(self, run_id, event_type, payload):
        self.records.append((event_type, dict(payload)))

    async def run_finished(self, run_id, *, status, final_state):
        self.records.append(
            ("run_finished", {"status": status, "final_state": final_state})
        )


class UnavailableObserver(RecordingObserver):
    async def event(self, run_id, event_type, payload):
        raise OSError("telemetry is offline")

    async def run_finished(self, run_id, *, status, final_state):
        raise OSError("telemetry is offline")


class StartUnavailableObserver(RecordingObserver):
    async def run_started(self, *, request_id, initial_state, manifest):
        raise OSError("telemetry is offline")


def test_graph_manifest_excludes_executable_and_sensitive_fields() -> None:
    @node(
        id="Sensitive",
        name="Visible node",
        description="SECRET description",
        prompt="SECRET prompt",
        tools=("SECRET_tool",),
        metadata={"token": "SECRET metadata"},
        group="private",
    )
    def sensitive(ctx):
        return ctx.result()

    @node(id="Fallback", is_fallback=True)
    def fallback(ctx):
        return ctx.result()

    graph = Graph(
        nodes=[sensitive, fallback],
        edges=[
            Edge(
                source="Start",
                target="Sensitive",
                label="first",
                description="SECRET edge",
                guard=lambda state: state.get("SECRET"),
            )
        ],
        groups=[Group(name="private", metadata={"SECRET": True})],
        validate_reachability=False,
    )

    manifest = graph_manifest(graph)
    encoded = json.dumps(manifest)

    assert "SECRET" not in encoded
    assert manifest["nodes"][0] == {
        "id": "Sensitive",
        "name": "Visible node",
        "group": "private",
        "is_fallback": False,
    }
    assert manifest["edges"] == [
        {"source": "Start", "target": "Sensitive", "label": "first"}
    ]
    assert manifest["groups"] == [{"name": "private"}]


def test_control_manager_reports_sanitized_lifecycle_in_order() -> None:
    observer = RecordingObserver()
    result = ControlManager(build_graph(), observer=observer).run(
        {
            "intent": "SECRET customer intent",
            "current_state": "Start",
            "state": {"SECRET": "state"},
            "metadata": {"SECRET": "metadata"},
        }
    )

    assert result.completed
    assert [event for event, _ in observer.records] == [
        "run_started",
        "plan_proposed",
        "step_started",
        "transition_committed",
        "step_completed",
        "run_finished",
    ]
    assert observer.records[-1][1] == {
        "status": "completed",
        "final_state": "VerifyIdentity",
    }
    assert "SECRET" not in json.dumps(observer.records)


def test_control_manager_succeeds_when_telemetry_is_unavailable() -> None:
    event_failure_result = ControlManager(
        build_graph(), observer=UnavailableObserver(), telemetry_timeout=0.01
    ).run({"intent": "refund", "current_state": "Start"})
    start_failure_result = ControlManager(
        build_graph(), observer=StartUnavailableObserver(), telemetry_timeout=0.01
    ).run({"intent": "refund", "current_state": "Start"})

    assert event_failure_result.completed
    assert start_failure_result.completed
    assert event_failure_result.final_state == "VerifyIdentity"
    assert start_failure_result.final_state == "VerifyIdentity"


def test_failed_execution_reports_failure_then_finish() -> None:
    @node(id="Fails")
    def fails(ctx):
        return ctx.result(status="failed", error="SECRET failure detail")

    @node(id="Fallback", is_fallback=True)
    def fallback(ctx):
        return ctx.result()

    graph = Graph(
        nodes=[fails, fallback],
        edges=[Edge(source="Start", target="Fails", label="first")],
        validate_reachability=False,
    )
    observer = RecordingObserver()

    result = ControlManager(graph, observer=observer).run(
        {"intent": "SECRET intent", "current_state": "Start"}
    )

    assert not result.completed
    assert [event for event, _ in observer.records][-3:] == [
        "step_completed",
        "run_failed",
        "run_finished",
    ]
    assert observer.records[-1][1]["status"] == "failed"
    assert "SECRET" not in json.dumps(observer.records)


def test_rejected_plan_reports_rejection_then_finish() -> None:
    class InvalidRouter:
        async def route(self, context, candidates):
            issue = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.node_id == "IssueRefund"
            )
            return RoutingPlan(
                topology=Topology.SEQUENTIAL, execution_plan=[[issue]]
            )

    observer = RecordingObserver()
    result = ControlManager(
        build_graph(), router=InvalidRouter(), observer=observer
    ).run({"intent": "refund", "current_state": "Start"})

    assert result.rejected
    assert [event for event, _ in observer.records][-2:] == [
        "run_rejected",
        "run_finished",
    ]
    assert observer.records[-1][1]["status"] == "rejected"


def test_api_key_headers_and_telemetry_endpoints(monkeypatch) -> None:
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/telemetry/runs"):
            return Response({"id": "run-123"})
        return Response({})

    monkeypatch.setattr(backend_module, "urlopen", fake_urlopen)
    client = BackendClient(
        "https://api.example.test",
        api_key="api-secret",
        project_id="project-123",
        telemetry_timeout=0.5,
    )
    reporter = BackendTelemetryReporter(client)

    run_id = asyncio.run(
        reporter.run_started(
            request_id="request-1",
            initial_state="Start",
            manifest={"schema_version": 1},
        )
    )
    asyncio.run(reporter.event(run_id or "", "plan_proposed", {"steps": []}))
    asyncio.run(reporter.event(run_id or "", "step_started", {"step": 0}))
    asyncio.run(
        reporter.run_finished(
            run_id or "", status="completed", final_state="End"
        )
    )

    assert run_id == "run-123"
    assert [request.full_url for request, _ in requests] == [
        "https://api.example.test/api/v1/telemetry/runs",
        "https://api.example.test/api/v1/telemetry/runs/run-123/events",
        "https://api.example.test/api/v1/telemetry/runs/run-123/events",
        "https://api.example.test/api/v1/telemetry/runs/run-123/finish",
    ]
    assert all(
        request.get_header("Authorization") == "Bearer api-secret"
        for request, _ in requests
    )
    assert all(
        request.get_header("X-neosyntropy-project-id") == "project-123"
        for request, _ in requests
    )
    payloads = [
        json.loads(request.data.decode())
        for request, _ in requests
    ]
    assert payloads == [
        {
            "external_id": "request-1",
            "name": "control-cycle",
            "metadata": {
                "initial_state": "Start",
                "graph": {"schema_version": 1},
            },
        },
        {
            "external_id": "run-123:1",
            "sequence": 1,
            "event_type": "plan_proposed",
            "payload": {"steps": []},
        },
        {
            "external_id": "run-123:2",
            "sequence": 2,
            "event_type": "step_started",
            "payload": {"step": 0},
        },
        {"status": "succeeded", "output": {"final_state": "End"}},
    ]


def test_access_token_remains_compatible(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"text":"ok"}'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(backend_module, "urlopen", fake_urlopen)
    client = BackendClient("https://api.example.test", "legacy-token")

    assert asyncio.run(client.generate("hello")) == "ok"
    assert captured["authorization"] == "Bearer legacy-token"


def test_telemetry_transport_failure_is_best_effort(monkeypatch) -> None:
    def unavailable(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr(backend_module, "urlopen", unavailable)
    client = BackendClient(
        "https://api.example.test",
        api_key="api-secret",
        project_id="project-123",
        telemetry_timeout=0.01,
    )

    assert (
        asyncio.run(
            client.telemetry_run_started(
                request_id="request-1",
                initial_state="Start",
                manifest={"schema_version": 1},
            )
        )
        is None
    )
