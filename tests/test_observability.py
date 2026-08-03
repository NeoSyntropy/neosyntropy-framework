from __future__ import annotations

import asyncio
import json
from typing import Any

from neosyntropy import (
    BackendClient,
    ControlManager,
    Edge,
    FSM,
    Group,
    OpenInput,
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

    async def run_started(self, *, request_id, initial_state, manifest, input=None):
        self.records.append(
            (
                "run_started",
                {
                    "request_id": request_id,
                    "initial_state": initial_state,
                    "manifest": manifest,
                    "input": input,
                },
            )
        )
        return "telemetry-run-1"

    async def event(self, run_id, event_type, payload):
        self.records.append((event_type, dict(payload)))

    async def run_finished(self, run_id, *, status, final_state, output=None):
        self.records.append(
            (
                "run_finished",
                {"status": status, "final_state": final_state, "output": output},
            )
        )


class UnavailableObserver(RecordingObserver):
    async def event(self, run_id, event_type, payload):
        raise OSError("telemetry is offline")

    async def run_finished(self, run_id, *, status, final_state, output=None):
        raise OSError("telemetry is offline")


class StartUnavailableObserver(RecordingObserver):
    async def run_started(self, *, request_id, initial_state, manifest, input=None):
        raise OSError("telemetry is offline")


def test_graph_manifest_includes_console_fields_but_excludes_executables() -> None:
    from neosyntropy import EmptyOutput, OpenInput

    @node(
        id="Sensitive",
        name="Visible node",
        description="Visible description",
        prompt="Visible prompt",
        tools=("lookup_docs",),
        input_schema=OpenInput, output_schema=EmptyOutput,
        metadata={"token": "SECRET metadata"},
        group="private",
    )
    def sensitive(ctx):
        return ctx.result(output={})

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    graph = FSM(
        nodes=[sensitive, fallback],
        edges=[
            Edge(
                source="Start",
                target="Sensitive",
                kind="deterministic",
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
        "description": "Visible description",
        "prompt": "Visible prompt",
        "mode": "reasoning",
        "tools": ["lookup_docs"],
        "input_schema": {
            "additionalProperties": True,
            "description": (
                "Permissive input for nodes that do not constrain workflow state."
            ),
            "properties": {},
            "title": "OpenInput",
            "type": "object",
        },
        "output_schema": {
            "additionalProperties": False,
            "description": (
                "Empty object schema for nodes that only update state or signal completion."
            ),
            "properties": {},
            "required": [],
            "title": "EmptyOutput",
            "type": "object",
        },
        "group": "private",
        "is_fallback": False,
    }
    assert manifest["edges"] == [
        {
            "source": "Start",
            "target": "Sensitive",
            "kind": "deterministic",
            "target_kind": "node",
        }
    ]
    assert manifest["groups"] == [{"name": "private"}]
    assert manifest["tools"] == []


def test_graph_manifest_includes_tool_catalog_and_node_output_schema() -> None:
    from pydantic import BaseModel, ConfigDict

    from neosyntropy import EmptyOutput, OpenInput, ToolRegistry, tool

    class LookupArgs(BaseModel):
        query: str

    class Reply(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str

    registry = ToolRegistry()

    @tool(registry=registry)
    def lookup_docs(args: LookupArgs) -> dict:
        """Search the approved knowledge base."""
        return {"matches": [args.query]}

    @node(
        id="Answer",
        tools=("lookup_docs",),
        input_schema=OpenInput, output_schema=Reply,
    )
    def answer(ctx):
        return ctx.result(output={"text": "ok"})

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    graph = FSM(
        nodes=[answer, fallback],
        edges=[Edge(source="Start", target="Answer")],
        validate_reachability=False,
    )
    manifest = graph_manifest(graph, registry)
    assert len(manifest["tools"]) == 1
    tool_spec = manifest["tools"][0]
    assert tool_spec["name"] == "lookup_docs"
    assert tool_spec["description"] == "Search the approved knowledge base."
    assert tool_spec["input_schema"]["properties"]["query"]["type"] == "string"
    assert tool_spec["output_schema"] == {"type": "object"}
    assert manifest["input_schema"] is None
    answer_node = next(node for node in manifest["nodes"] if node["id"] == "Answer")
    assert answer_node["mode"] == "reasoning"
    assert answer_node["output_schema"]["properties"]["text"]["type"] == "string"
    assert answer_node["output_schema"]["required"] == ["text"]
    fallback_node = next(node for node in manifest["nodes"] if node["id"] == "Fallback")
    assert fallback_node["mode"] == "schema_extraction"


def test_control_manager_reports_sanitized_lifecycle_in_order() -> None:
    observer = RecordingObserver()
    result = ControlManager(
        build_graph(), observer=observer, capture_payloads=False
    ).run(
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
        "output": None,
    }
    assert "SECRET" not in json.dumps(observer.records)


def test_control_manager_captures_run_and_step_payloads_by_default() -> None:
    observer = RecordingObserver()
    result = ControlManager(build_graph(), observer=observer).run(
        {
            "intent": "refund my order",
            "current_state": "Start",
            "state": {"requested_amount": 25.0},
            "metadata": {"channel": "email"},
        }
    )

    assert result.completed
    records = dict(observer.records)

    run_input = records["run_started"]["input"]
    assert run_input["intent"] == "refund my order"
    assert run_input["current_state"] == "Start"
    assert run_input["state"] == {"requested_amount": 25.0}
    assert run_input["metadata"] == {"channel": "email"}

    step_started = records["step_started"]
    assert step_started["input"] == {
        "current_state": "Start",
        "state": {"requested_amount": 25.0},
    }

    step_completed = records["step_completed"]
    assert step_completed["status"] == "completed"
    [node_result] = step_completed["output"]["results"]
    assert node_result["node_id"] == "VerifyIdentity"
    assert node_result["status"] == "succeeded"
    assert node_result["state_updates"] == {"verified": True}
    assert step_completed["output"]["state"]["verified"] is True

    run_output = records["run_finished"]["output"]
    assert run_output["state"]["verified"] is True
    assert run_output["committed_transitions"] == ["Start->VerifyIdentity"]


def test_rejected_step_payload_includes_rejection_reason() -> None:
    from neosyntropy import EmptyOutput, OpenInput

    @node(id="Rogue", input_schema=OpenInput, output_schema=EmptyOutput)
    def rogue(ctx):
        return ctx.result(output={}, next_state="End")

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    graph = FSM(
        nodes=[rogue, fallback],
        edges=[Edge(source="Start", target="Rogue", kind="deterministic")],
        validate_reachability=False,
    )
    observer = RecordingObserver()
    result = ControlManager(graph, observer=observer).run(
        {"intent": "anything", "current_state": "Start"}
    )

    assert result.rejected
    step_completed = dict(observer.records)["step_completed"]
    assert step_completed["status"] == "rejected"
    assert "no legal guard-allowed transition" in (step_completed["rejection"] or "")


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
    from neosyntropy import EmptyOutput, OpenInput

    @node(id="Fails", input_schema=OpenInput, output_schema=EmptyOutput)
    def fails(ctx):
        return ctx.result(status="failed", error="SECRET failure detail", output={})

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    graph = FSM(
        nodes=[fails, fallback],
        edges=[Edge(source="Start", target="Fails", kind="deterministic")],
        validate_reachability=False,
    )
    observer = RecordingObserver()

    result = ControlManager(graph, observer=observer, capture_payloads=False).run(
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
            calc = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.node_id == "CalculateRefund"
            )
            return RoutingPlan(
                topology=Topology.SEQUENTIAL, execution_plan=[[calc]]
            )

    base = build_graph()
    graph = FSM(
        nodes=list(base.nodes.values()),
        edges=[
            Edge(source="Start", target="VerifyIdentity", kind="semantic"),
            Edge(source="Start", target="CalculateRefund", kind="semantic"),
            Edge(source="VerifyIdentity", target="CalculateRefund", kind="deterministic"),
            Edge(source="CalculateRefund", target="IssueRefund", kind="deterministic"),
            Edge(source="IssueRefund", target="End", kind="deterministic"),
            Edge(source="Start", target="OutOfScope", kind="fallback"),
        ],
    )
    observer = RecordingObserver()
    result = ControlManager(
        graph, router=InvalidRouter(), observer=observer
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
            input={"intent": "hello", "state": {}},
        )
    )
    asyncio.run(reporter.event(run_id or "", "plan_proposed", {"steps": []}))
    asyncio.run(reporter.event(run_id or "", "step_started", {"step": 0}))
    asyncio.run(
        reporter.run_finished(
            run_id or "",
            status="completed",
            final_state="End",
            output={"state": {"done": True}},
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
            "input": {"intent": "hello", "state": {}},
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
        {
            "status": "succeeded",
            "output": {"final_state": "End", "state": {"done": True}},
        },
    ]


def test_oversized_event_payload_is_truncated_not_dropped() -> None:
    from neosyntropy.observability import bounded_event_payload

    big = {"step": 0, "node_ids": ["A"], "input": {"state": {"blob": "x" * 5000}}}
    bounded = bounded_event_payload(big, limit=1024)
    assert bounded["step"] == 0
    assert bounded["node_ids"] == ["A"]
    assert bounded["input"] == {
        "truncated": True,
        "reason": "payload exceeded telemetry size limit",
    }

    small = {"step": 0, "input": {"state": {}}}
    assert bounded_event_payload(small, limit=1024) is small


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
