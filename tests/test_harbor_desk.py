"""Harbor Signal Desk: grouped reasoning (str + tools) → schema extraction (JSON)."""
from __future__ import annotations

import json

import pytest

from neosyntropy import (
    OpenInput,
    ControlManager,
    REASONING_OUTPUT_SCHEMA,
    SchemaNode,
    RoutingPlan,
    RunRequest,
    Topology,
    graph_manifest,
)
from neosyntropy.routing.preferred import PreferredPathRouter

from .harbor_desk import (
    BERTH_CLEARANCE,
    BERTH_SCOUT,
    CARGO_INSPECT,
    CARGO_MANIFEST,
    FALLBACK,
    PILOT_ADVISORY,
    PILOT_BRIEF,
    build_harbor_graph,
    build_harbor_tools,
    harbor_manifest,
)


class PreferFirstRouter:
    """Force the first cycle onto a chosen lane; then follow preferred edges."""

    def __init__(self, graph, first_node_id: str):
        self.first_node_id = first_node_id
        self._inner = PreferredPathRouter(graph)
        self._used = False

    async def route(self, context, candidates):
        if not self._used:
            self._used = True
            index = next(
                i
                for i, candidate in enumerate(candidates)
                if candidate.node_id == self.first_node_id
            )
            return RoutingPlan(
                reasoning=f"Test prefers {self.first_node_id}.",
                topology=Topology.SEQUENTIAL,
                execution_plan=[[index]],
            )
        return await self._inner.route(context, candidates)


@pytest.fixture
def graph():
    return build_harbor_graph()


@pytest.fixture
def tools():
    return build_harbor_tools()


def _manager(graph, tools, *, first: str | None = None) -> ControlManager:
    kwargs: dict = {"tools": tools}
    if first is not None:
        kwargs["router"] = PreferFirstRouter(graph, first)
    return ControlManager(graph, **kwargs)


def test_reasoning_nodes_are_str_schema_with_tools(graph):
    for node_id in (BERTH_SCOUT, CARGO_INSPECT, PILOT_BRIEF):
        item = graph.nodes[node_id]
        assert item.mode == "reasoning"
        assert item.tools
        assert item.output_schema == REASONING_OUTPUT_SCHEMA
        assert item.output_schema["type"] == "string"


def test_schema_extraction_nodes_forbid_tools(graph):
    for node_id in (BERTH_CLEARANCE, CARGO_MANIFEST, PILOT_ADVISORY, FALLBACK):
        item = graph.nodes[node_id]
        assert item.mode == "schema_extraction"
        assert item.tools == ()


def test_schema_extraction_cannot_declare_tools():
    from neosyntropy import Node

    with pytest.raises(Exception, match="schema_extraction.*cannot declare tools"):
        Node(
            id="broken.Extract",
            mode="schema_extraction",
            tools=("lookup_slip",),
            input_schema=OpenInput,
            output_schema=REASONING_OUTPUT_SCHEMA,
        )
    # SchemaNode has no tools parameter — use ReasoningNode / CombineNode instead.
    schema = SchemaNode(
        id="ok.Extract",
        input_schema=OpenInput,
        output_schema=REASONING_OUTPUT_SCHEMA,
        prompt="extract",
    )
    assert schema.mode == "schema_extraction"
    assert schema.tools == ()


def test_groups_pair_reasoning_then_extraction(graph):
    by_group: dict[str, list[str]] = {}
    for item in graph.nodes.values():
        if item.group:
            by_group.setdefault(item.group, []).append(item.id)
    assert set(by_group) == {"berth", "cargo", "pilot"}
    for group, ids in by_group.items():
        modes = {graph.nodes[node_id].mode for node_id in ids}
        assert modes == {"reasoning", "schema_extraction"}, group


def test_manifest_exposes_modes_and_tool_catalog():
    manifest = harbor_manifest()
    modes = {node["id"]: node["mode"] for node in manifest["nodes"]}
    assert modes[BERTH_SCOUT] == "reasoning"
    assert modes[BERTH_CLEARANCE] == "schema_extraction"
    assert modes[CARGO_INSPECT] == "reasoning"
    assert modes[CARGO_MANIFEST] == "schema_extraction"
    assert {tool["name"] for tool in manifest["tools"]} == {
        "lookup_slip",
        "weigh_crate",
        "tide_chart",
    }
    assert {group["name"] for group in manifest["groups"]} == {
        "berth",
        "cargo",
        "pilot",
    }


def _walk(manager: ControlManager, *, intent: str, state: dict, start: str = "Start"):
    """Run sequential cycles until End or rejection."""
    current, prior, snapshots = start, [], []
    for _ in range(5):
        result = manager.run(
            RunRequest(
                intent=intent,
                current_state=current,
                state=state,
                prior_executions=prior,
            )
        )
        assert not result.rejected, result.rejection
        snapshots.append(result)
        state = result.state
        current = result.final_state
        prior = prior + [
            {
                "node_id": item.node_id,
                "status": item.status,
                "output": item.output,
                "state_updates": item.state_updates,
            }
            for step in result.steps
            for item in step.results
        ]
        if current == "End":
            break
    return snapshots, state, current


def test_berth_lane_reasoning_str_then_json_clearance(graph, tools):
    manager = _manager(graph, tools, first=BERTH_SCOUT)
    results, state, final = _walk(
        manager,
        intent="dock the Aurora at first light",
        state={"vessel": "Aurora"},
    )
    assert results[0].final_state == BERTH_SCOUT
    scout = results[0].steps[0].results[0]
    assert isinstance(scout.output, str)
    assert "B-12" in scout.output
    assert state["reasoning_text"] == scout.output
    assert state["tool_evidence"][0]["tool"] == "lookup_slip"

    assert results[1].final_state == BERTH_CLEARANCE or final == "End"
    assert final == "End"
    clearance = results[1].steps[0].results[0].output
    assert clearance["status"] == "cleared"
    assert clearance["slip"] == "B-12"
    assert clearance["evidence_tools"] == ["lookup_slip"]
    assert "B-12" in clearance["guest_text"]
    assert isinstance(clearance, dict)
    json.dumps(clearance)


def test_cargo_lane_quarantines_hazard_crate(graph, tools):
    manager = _manager(graph, tools, first=CARGO_INSPECT)
    results, state, final = _walk(
        manager,
        intent="weigh inbound crate",
        state={"crate_id": "CR-9"},
    )
    assert results[0].final_state == CARGO_INSPECT
    assert isinstance(results[0].steps[0].results[0].output, str)
    assert "HAZARD" in results[0].steps[0].results[0].output
    assert final == "End"
    manifest = results[1].steps[0].results[0].output
    assert manifest["disposition"] == "quarantine"
    assert manifest["crate_id"] == "CR-9"
    assert manifest["evidence_tools"] == ["weigh_crate"]
    assert state["lane"] == "cargo"


def test_pilot_lane_waits_on_strong_current(graph, tools):
    manager = _manager(graph, tools, first=PILOT_BRIEF)
    results, _, final = _walk(
        manager,
        intent="need north channel advice",
        state={"channel": "north"},
    )
    assert results[0].final_state == PILOT_BRIEF
    assert final == "End"
    advisory = results[1].steps[0].results[0].output
    assert results[1].steps[0].results[0].node_id == PILOT_ADVISORY
    assert advisory["advice"] == "wait"
    assert advisory["channel"] == "north"
    assert advisory["evidence_tools"] == ["tide_chart"]


def test_fallback_is_schema_extraction_without_tools(graph, tools):
    assert graph.nodes[FALLBACK].mode == "schema_extraction"
    manager = _manager(graph, tools)
    result = manager.run(RunRequest(intent="write me a sonnet", current_state="End"))
    assert not result.rejected
    assert result.final_state == "End"
    output = result.steps[0].results[0].output
    assert output["department"] == "out_of_scope"


def test_the_desk_declares_what_a_call_must_carry(graph):
    schema = graph.input_schema
    assert set(schema["properties"]) == {"vessel", "crate_id", "channel"}
    assert schema["additionalProperties"] is False
    assert graph.entry_input_error({"vessel": "Aurora"}) is None
    # A bare call names nothing the desk can act on.
    assert graph.entry_input_error({}) is not None
    assert graph.entry_input_error({"tug": "Pilot One"}) is not None


def test_a_call_that_breaks_the_entry_contract_never_reaches_a_lane(graph, tools):
    manager = _manager(graph, tools, first=BERTH_SCOUT)
    result = manager.run(RunRequest(intent="dock the Aurora", state={"berth": 4}))

    assert result.rejected
    assert "input schema" in (result.rejection or "")
    assert result.final_state == "Start"
    assert result.steps == []
    assert result.audit.committed_transitions == []


def test_graph_manifest_helper_matches_builder(graph, tools):
    manifest = harbor_manifest()
    assert manifest == graph_manifest(graph, tools)
    assert manifest["input_schema"] == graph.input_schema
