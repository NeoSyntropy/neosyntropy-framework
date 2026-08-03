from __future__ import annotations

import json

from neosyntropy import (
    ControlManager,
    Edge,
    FSM,
    JsonlDecisionLogger,
    OpenInput,
    RunRequest,
    node,
)

from .conftest import build_graph


def make_request(**overrides) -> RunRequest:
    payload = {"intent": "refund my order", "current_state": "Start"}
    payload.update(overrides)
    return RunRequest.model_validate(payload)


def test_legal_cycle_commits_exactly_one_transition(refund_graph):
    manager = ControlManager(refund_graph)
    result = manager.run(make_request())
    assert not result.rejected
    assert result.completed
    assert result.final_state == "VerifyIdentity"
    assert result.state["verified"] is True
    assert result.audit.committed_transitions == ["Start->VerifyIdentity"]
    assert all(check.passed for check in result.audit.gate_checks)


def test_full_walk_across_cycles(refund_graph):
    manager = ControlManager(refund_graph)
    current, state, prior = "Start", {"requested_amount": 80.0}, []
    for _ in range(3):
        result = manager.run(
            make_request(current_state=current, state=state, prior_executions=prior)
        )
        assert not result.rejected, result.rejection
        current, state = result.final_state, result.state
        prior = prior + [
            {"node_id": item.node_id, "status": item.status}
            for step in result.steps
            for item in step.results
        ]
    # IssueRefund proposed End explicitly; the edge permits it.
    assert current == "End"
    assert state["refund_issued"] is True



def test_illegal_plan_is_rejected_before_execution():
    class JumpRouter:
        async def route(self, context, candidates):
            from neosyntropy import RoutingPlan, Topology

            # Propose CalculateRefund before VerifyIdentity — permitted by the
            # semantic edge but blocked by prerequisites.
            calc = next(
                i for i, c in enumerate(candidates) if c.node_id == "CalculateRefund"
            )
            return RoutingPlan(
                topology=Topology.SEQUENTIAL, execution_plan=[[calc]]
            )

    # Semantic scope from Start so the (misbehaving) router is consulted;
    # deterministic short-circuit would otherwise skip it.
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
    manager = ControlManager(graph, router=JumpRouter())
    result = manager.run(make_request())
    assert result.rejected
    assert "invalid routing plan" in (result.rejection or "")
    assert result.steps == []
    assert result.final_state == "Start"


def test_fallback_cycle_is_a_safe_stop(refund_graph):
    manager = ControlManager(refund_graph)
    result = manager.run(make_request(intent="write me a poem", current_state="End"))
    assert not result.rejected
    assert result.plan.topology.value == "fallback"
    # Fallback keeps the workflow where it was.
    assert result.final_state == "End"
    assert result.audit.committed_transitions == []


def _entry_guarded_graph() -> FSM:
    from pydantic import BaseModel, ConfigDict

    class RefundInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        requested_amount: float
        currency: str = "USD"

    return build_graph(input_schema=RefundInput)


def test_entry_input_is_gated_before_anything_runs():
    manager = ControlManager(_entry_guarded_graph())
    result = manager.run(make_request(state={"currency": "USD"}))

    assert result.rejected
    assert "input schema" in (result.rejection or "")
    # Rejected at the door: no plan, no candidates, no steps, no commit.
    assert result.plan is None
    assert result.candidates == []
    assert result.steps == []
    assert result.final_state == "Start"
    assert result.audit.committed_transitions == []
    assert [check.name for check in result.audit.gate_checks] == ["InputSchema"]


def test_valid_entry_input_runs_normally():
    manager = ControlManager(_entry_guarded_graph())
    result = manager.run(make_request(state={"requested_amount": 80.0}))

    assert not result.rejected, result.rejection
    assert result.final_state == "VerifyIdentity"
    assert [check.name for check in result.audit.gate_checks][0] == "InputSchema"


def test_entry_contract_only_gates_the_entry_point():
    """Resuming mid-workflow carries state the workflow itself produced."""
    manager = ControlManager(_entry_guarded_graph())
    result = manager.run(
        make_request(
            current_state="VerifyIdentity",
            state={"verified": True, "requested_amount": 80.0},
            prior_executions=[{"node_id": "VerifyIdentity", "status": "succeeded"}],
        )
    )

    assert not result.rejected, result.rejection
    assert "InputSchema" not in [check.name for check in result.audit.gate_checks]


def test_node_input_schema_rejects_before_execution():
    from pydantic import BaseModel, ConfigDict

    from neosyntropy import EmptyOutput, OpenInput, TextOutput

    class NeedsVerified(BaseModel):
        model_config = ConfigDict(extra="forbid")

        verified: bool

    @node(id="NeedsState", input_schema=NeedsVerified, output_schema=EmptyOutput)
    def needs_state(ctx):
        return ctx.result(output={}, state_updates={"ran": True})

    @node(id="Safe", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def safe(ctx):
        return ctx.result(output={"message": "stop"})

    graph = FSM(
        nodes=[needs_state, safe],
        edges=[Edge(source="Start", target="NeedsState", kind="deterministic")],
        validate_reachability=False,
    )
    result = ControlManager(graph).run(make_request(intent="go", state={}))

    assert result.rejected
    assert "input_schema" in (result.rejection or "")
    assert result.final_state == "Start"
    assert "ran" not in result.state
    failed = [c for c in result.audit.gate_checks if not c.passed]
    assert failed and failed[0].name == "NodeInputSchema"


def test_node_input_schema_allows_matching_state():
    from pydantic import BaseModel, ConfigDict

    from neosyntropy import EmptyOutput, OpenInput, TextOutput

    class NeedsVerified(BaseModel):
        model_config = ConfigDict(extra="forbid")

        verified: bool

    @node(id="NeedsState", input_schema=NeedsVerified, output_schema=EmptyOutput)
    def needs_state(ctx):
        return ctx.result(output={}, state_updates={"ran": True})

    @node(id="Safe", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def safe(ctx):
        return ctx.result(output={"message": "stop"})

    graph = FSM(
        nodes=[needs_state, safe],
        edges=[Edge(source="Start", target="NeedsState", kind="deterministic")],
        validate_reachability=False,
    )
    result = ControlManager(graph).run(
        make_request(intent="go", state={"verified": True})
    )

    assert not result.rejected, result.rejection
    assert result.state["ran"] is True
    assert any(
        check.name == "NodeInputSchema" and check.passed
        for check in result.audit.gate_checks
    )


def test_handler_proposed_illegal_transition_is_rejected():
    from neosyntropy import EmptyOutput, OpenInput, TextOutput

    @node(id="Rogue", input_schema=OpenInput, output_schema=EmptyOutput)
    def rogue(ctx):
        return ctx.result(
            output={},
            next_state="SomewhereIllegal",
            state_updates={"x": 1},
        )

    @node(id="Safe", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def safe(ctx):
        return ctx.result(output={"message": "stop"})

    graph = FSM(
        nodes=[rogue, safe],
        edges=[Edge(source="Start", target="Rogue", kind="deterministic")],
        validate_reachability=False,
    )
    manager = ControlManager(graph)
    result = manager.run(make_request(intent="rogue"))
    assert result.rejected
    assert "no legal guard-allowed transition" in (result.rejection or "")
    assert result.final_state == "Start"
    assert result.state == {}


def _guarded_graph(*, with_semantic: bool = False) -> FSM:
    graph = build_graph()
    # Rebuild with a guard that requires a positive refund amount.
    edges = [
        Edge(source="Start", target="VerifyIdentity", kind="deterministic"),
        Edge(source="VerifyIdentity", target="CalculateRefund", kind="deterministic"),
        Edge(
            source="CalculateRefund",
            target="IssueRefund",
            kind="deterministic",
            guard=lambda state: state.get("refund_amount", 0.0) > 0.0,
        ),
        Edge(source="IssueRefund", target="End", kind="deterministic"),
        Edge(source="CalculateRefund", target="OutOfScope", kind="fallback"),
    ]
    if with_semantic:
        edges.append(
            Edge(
                source="CalculateRefund",
                target="IssueRefund",
                kind="semantic",
            )
        )
    return FSM(nodes=list(graph.nodes.values()), edges=edges)


def _zero_refund_request():
    # refund_amount 0 -> the guard denies CalculateRefund->IssueRefund.
    return make_request(
        current_state="CalculateRefund",
        state={"verified": True, "refund_amount": 0.0},
        prior_executions=[
            {"node_id": "VerifyIdentity", "status": "succeeded"},
            {"node_id": "CalculateRefund", "status": "succeeded"},
        ],
    )


def test_edge_guard_denies_forced_plan_before_execution():
    # A misbehaving router proposes the guarded hop anyway; the control
    # layer's guard gate rejects it before the node ever runs.
    class ForcedRouter:
        async def route(self, context, candidates):
            from neosyntropy import RoutingPlan, Topology

            issue = next(
                i for i, c in enumerate(candidates) if c.node_id == "IssueRefund"
            )
            return RoutingPlan(topology=Topology.SEQUENTIAL, execution_plan=[[issue]])

    manager = ControlManager(
        _guarded_graph(with_semantic=True), router=ForcedRouter()
    )
    result = manager.run(_zero_refund_request())
    assert result.rejected
    assert "edge guard denied" in (result.rejection or "")
    assert result.final_state == "CalculateRefund"
    assert "refund_issued" not in result.state


def test_deterministic_router_honors_guards_and_falls_back():
    manager = ControlManager(_guarded_graph())
    result = manager.run(_zero_refund_request())
    assert not result.rejected
    assert result.plan.topology.value == "fallback"
    assert result.final_state == "CalculateRefund"
    assert result.audit.committed_transitions == []


def test_tool_allow_list_violation_is_a_rejection():
    from neosyntropy import EmptyOutput, OpenInput, TextOutput

    @node(id="Sneaky", tools=(), input_schema=OpenInput, output_schema=EmptyOutput)
    def sneaky(ctx):
        return ctx.tools.invoke("not_registered", {})

    @node(id="Safe", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def safe(ctx):
        return ctx.result(output={"message": "stop"})

    graph = FSM(
        nodes=[sneaky, safe],
        edges=[Edge(source="Start", target="Sneaky", kind="deterministic")],
        validate_reachability=False,
    )
    manager = ControlManager(graph)
    result = manager.run(make_request(intent="sneaky"))
    assert result.rejected
    assert "not allowed on node" in (result.rejection or "")
    assert result.state == {}


def test_decision_logger_writes_jsonl(tmp_path, refund_graph):
    log_path = tmp_path / "decisions" / "router.jsonl"
    manager = ControlManager(
        refund_graph, decision_logger=JsonlDecisionLogger(log_path)
    )
    manager.run(make_request())
    manager.run(make_request())
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["node"] == "Start"
    assert record["user_query"] == "refund my order"
    assert record["output"]["topology"] == "sequential"


def test_audit_record_tells_the_whole_story(refund_graph):
    manager = ControlManager(refund_graph)
    result = manager.run(make_request())
    audit = result.audit
    assert audit.request_id == result.request_id
    assert audit.initial_state == "Start"
    assert audit.final_state == "VerifyIdentity"
    assert audit.plan == result.plan
    assert any(check.name == "PlanValidator" for check in audit.gate_checks)
    assert any(check.name == "TransitionLegality" for check in audit.gate_checks)
