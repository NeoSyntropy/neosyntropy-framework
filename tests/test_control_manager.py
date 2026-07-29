from __future__ import annotations

import json

from neosyntropy import (
    ControlManager,
    Edge,
    Graph,
    JsonlDecisionLogger,
    RunRequest,
    axiom,
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
    assert all(check.passed for check in result.audit.axiom_checks)


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


def test_broken_axiom_rejects_step_with_no_state_change():
    @axiom(name="MaxRefund", error_message="refund exceeds cap")
    def max_refund(ctx, proposal):
        return proposal.state.get("refund_amount", 0.0) <= 200.0

    graph = build_graph(axioms=[max_refund])
    manager = ControlManager(graph)
    result = manager.run(
        make_request(
            current_state="VerifyIdentity",
            state={"verified": True, "requested_amount": 900.0},
            prior_executions=[{"node_id": "VerifyIdentity", "status": "succeeded"}],
        )
    )
    assert result.rejected
    assert "refund exceeds cap" in (result.rejection or "")
    # Nothing committed: still in VerifyIdentity, no refund_amount in state.
    assert result.final_state == "VerifyIdentity"
    assert "refund_amount" not in result.state
    assert result.audit.committed_transitions == []
    failed = [c for c in result.audit.axiom_checks if not c.passed]
    assert failed and failed[0].name == "MaxRefund"


def test_illegal_plan_is_rejected_before_execution(refund_graph):
    class JumpRouter:
        async def route(self, context, candidates):
            from neosyntropy import RoutingPlan, Topology

            issue = next(
                i for i, c in enumerate(candidates) if c.node_id == "IssueRefund"
            )
            return RoutingPlan(
                topology=Topology.SEQUENTIAL, execution_plan=[[issue]]
            )

    manager = ControlManager(refund_graph, router=JumpRouter())
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


def test_handler_proposed_illegal_transition_is_rejected():
    @node(id="Rogue")
    def rogue(ctx):
        return ctx.result(next_state="SomewhereIllegal", state_updates={"x": 1})

    @node(id="Safe", is_fallback=True)
    def safe(ctx):
        return ctx.result(output="stop")

    graph = Graph(
        nodes=[rogue, safe],
        edges=[Edge(source="Start", target="Rogue", label="first")],
        validate_reachability=False,
    )
    manager = ControlManager(graph)
    result = manager.run(make_request(intent="rogue"))
    assert result.rejected
    assert "no legal guard-allowed transition" in (result.rejection or "")
    assert result.final_state == "Start"
    assert result.state == {}


def _guarded_graph() -> Graph:
    graph = build_graph()
    # Rebuild with a guard that requires a positive refund amount.
    return Graph(
        nodes=list(graph.nodes.values()),
        edges=[
            Edge(source="Start", target="VerifyIdentity", label="first"),
            Edge(source="VerifyIdentity", target="CalculateRefund", label="next"),
            Edge(
                source="CalculateRefund",
                target="IssueRefund",
                label="next",
                guard=lambda state: state.get("refund_amount", 0.0) > 0.0,
            ),
            Edge(source="IssueRefund", target="End", label="complete"),
        ],
    )


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

    manager = ControlManager(_guarded_graph(), router=ForcedRouter())
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
    @node(id="Sneaky", tools=())
    def sneaky(ctx):
        return ctx.tools.invoke("not_registered", {})

    @node(id="Safe", is_fallback=True)
    def safe(ctx):
        return ctx.result(output="stop")

    graph = Graph(
        nodes=[sneaky, safe],
        edges=[Edge(source="Start", target="Sneaky", label="first")],
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
    assert any(check.name == "PlanValidator" for check in audit.axiom_checks)
    assert any(check.name == "TransitionLegality" for check in audit.axiom_checks)
