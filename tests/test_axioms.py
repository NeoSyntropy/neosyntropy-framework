from __future__ import annotations

from neosyntropy import (
    Axiom,
    AxiomEngine,
    OutputAxiom,
    Proposal,
    RunContext,
    axiom,
)
from neosyntropy.core.models import NodeResult


def make_context(**overrides) -> RunContext:
    payload = {
        "request_id": "req-1",
        "intent": "refund my order",
        "current_state": "Start",
        "state": {},
    }
    payload.update(overrides)
    return RunContext.model_validate(payload)


def result_proposal(node_id: str, state: dict, output=None) -> Proposal:
    return Proposal(
        kind="result",
        state=state,
        current_state="Start",
        node_id=node_id,
        result=NodeResult(node_id=node_id, output=output),
    )


def test_predicate_axiom_pass_and_fail():
    @axiom(name="MaxRefund", error_message="cap exceeded")
    def max_refund(ctx, proposal):
        return proposal.state.get("refund_amount", 0) <= 200

    context = make_context()
    ok = max_refund.check(context, result_proposal("IssueRefund", {"refund_amount": 100}))
    assert ok.passed
    bad = max_refund.check(context, result_proposal("IssueRefund", {"refund_amount": 900}))
    assert not bad.passed
    assert "cap exceeded" in bad.message


def test_predicate_exception_is_a_violation_fail_closed():
    @axiom(name="Broken")
    def broken(ctx, proposal):
        raise RuntimeError("boom")

    check = broken.check(make_context(), result_proposal("X", {}))
    assert not check.passed
    assert "boom" in check.message


def test_node_scope_only_applies_to_listed_nodes():
    guard = Axiom(
        "OnlyRefund",
        predicate=lambda ctx, proposal: False,
        nodes=("IssueRefund",),
    )
    assert guard.applies("result", current_state="Any", node_id="IssueRefund")
    assert not guard.applies("result", current_state="Any", node_id="Other")
    assert not guard.applies("result", current_state="Any", node_id=None)


def test_state_scope_only_applies_in_listed_states():
    guard = Axiom(
        "CheckoutOnly",
        predicate=lambda ctx, proposal: True,
        states=("Checkout",),
    )
    assert guard.applies("plan", current_state="Checkout", node_id=None)
    assert not guard.applies("plan", current_state="Browse", node_id=None)


def test_stage_scoping():
    plan_only = Axiom("PlanOnly", predicate=lambda c, p: True, stage="plan")
    assert plan_only.applies("plan", current_state="S", node_id=None)
    assert not plan_only.applies("result", current_state="S", node_id=None)


def test_output_axiom_validates_json_schema():
    schema = {
        "type": "object",
        "properties": {"total": {"type": "number"}},
        "required": ["total"],
    }
    output_axiom = OutputAxiom("TotalsShape", schema, nodes=("Report",))
    context = make_context()
    good = output_axiom.check(context, result_proposal("Report", {}, output={"total": 4.2}))
    assert good.passed
    bad = output_axiom.check(context, result_proposal("Report", {}, output={"x": 1}))
    assert not bad.passed
    assert "Schema validation failed" in bad.message


def test_engine_evaluates_only_applicable_axioms():
    engine = AxiomEngine(
        [
            Axiom("Global", predicate=lambda c, p: True),
            Axiom("NodeScoped", predicate=lambda c, p: False, nodes=("A",)),
        ]
    )
    context = make_context()
    plain = engine.evaluate("result", context, result_proposal("B", {}))
    assert [check.name for check in plain] == ["Global"]

    scoped = engine.evaluate("result", context, result_proposal("A", {}))
    assert {check.name for check in scoped} == {"Global", "NodeScoped"}

    node_only = engine.evaluate(
        "result", context, result_proposal("A", {}), node_scoped_only=True
    )
    assert [check.name for check in node_only] == ["NodeScoped"]
