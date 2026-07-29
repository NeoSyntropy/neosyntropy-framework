"""End-to-end refund workflow: states, edges, guards, axioms, and tools.

Run from the repository root::

    python examples/refund_workflow.py

The demo walks three control cycles (verify -> calculate -> issue), then
shows two rejections: a refund that breaks the MarginFloor axiom, and an
out-of-scope intent that routes to the dedicated fallback.
"""
from __future__ import annotations

from pydantic import BaseModel

from neosyntropy import (
    ControlManager,
    Edge,
    Graph,
    Group,
    RunRequest,
    ToolRegistry,
    axiom,
    node,
    tool,
)

# --- Tools (capabilities, never graph vertices) ------------------------------

registry = ToolRegistry()


class LookupOrderArgs(BaseModel):
    order_id: str


@tool(registry=registry)
def lookup_order(args: LookupOrderArgs) -> dict:
    """Look up an order and return its paid amount."""
    return {"order_id": args.order_id, "amount": 120.0, "customer": "c_42"}


# --- Nodes (executable capabilities) -----------------------------------------


@node(id="VerifyIdentity", group="refunds", tools=("lookup_order",))
def verify_identity(ctx):
    """Verify the requester owns the order."""
    order = ctx.tools.invoke("lookup_order", {"order_id": ctx.state.get("order_id", "?")})
    return ctx.result(
        output=f"verified owner of {order['order_id']}",
        state_updates={"verified": True, "order_amount": order["amount"]},
    )


@node(id="CalculateRefund", group="refunds", prerequisites=("VerifyIdentity",))
def calculate_refund(ctx):
    """Propose a refund amount from the order and the requested amount."""
    requested = ctx.state.get("requested_amount", ctx.state.get("order_amount", 0.0))
    return ctx.result(
        output=f"refund of {requested} calculated",
        state_updates={"refund_amount": requested},
    )


@node(id="IssueRefund", group="refunds", prerequisites=("CalculateRefund",))
def issue_refund(ctx):
    """Issue the previously calculated refund."""
    return ctx.result(
        output=f"issued {ctx.state['refund_amount']}",
        state_updates={"refund_issued": True},
        next_state="End",
    )


@node(id="OutOfScope", is_fallback=True)
def out_of_scope(ctx):
    """Safe stop for anything the workflow does not support."""
    return ctx.result(output="This request is out of scope for the refund workflow.")


# --- Axioms (invariants the AI can never break) -------------------------------


@axiom(name="MarginFloor", error_message="Refund exceeds the allowed maximum of 200.")
def margin_floor(ctx, proposal):
    return proposal.state.get("refund_amount", 0.0) <= 200.0


@axiom(name="VerifiedBeforeRefund", nodes=("IssueRefund",))
def verified_before_refund(ctx, proposal):
    return proposal.state.get("verified", False)


# --- Graph (the single source of permission) ----------------------------------

graph = Graph(
    nodes=[verify_identity, calculate_refund, issue_refund, out_of_scope],
    edges=[
        Edge(source="Start", target="VerifyIdentity", label="first"),
        Edge(source="VerifyIdentity", target="CalculateRefund", label="next"),
        Edge(
            source="CalculateRefund",
            target="IssueRefund",
            label="next",
            # Guard: the edge itself refuses zero/negative refunds (fail-closed).
            guard=lambda state: state.get("refund_amount", 0.0) > 0.0,
        ),
        Edge(source="IssueRefund", target="End", label="complete"),
    ],
    groups=[Group(name="refunds", description="Refund handling capabilities")],
    axioms=[margin_floor, verified_before_refund],
)

manager = ControlManager(graph, tools=registry)


def show(title: str, result) -> None:
    print(f"\n=== {title} ===")
    if result.plan is not None:
        print(f"plan:      {result.plan.topology.value} {result.plan.execution_plan}")
    else:
        print("plan:      (owned by backend)")
    print(f"final:     {result.final_state} (completed={result.completed})")
    if result.rejected:
        print(f"REJECTED:  {result.rejection}")
    print(f"commits:   {result.audit.committed_transitions}")
    for check in result.audit.axiom_checks:
        status = "ok " if check.passed else "FAIL"
        print(f"  [{status}] {check.stage}:{check.name} {check.message}")


def main() -> None:
    # Cycle 1..3: the happy path, one verified transition per cycle.
    state: dict = {"order_id": "ord_123", "requested_amount": 80.0}
    current, prior = "Start", []
    for cycle in range(3):
        result = manager.run(
            RunRequest(
                intent="please refund my order",
                current_state=current,
                state=state,
                prior_executions=prior,
            )
        )
        show(f"Cycle {cycle + 1}: from {current}", result)
        current, state = result.final_state, result.state
        prior = prior + [
            {"node_id": item.node_id, "status": item.status, "output": item.output}
            for step in result.steps
            for item in step.results
        ]

    # Rejection: a refund that violates the MarginFloor axiom. No commit.
    over_limit = manager.run(
        RunRequest(
            intent="refund my order",
            current_state="VerifyIdentity",
            state={"verified": True, "order_amount": 900.0, "requested_amount": 900.0},
            prior_executions=[{"node_id": "VerifyIdentity", "status": "succeeded"}],
        )
    )
    show("Broken axiom: refund of 900", over_limit)

    # Fallback: nothing legal from End, so the router proposes the safe stop.
    lost = manager.run(RunRequest(intent="write me a poem", current_state="End"))
    show("Out of scope intent", lost)


if __name__ == "__main__":
    main()
