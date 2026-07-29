from __future__ import annotations

import pytest

from neosyntropy import Edge, Graph, node


@node(id="VerifyIdentity", group="refunds")
def verify_identity(ctx):
    """Verify the requester owns the order."""
    return ctx.result(state_updates={"verified": True})


@node(id="CalculateRefund", group="refunds", prerequisites=("VerifyIdentity",))
def calculate_refund(ctx):
    """Calculate the refund amount."""
    requested = ctx.state.get("requested_amount", 0.0)
    return ctx.result(state_updates={"refund_amount": requested})


@node(id="IssueRefund", group="refunds", prerequisites=("CalculateRefund",))
def issue_refund(ctx):
    """Issue the calculated refund."""
    return ctx.result(state_updates={"refund_issued": True}, next_state="End")


@node(id="OutOfScope", is_fallback=True)
def out_of_scope(ctx):
    """Safe stop."""
    return ctx.result(output="out of scope")


def build_graph(**kwargs) -> Graph:
    return Graph(
        nodes=[verify_identity, calculate_refund, issue_refund, out_of_scope],
        edges=[
            Edge(source="Start", target="VerifyIdentity", label="first"),
            Edge(source="VerifyIdentity", target="CalculateRefund", label="next"),
            Edge(source="CalculateRefund", target="IssueRefund", label="next"),
            Edge(source="IssueRefund", target="End", label="complete"),
        ],
        **kwargs,
    )


@pytest.fixture
def refund_graph() -> Graph:
    return build_graph()
