from __future__ import annotations



import pytest



from neosyntropy import (
    OpenInput,

    EmptyOutput,

    Graph,

    TextOutput,

    edge_deterministic,

    edge_fallback,

    node,

)





@node(id="VerifyIdentity", group="refunds", input_schema=OpenInput, output_schema=EmptyOutput)

def verify_identity(ctx):

    """Verify the requester owns the order."""

    return ctx.result(output={}, state_updates={"verified": True})





@node(

    id="CalculateRefund",

    group="refunds",

    prerequisites=("VerifyIdentity",),

    input_schema=OpenInput, output_schema=EmptyOutput,

)

def calculate_refund(ctx):

    """Calculate the refund amount."""

    requested = ctx.state.get("requested_amount", 0.0)

    return ctx.result(output={}, state_updates={"refund_amount": requested})





@node(

    id="IssueRefund",

    group="refunds",

    prerequisites=("CalculateRefund",),

    input_schema=OpenInput, output_schema=EmptyOutput,

)

def issue_refund(ctx):

    """Issue the calculated refund."""

    return ctx.result(

        output={},

        state_updates={"refund_issued": True},

        next_state="End",

    )





@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)

def out_of_scope(ctx):

    """Safe stop."""

    return ctx.result(output={"message": "out of scope"})





def build_graph(**kwargs) -> Graph:

    return Graph(

        nodes=[verify_identity, calculate_refund, issue_refund, out_of_scope],

        edges=[

            edge_deterministic("Start", "VerifyIdentity"),

            edge_deterministic("VerifyIdentity", "CalculateRefund"),

            edge_deterministic("CalculateRefund", "IssueRefund"),

            edge_deterministic("IssueRefund", "End"),

            edge_fallback("Start", "OutOfScope"),

        ],

        **kwargs,

    )





@pytest.fixture

def refund_graph() -> Graph:

    return build_graph()


