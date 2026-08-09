"""End-to-end refund workflow: states, edges, guards, and tools.



Run from the repository root::



    python examples/refund_workflow.py



The demo walks three control cycles (verify -> calculate -> issue), then

shows a guard rejection (zero refund) and an out-of-scope input that routes

to the dedicated fallback.

"""

from __future__ import annotations



from pydantic import BaseModel



from neosyntropy import (
    OpenInput,

    ControlManager,

    FSM,

    Group,

    RunRequest,

    TextOutput,

    ToolRegistry,

    edge_deterministic,

    edge_fallback,

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





@node(id="VerifyIdentity", group="refunds", tools=("lookup_order",), output_schema=TextOutput)

def verify_identity(ctx):

    """Verify the requester owns the order."""

    order = ctx.tools.invoke("lookup_order", {"order_id": ctx.state.get("order_id", "?")})

    return ctx.result(

        output={"message": f"verified owner of {order['order_id']}"},

        state_updates={"verified": True, "order_amount": order["amount"]},

    )





@node(

    id="CalculateRefund",

    group="refunds",

    prerequisites=("VerifyIdentity",), input_schema=OpenInput, output_schema=TextOutput,

)

def calculate_refund(ctx):

    """Propose a refund amount from the order and the requested amount."""

    requested = ctx.state.get("requested_amount", ctx.state.get("order_amount", 0.0))

    return ctx.result(

        output={"message": f"refund of {requested} calculated"},

        state_updates={"refund_amount": requested},

    )





@node(

    id="IssueRefund",

    group="refunds",

    prerequisites=("CalculateRefund",), input_schema=OpenInput, output_schema=TextOutput,

)

def issue_refund(ctx):

    """Issue the previously calculated refund."""

    return ctx.result(

        output={"message": f"issued {ctx.state['refund_amount']}"},

        state_updates={"refund_issued": True},

        next_state="End",

    )





@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)

def out_of_scope(ctx):

    """Safe stop for anything the workflow does not support."""

    return ctx.result(

        output={"message": "This request is out of scope for the refund workflow."}

    )





# --- FSM (the single source of permission) ----------------------------------



graph = FSM(
        entry="ENTRY",

    nodes=[verify_identity, calculate_refund, issue_refund, out_of_scope],

    edges=[

        edge_deterministic("ENTRY", "VerifyIdentity"),

        edge_deterministic("VerifyIdentity", "CalculateRefund"),

        edge_deterministic(

            "CalculateRefund",

            "IssueRefund",

            # Guard: the edge itself refuses zero/negative refunds (fail-closed).

            guard=lambda state: state.get("refund_amount", 0.0) > 0.0,

        ),

        edge_deterministic("IssueRefund", "End"),

        edge_fallback("ENTRY", "OutOfScope"),

        edge_fallback("CalculateRefund", "OutOfScope"),

    ],

    groups=[Group(name="refunds", description="Refund handling capabilities")]
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

    for check in result.audit.gate_checks:

        status = "ok " if check.passed else "FAIL"

        print(f"  [{status}] {check.stage}:{check.name} {check.message}")





def main() -> None:

    # Cycle 1..3: the happy path, one verified transition per cycle.

    state: dict = {"order_id": "ord_123", "requested_amount": 80.0}

    current, prior = "ENTRY", []

    for cycle in range(3):

        result = manager.run(

            RunRequest(

                input={"text": "please refund my order"},

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



    # Rejection: edge guard blocks a non-positive refund. No commit.

    blocked = manager.run(

        RunRequest(

            input={"text": "refund my order"},

            current_state="CalculateRefund",

            state={"verified": True, "refund_amount": 0.0},

            prior_executions=[

                {"node_id": "VerifyIdentity", "status": "succeeded"},

                {"node_id": "CalculateRefund", "status": "succeeded"},

            ],

        )

    )

    show("Guard blocked: refund of 0", blocked)



    # Fallback: nothing legal from End, so the router proposes the safe stop.

    lost = manager.run(
        RunRequest(input={"text": "write me a poem"}, current_state="End")
    )

    show("Out of scope input", lost)





if __name__ == "__main__":

    main()


