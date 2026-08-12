from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    TextOutput,
    ToolRegistry,
    CombineNode,
    node,
    edge_deterministic,
    edge_fallback,
    tool,
)

client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="https://api.neosyntropy.com",
    telemetry_timeout=20.0,
)
registry = ToolRegistry()

VERTEX_MODEL = "gemini-2.5-flash"


class ClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str


class ClaimSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    claim_valid: bool
    decision: str  # "approve" | "deny" | "escalate"
    rationale: str


class FetchOrderArgs(BaseModel):
    order_id: str


@tool(registry=registry)
def fetch_order(args: FetchOrderArgs) -> dict:
    """Fetch purchase date and item condition for the given order."""
    return {"purchase_days_ago": 20, "item_condition": "defective", "amount": 499.0}


investigate_and_summarise = CombineNode(
    id="InvestigateClaim",
    input_schema=ClaimInput,
    provider=VERTEX_MODEL,
    tools=("fetch_order",),
    output_schema=ClaimSummary,
    prompt=(
        "Use the fetch_order tool to look up the order. "
        "Reason about whether the refund claim is valid."
    ),
)


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


fsm = FSM(
    entry=investigate_and_summarise,
    nodes=[investigate_and_summarise, out_of_scope],
    edges=[
        edge_deterministic("InvestigateClaim.Schema", "End"),
        edge_fallback("InvestigateClaim", "OutOfScope"),
    ],
)


def test_combine_node():
    result = fsm.run(
        ClaimInput(intent="Refund order ord_77 — item was defective"),
        state={"order_id": "ord_77", "customer_id": "cust_55"},
        client=client,
        tools=registry,
    )
    print(f"Final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"

    found_reasoning = False
    found_schema = False
    for step in result.steps:
        for item in step.results:
            print(f"Node: {item.node_id}, Output: {item.output}")
            if item.node_id == "InvestigateClaim":
                found_reasoning = True
            elif item.node_id == "InvestigateClaim.Schema":
                found_schema = True
                assert "order_id" in item.output
                assert "claim_valid" in item.output
    assert found_reasoning
    assert found_schema


if __name__ == "__main__":
    test_combine_node()
