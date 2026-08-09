from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
    tool,
)

# 1. Client setup with credentials
client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="http://127.0.0.1:8001",
    telemetry_timeout=20.0,
)

registry = ToolRegistry()


class RefundInvestigationInput(BaseModel):
    """Typed run input required at the FSM entry before the investigation starts."""

    model_config = ConfigDict(extra="forbid")
    intent: str


# --- Tool 1: Lookup Customer Account ---
class LookupCustomerArgs(BaseModel):
    customer_id: str


@tool(registry=registry)
def lookup_customer_account(args: LookupCustomerArgs) -> dict:
    """Look up customer account details including loyalty tier, lifetime spend, and account status."""
    return {
        "customer_id": args.customer_id,
        "vip_tier": "Gold",
        "account_status": "active",
        "extended_return_window_days": 60,
    }


# --- Tool 2: Fetch Transaction History ---
class FetchTransactionArgs(BaseModel):
    order_id: str


@tool(registry=registry)
def fetch_transaction_history(args: FetchTransactionArgs) -> dict:
    """Fetch details of an order transaction including purchase date, item ID, and purchase price."""
    return {
        "order_id": args.order_id,
        "product_id": "prod_laptop_pro",
        "purchase_days_ago": 35,
        "amount": 1299.99,
        "item_condition_reported": "defective",
    }


# --- Tool 3: Check Product Return Policy ---
class ProductPolicyArgs(BaseModel):
    product_id: str


@tool(registry=registry)
def check_product_return_policy(args: ProductPolicyArgs) -> dict:
    """Check return policy rules, warranty coverage, and restocking fees for a product."""
    return {
        "product_id": args.product_id,
        "category": "electronics",
        "standard_return_days": 30,
        "allows_defective_return": True,
        "restocking_fee_percent": 0.0,
    }


# Vertex Gemini model id (anything other than neosyntropy/base routes to Vertex).
VERTEX_MODEL = "gemini-2.5-flash"


class RefundClaimSummary(BaseModel):
    """Structured JSON summary of the refund investigation."""

    model_config = ConfigDict(extra="forbid")
    customer_id: str
    order_id: str
    claim_valid: bool
    decision: str
    rationale: str
    vip_tier: str | None = None
    purchase_days_ago: int | None = None
    refund_amount: float | None = None


# --- ReasoningNode — The central node under test ---
reason_node = ReasoningNode(
    id="ReasoningNode",
    input_schema=RefundInvestigationInput,
    provider=VERTEX_MODEL,
    prompt=(
        "Investigate the customer's request by calling tools to look up customer account details, "
        "fetch transaction history, and check product return policy. Reason step-by-step using "
        "the tool outputs to determine if the claim is valid."
    ),
    tools=(
        "lookup_customer_account",
        "fetch_transaction_history",
        "check_product_return_policy",
    ),
)

# SchemaNode — summarize investigation as constrained JSON
summarize = SchemaNode(
    id="Summarize",
    input_schema=OpenInput,
    output_schema=RefundClaimSummary,
    provider=VERTEX_MODEL,
    prompt=(
        "Summarize the investigation into a refund claim decision as JSON. "
        "Use the prior reasoning and tool findings. Set claim_valid and decision "
        "(approve / deny / escalate) with a short rationale."
    ),
)

out_of_scope = SchemaNode(
    id="OutOfScope",
    is_fallback=True,
    input_schema=OpenInput,
    output_schema=TextOutput,
    provider=VERTEX_MODEL,
    prompt="Politely refuse out-of-scope requests.",
)

# --- FSM Workflow: Reason → Summarize (JSON) ---
fsm = FSM(
    entry=reason_node,
    nodes=[reason_node, summarize, out_of_scope],
    edges=[
        edge_deterministic("ReasoningNode", "Summarize"),
        edge_deterministic("Summarize", "End"),
        edge_fallback("ReasoningNode", "OutOfScope"),
    ],
)

if __name__ == "__main__":
    result = fsm.run(
        RefundInvestigationInput(
            intent="Investigate refund request for order ord_98765 for customer cust_12345 (laptop defect)",
        ),
        state={
            "customer_id": "cust_12345",
            "order_id": "ord_98765",
            "issue": "laptop defect",
        },
        client=client,
        tools=registry,
    )
    print("Execution Finished!")
    print(f"Final State: {result.final_state} (Rejected: {result.rejected})")
    
    if result.rejection:
        print(f"REJECTED: {result.rejection}")
        
    print(f"Audit Log: {result.audit}")
    
    for step in result.steps:
        for item in step.results:
            print(f"\n=== Output ===\n{item.output}")
            for record in item.tool_calls:
                verdict = "ok" if record.ok else ("denied" if record.denied else "failed")
                print(f"  [{verdict}] Tool Call: {record.tool} {record.arguments} {record.error or ''}")