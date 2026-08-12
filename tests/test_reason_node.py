"""Local runnable demo: ReasoningNode investigates a refund with tools.

Loads credentials from ``tests/.env`` (see ``tests/.env.example``).
Do not hardcode ``api_key`` / project ids in source — CI secret scan will fail.

Run::

    python tests/test_reason_node.py
"""

from __future__ import annotations

import os
from pathlib import Path

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

VERTEX_MODEL = "gemini-2.5-flash"
TESTS_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_tests_env() -> None:
    if not TESTS_ENV_PATH.is_file():
        raise SystemExit(
            f"Missing {TESTS_ENV_PATH}. Copy tests/.env.example to tests/.env and fill values."
        )
    for raw in TESTS_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required value {name} in {TESTS_ENV_PATH}.")
    return value


def _client_from_env() -> Client:
    return Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        project_id=_require_env("NEOSYNTROPY_PROJECT_ID"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com").strip()
        or "https://api.neosyntropy.com",
        telemetry_timeout=20.0,
    )


registry = ToolRegistry()


class RefundInvestigationInput(BaseModel):
    """Typed run input required at the FSM entry before the investigation starts."""

    model_config = ConfigDict(extra="forbid")
    intent: str


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
    _load_tests_env()
    client = _client_from_env()
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
                print(
                    f"  [{verdict}] Tool Call: {record.tool} {record.arguments} "
                    f"{record.error or ''}"
                )
