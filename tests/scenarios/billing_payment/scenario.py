"""Scenario: billing payment.

A Group owns card validation and a deterministic router. A valid Visa
(prefix 4) is captured into the payments table; anything else is rejected
and recorded as declined. Deliveries query that table after the run.

```mermaid
flowchart LR
    ValidateCard --> BillingLogic
    BillingLogic -->|card_valid| ProcessPayment --> End
    BillingLogic -->|not valid| RejectCard --> End
    ValidateCard -.-> OutOfScope
```
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    DeterministicRouter,
    Group,
    NodeContext,
    OpenInput,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
)
from tests.scenarios.stores import SqliteDatabase, run_fsm

SCENARIO_ID = "billing_payment"
SCENARIO_TITLE = "Billing payment"


class CardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payment_id: str
    card_number: str
    amount: float


class CardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    card_valid: bool


class PaymentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payment_id: str
    status: str


@dataclass
class BillingPaymentScenario:
    fsm: FSM
    db: SqliteDatabase

    def run(self, payload: CardRequest):
        return run_fsm(self.fsm, payload)


def _schema(db: SqliteDatabase) -> None:
    db.execute(
        "CREATE TABLE payments ("
        "payment_id TEXT PRIMARY KEY, "
        "card_last4 TEXT NOT NULL, "
        "amount REAL NOT NULL, "
        "status TEXT NOT NULL)"
    )


def build_scenario() -> BillingPaymentScenario:
    db = SqliteDatabase(name="payments_db")
    _schema(db)
    billing = Group(name="billing")

    @billing.node(id="ValidateCard", input_schema=CardRequest, output_schema=CardOutput)
    def validate_card(ctx: NodeContext) -> object:
        card_number = str(ctx.input["card_number"])
        valid = card_number.startswith("4")
        return ctx.result(
            output={"card_valid": valid},
            state_updates={
                "card_valid": valid,
                "payment_id": ctx.input["payment_id"],
                "amount": ctx.input["amount"],
                "card_last4": card_number[-4:],
            },
        )

    @billing.node(id="ProcessPayment", input_schema=OpenInput, output_schema=PaymentOutput)
    def process_payment(ctx: NodeContext) -> object:
        payment_id = str(ctx.state["payment_id"])
        db.insert_row(
            "payments",
            {
                "payment_id": payment_id,
                "card_last4": str(ctx.state["card_last4"]),
                "amount": float(ctx.state["amount"]),
                "status": "captured",
            },
        )
        return ctx.result(
            output={"payment_id": payment_id, "status": "captured"},
            state_updates={"paid": True},
            next_state="End",
        )

    @billing.node(id="RejectCard", input_schema=OpenInput, output_schema=PaymentOutput)
    def reject_card(ctx: NodeContext) -> object:
        payment_id = str(ctx.state["payment_id"])
        db.insert_row(
            "payments",
            {
                "payment_id": payment_id,
                "card_last4": str(ctx.state["card_last4"]),
                "amount": float(ctx.state["amount"]),
                "status": "declined",
            },
        )
        return ctx.result(
            output={"payment_id": payment_id, "status": "declined"},
            state_updates={"paid": False},
            next_state="End",
        )

    logic = DeterministicRouter(
        id="BillingLogic",
        rules=[
            (lambda ctx: ctx.state.get("card_valid") is True, "ProcessPayment"),
            (lambda ctx: ctx.state.get("card_valid") is False, "RejectCard"),
        ],
    )
    billing.routers = [logic]
    billing.entry = "ValidateCard"
    billing.add_edge("ValidateCard", "BillingLogic")

    @node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def out_of_scope(ctx: NodeContext) -> object:
        return ctx.result(output={"message": "Cannot process payment."})

    fsm = FSM(
        entry="ValidateCard",
        nodes=[out_of_scope],
        groups=[billing],
        edges=[
            edge_deterministic("ProcessPayment", "End"),
            edge_deterministic("RejectCard", "End"),
            edge_fallback("ValidateCard", "OutOfScope"),
        ],
    )
    return BillingPaymentScenario(fsm=fsm, db=db)
