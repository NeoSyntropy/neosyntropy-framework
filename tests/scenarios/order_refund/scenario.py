"""Scenario: order refund.

A customer asks to refund an existing order. Trusted Python nodes validate
the order against the orders table, persist a refund row, then mark the
order refunded. Deliveries query both tables after the run.

```mermaid
flowchart LR
    ValidateOrder --> WriteRefund --> ConfirmRefund --> End
    ValidateOrder -.-> ErrorFallback
    WriteRefund -.-> ErrorFallback
    ConfirmRefund -.-> ErrorFallback
```
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    NodeContext,
    OpenInput,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
)
from tests.scenarios.stores import SqliteDatabase, run_fsm

SCENARIO_ID = "order_refund"
SCENARIO_TITLE = "Order refund"


class RefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    amount: float


class ValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str


class RefundOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refund_id: str
    order_id: str
    amount: float


class ConfirmOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    status: str


@dataclass
class OrderRefundScenario:
    fsm: FSM
    db: SqliteDatabase

    def run(self, payload: RefundRequest):
        return run_fsm(self.fsm, payload)


def _schema(db: SqliteDatabase) -> None:
    db.execute(
        "CREATE TABLE orders ("
        "order_id TEXT PRIMARY KEY, "
        "customer TEXT NOT NULL, "
        "amount REAL NOT NULL, "
        "status TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE refunds ("
        "refund_id TEXT PRIMARY KEY, "
        "order_id TEXT NOT NULL, "
        "amount REAL NOT NULL, "
        "status TEXT NOT NULL)"
    )
    db.insert_row(
        "orders",
        {
            "order_id": "ORD-001",
            "customer": "Alice",
            "amount": 149.99,
            "status": "paid",
        },
    )


def build_scenario() -> OrderRefundScenario:
    db = SqliteDatabase(name="orders_db")
    _schema(db)

    @node(id="ValidateOrder", input_schema=RefundRequest, output_schema=ValidationOutput)
    def validate_order(ctx: NodeContext) -> object:
        order_id = str(ctx.input["order_id"])
        amount = float(ctx.input["amount"])
        rows = db.read_sql(
            "SELECT order_id, amount, status FROM orders WHERE order_id = ?",
            params=(order_id,),
        )
        if not rows:
            return ctx.result(
                output={"valid": False, "reason": f"unknown order {order_id}"},
                state_updates={"valid": False, "reason": f"unknown order {order_id}"},
            )
        order = rows[0]
        if order["status"] != "paid":
            reason = f"order {order_id} is {order['status']}"
            return ctx.result(
                output={"valid": False, "reason": reason},
                state_updates={"valid": False, "reason": reason},
            )
        if amount <= 0 or amount > float(order["amount"]):
            reason = f"refund amount {amount} is not allowed"
            return ctx.result(
                output={"valid": False, "reason": reason},
                state_updates={"valid": False, "reason": reason},
            )
        return ctx.result(
            output={"valid": True, "reason": "order is refundable"},
            state_updates={
                "valid": True,
                "order_id": order_id,
                "amount": amount,
            },
        )

    @node(id="WriteRefund", input_schema=OpenInput, output_schema=RefundOutput)
    def write_refund(ctx: NodeContext) -> object:
        if not ctx.state.get("valid"):
            reason = str(ctx.state.get("reason", "validation failed"))
            return ctx.result(
                output={"refund_id": "", "order_id": "", "amount": 0.0},
                state_updates={"refund_written": False, "reason": reason},
                next_state="ConfirmRefund",
            )
        order_id = str(ctx.state["order_id"])
        amount = float(ctx.state["amount"])
        refund_id = f"ref_{order_id}"
        db.insert_row(
            "refunds",
            {
                "refund_id": refund_id,
                "order_id": order_id,
                "amount": amount,
                "status": "posted",
            },
        )
        return ctx.result(
            output={"refund_id": refund_id, "order_id": order_id, "amount": amount},
            state_updates={"refund_id": refund_id, "refund_written": True},
        )

    @node(id="ConfirmRefund", input_schema=OpenInput, output_schema=ConfirmOutput)
    def confirm_refund(ctx: NodeContext) -> object:
        order_id = str(ctx.state.get("order_id") or ctx.input.get("order_id", ""))
        if ctx.state.get("refund_written"):
            db.execute(
                "UPDATE orders SET status = ? WHERE order_id = ?",
                ("refunded", order_id),
            )
            status = "refunded"
        else:
            status = "unchanged"
        return ctx.result(output={"order_id": order_id, "status": status})

    @node(
        id="ErrorFallback",
        input_schema=OpenInput,
        output_schema=TextOutput,
        is_fallback=True,
    )
    def error_fallback(ctx: NodeContext) -> object:
        reason = ctx.state.get("reason", "refund failed")
        return ctx.result(output={"message": f"refund error: {reason}"})

    fsm = FSM(
        entry=validate_order,
        nodes=[validate_order, write_refund, confirm_refund, error_fallback],
        edges=[
            edge_deterministic("ValidateOrder", "WriteRefund"),
            edge_deterministic("WriteRefund", "ConfirmRefund"),
            edge_deterministic("ConfirmRefund", "End"),
            edge_fallback("ValidateOrder", "ErrorFallback"),
            edge_fallback("WriteRefund", "ErrorFallback"),
            edge_fallback("ConfirmRefund", "ErrorFallback"),
        ],
    )
    return OrderRefundScenario(fsm=fsm, db=db)
