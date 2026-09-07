"""Debug: trace FSM routing for the python node example."""
import os
from pathlib import Path

tests_env = Path("tests/.env")
for line in tests_env.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    os.environ[k.strip()] = v.strip().strip("'\"")

from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client, FSM, NodeContext, OpenInput, TextOutput,
    edge_deterministic, node,
)

class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    customer: str
    amount: float
    currency: str = "USD"

class ValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str

class ConfirmationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str

@node(id="ValidateOrder", input_schema=OrderInput, output_schema=ValidationOutput)
def validate_order(ctx: NodeContext):
    data = ctx.input
    order_id = data.get("order_id", "").strip()
    amount = float(data.get("amount", 0))
    customer = data.get("customer", "").strip()
    print(f"[ValidateOrder] state before: {ctx.state}")
    if not order_id or amount <= 0:
        print(f"[ValidateOrder] FAIL — setting validated=False")
        return ctx.result(
            output={"valid": False, "reason": f"invalid: order_id={order_id!r} amount={amount}"},
            state_updates={"validated": False, "reason": f"order_id={order_id!r} amount={amount}"},
        )
    print(f"[ValidateOrder] OK — setting validated=True")
    return ctx.result(
        output={"valid": True, "reason": "ok"},
        state_updates={"validated": True, "order_id": order_id, "customer": customer, "amount": amount},
    )

@node(id="ConfirmOrder", input_schema=OpenInput, output_schema=ConfirmationOutput)
def confirm_order(ctx: NodeContext):
    print(f"[ConfirmOrder] state: {ctx.state}")
    return ctx.result(output={"message": f"Confirmed for {ctx.state.get('customer', '?')}"})

@node(id="InvalidOrder", input_schema=OpenInput, output_schema=TextOutput, is_fallback=True)
def invalid_order(ctx: NodeContext):
    print(f"[InvalidOrder] state: {ctx.state}")
    return ctx.result(output={"message": f"Rejected: {ctx.state.get('reason', 'unknown')}"})

client = Client(api_key=os.environ["NEOSYNTROPY_API_KEY"], base_url="http://127.0.0.1:8000")
client.project_id = "fa8c4e01-fb23-4357-a96e-83f6c872f69e"
client._backend.project_id = client.project_id

fsm = FSM(
    entry=validate_order,
    nodes=[validate_order, confirm_order, invalid_order],
    edges=[
        edge_deterministic("ValidateOrder", "ConfirmOrder", guard=lambda s: s.get("validated", False)),
        edge_deterministic("ValidateOrder", "InvalidOrder", guard=lambda s: not s.get("validated", True)),
        edge_deterministic("ConfirmOrder", "End"),
        edge_deterministic("InvalidOrder", "End"),
    ],
)

print("\n=== VALID ORDER ===")
result = fsm.run(
    OrderInput(order_id="ORD-001", customer="Alice", amount=149.99),
    state={}, client=client,
)
print(f"final_state: {result.final_state}")
for step in result.steps:
    for item in step.results:
        print(f"  [{item.node_id}] status={item.status} output={item.output}")
