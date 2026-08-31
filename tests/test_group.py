from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    EmptyOutput,
    TextOutput,
    Group,
    DeterministicRouter,
    node,
    edge_deterministic,
    edge_fallback,
)

client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="https://api.neosyntropy.com",
    telemetry_timeout=20.0,
)

# 1. Author a named Group with internal routing
billing = Group(name="billing")


class CardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    card_number: str


@billing.node(id="ValidateCard", input_schema=CardRequest, output_schema=EmptyOutput)
def validate(ctx):
    valid = ctx.input["card_number"].startswith("4")  # e.g., Visa card
    return ctx.result(output={}, state_updates={"card_valid": valid})


@billing.node(id="ProcessPayment", input_schema=OpenInput, output_schema=EmptyOutput)
def pay(ctx):
    return ctx.result(output={}, state_updates={"paid": True}, next_state="End")


@billing.node(id="RejectCard", input_schema=OpenInput, output_schema=TextOutput)
def reject(ctx):
    return ctx.result(output={"message": "Card rejected."}, next_state="End")


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
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process request."})


# 2. Build the parent FSM using the Group
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


def test_group_valid_card():
    result = fsm.run(
        CardRequest(card_number="4111222233334444"),
        state={},
        client=client,
    )
    print(f"Valid card final state: {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("card_valid") is True
    assert result.state.get("paid") is True


def test_group_invalid_card():
    result = fsm.run(
        CardRequest(card_number="5111222233334444"),
        state={},
        client=client,
    )
    print(f"Invalid card final state: {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("card_valid") is False
    assert result.state.get("paid") is not True


if __name__ == "__main__":
    test_group_valid_card()
    test_group_invalid_card()
