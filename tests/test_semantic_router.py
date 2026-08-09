from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    EmptyOutput,
    TextOutput,
    DeterministicRouter,
    SemanticRouter,
    node,
    edge_deterministic,
    edge_fallback,
)

client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="http://127.0.0.1:8001",
    telemetry_timeout=20.0,
)


class CustomerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    token: str


@node(id="CheckAuth", input_schema=CustomerRequest, output_schema=EmptyOutput)
def check_auth(ctx):
    valid = ctx.input["token"].startswith("tok_")
    return ctx.result(output={}, state_updates={"token_valid": valid})


@node(id="ProcessRefund", input_schema=OpenInput, output_schema=EmptyOutput)
def process_refund(ctx):
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")


@node(id="LookupOrderStatus", input_schema=OpenInput, output_schema=EmptyOutput)
def lookup_order_status(ctx):
    return ctx.result(output={}, state_updates={"status_fetched": True}, next_state="End")


@node(id="RequireLogin", input_schema=OpenInput, output_schema=TextOutput)
def require_login(ctx):
    return ctx.result(output={"message": "Please log in first."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


# DeterministicRouter: hard auth gate first
auth_gate = DeterministicRouter(
    id="AuthGate",
    input_schema=CustomerRequest,
    rules=[
        (lambda ctx: ctx.state.get("token_valid") is True, "CustomerIntent"),
        (lambda ctx: ctx.state.get("token_valid") is False, "RequireLogin"),
    ],
)

# SemanticRouter: model picks the intent lane
intent_router = SemanticRouter(
    id="CustomerIntent",
    routes={
        "refund": process_refund,
        "status": lookup_order_status,
    },
    fallback_node=out_of_scope,
)

fsm = FSM(
    entry=check_auth,
    nodes=[check_auth, process_refund, lookup_order_status, require_login, out_of_scope],
    routers=[auth_gate, intent_router],
    edges=[
        edge_deterministic("CheckAuth", "AuthGate"),
        edge_deterministic("AuthGate", "CustomerIntent"),
        edge_deterministic("AuthGate", "RequireLogin"),
        edge_deterministic("ProcessRefund", "End"),
        edge_deterministic("LookupOrderStatus", "End"),
        edge_deterministic("RequireLogin", "End"),
        edge_fallback("CustomerIntent", "OutOfScope"),
    ],
)


def test_semantic_router_refund():
    result = fsm.run(
        CustomerRequest(text="I want a refund for order ord_42", token="tok_abc123"),
        state={},
        client=client,
    )
    print(f"Refund run final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("refund_issued") is True


def test_semantic_router_status():
    result = fsm.run(
        CustomerRequest(text="Check status of order ord_42", token="tok_abc123"),
        state={},
        client=client,
    )
    print(f"Status run final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("status_fetched") is True


def test_semantic_router_unauthorized():
    result = fsm.run(
        CustomerRequest(text="I want a refund for order ord_42", token="invalid_token"),
        state={},
        client=client,
    )
    print(f"Unauthorized run final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("token_valid") is False
    assert result.state.get("refund_issued") is not True


if __name__ == "__main__":
    test_semantic_router_refund()
    test_semantic_router_status()
    test_semantic_router_unauthorized()
