from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    EmptyOutput,
    TextOutput,
    DeterministicRouter,
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


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    account_age_days: int


@node(id="CheckPolicy", input_schema=PolicyRequest, output_schema=EmptyOutput)
def check_policy(ctx):
    """Evaluate eligibility and write the result to state."""
    eligible = ctx.input["account_age_days"] >= 30
    return ctx.result(output={}, state_updates={"eligible": eligible})


@node(id="ApproveRequest", input_schema=OpenInput, output_schema=EmptyOutput)
def approve_request(ctx):
    return ctx.result(output={}, state_updates={"approved": True}, next_state="End")


@node(id="DenyRequest", input_schema=OpenInput, output_schema=TextOutput)
def deny_request(ctx):
    return ctx.result(output={"message": "Account too new — request denied."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process request."})


# Hard rule: reads ctx.state set by CheckPolicy
eligibility = DeterministicRouter(
    id="EligibilityGate",
    rules=[
        (lambda ctx: ctx.state.get("eligible") is True, "ApproveRequest"),
        (lambda ctx: ctx.state.get("eligible") is False, "DenyRequest"),
    ],
)

fsm = FSM(
    entry=check_policy,
    nodes=[check_policy, approve_request, deny_request, out_of_scope],
    routers=[eligibility],
    edges=[
        edge_deterministic("CheckPolicy", "EligibilityGate"),
        edge_deterministic("EligibilityGate", "ApproveRequest"),
        edge_deterministic("EligibilityGate", "DenyRequest"),
        edge_deterministic("ApproveRequest", "End"),
        edge_deterministic("DenyRequest", "End"),
        edge_fallback("CheckPolicy", "OutOfScope"),
    ],
)


def test_deterministic_router_eligible():
    result = fsm.run(
        PolicyRequest(text="I'd like a refund", account_age_days=45),
        state={},
        client=client,
    )
    print(f"Eligible run final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("eligible") is True
    assert result.state.get("approved") is True
    assert "EligibilityGate" in result.audit.committed_transitions


def test_deterministic_router_ineligible():
    result = fsm.run(
        PolicyRequest(text="I'd like a refund", account_age_days=15),
        state={},
        client=client,
    )
    print(f"Ineligible run final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("eligible") is False
    assert result.state.get("approved") is not True
    assert "EligibilityGate" in result.audit.committed_transitions


if __name__ == "__main__":
    test_deterministic_router_eligible()
    test_deterministic_router_ineligible()
