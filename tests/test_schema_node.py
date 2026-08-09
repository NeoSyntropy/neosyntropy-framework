from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
)

client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="http://127.0.0.1:8001",
    telemetry_timeout=20.0,
)

VERTEX_MODEL = "gemini-2.5-flash"


class CustomerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class SupportTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    issue: str
    priority: str


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


extract_ticket = SchemaNode(
    id="ExtractTicket",
    input_schema=CustomerMessage,
    output_schema=SupportTicket,
    provider=VERTEX_MODEL,
    prompt=(
        "Extract a structured support ticket from the customer message. "
        "Set priority to 'high' if the customer mentions urgency or damage."
    ),
)

fsm = FSM(
    entry=extract_ticket,
    nodes=[extract_ticket, out_of_scope],
    edges=[
        edge_deterministic("ExtractTicket", "End"),
        edge_fallback("ExtractTicket", "OutOfScope"),
    ],
)


def test_schema_node():
    result = fsm.run(
        CustomerMessage(text="My order ord_99 arrived broken, need help urgently!"),
        state={},
        client=client,
    )
    print(f"Final state : {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"

    found_output = False
    for step in result.steps:
        for item in step.results:
            print(f"Node: {item.node_id}, Output: {item.output}")
            if item.node_id == "ExtractTicket":
                assert "order_id" in item.output
                assert "issue" in item.output
                found_output = True
    assert found_output


if __name__ == "__main__":
    test_schema_node()
