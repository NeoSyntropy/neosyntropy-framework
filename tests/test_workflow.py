from pydantic import BaseModel, ConfigDict
from neosyntropy import (
    Client,
    Workflow,
    node,
    OpenInput,
    EmptyOutput,
    TextOutput,
)

client = Client(
    api_key="nsk_ed3e7cad3792_b-4ZWhR5dZRnFv3GVPEc8Vddnwercpqfuqfk-uTqyZk",
    project_id="f63ebb40-c287-493e-972a-ac66546f92db",
    base_url="http://127.0.0.1:8001",
    telemetry_timeout=20.0,
)


class Step1Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: str


@node(id="Step1", input_schema=Step1Input, output_schema=EmptyOutput)
def step1(ctx):
    return ctx.result(output={}, state_updates={"step1_done": True})


@node(id="Step2", input_schema=OpenInput, output_schema=EmptyOutput)
def step2(ctx):
    return ctx.result(output={}, state_updates={"step2_done": True}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope."})


# Automatically wires: Step1 -> Step2 -> End
# Fallback: Step1 -> OutOfScope
fsm = Workflow(
    sequence=[step1, step2],
    fallback=out_of_scope,
)


def test_workflow():
    result = fsm.run(
        Step1Input(task_name="Verify workflow concept"),
        state={},
        client=client,
    )
    print(f"Workflow final state: {result.final_state}")
    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("step1_done") is True
    assert result.state.get("step2_done") is True


if __name__ == "__main__":
    test_workflow()
