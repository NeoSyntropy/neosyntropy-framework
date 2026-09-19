"""Failed handler nodes must not complete the FSM or be retried to End."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from neosyntropy import EmptyOutput, OpenInput, TextOutput, Workflow, node


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: str


@node(id="Boom", input_schema=TaskInput, output_schema=EmptyOutput)
def boom(ctx):
    raise RuntimeError("handler exploded")


@node(id="Step1", input_schema=TaskInput, output_schema=EmptyOutput)
def step1(ctx):
    return ctx.result(output={}, state_updates={"step1_done": True})


@node(id="BoomLast", input_schema=OpenInput, output_schema=EmptyOutput)
def boom_last(ctx):
    raise RuntimeError("last step exploded")


@node(id="Done", input_schema=OpenInput, output_schema=EmptyOutput)
def done(ctx):
    return ctx.result(output={}, state_updates={"done": True})


@node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def fallback(ctx):
    return ctx.result(output={"message": "fallback"})


def test_failed_only_step_is_not_marked_completed():
    fsm = Workflow([boom], fallback=fallback)
    result = fsm.run(TaskInput(task_name="only"), state={})

    assert not result.rejected
    assert result.completed is False
    assert result.final_state != "End"
    assert result.steps
    assert result.steps[0].results[0].status == "failed"
    assert "handler exploded" in (result.steps[0].results[0].error or "")


def test_failed_last_step_is_not_marked_completed():
    fsm = Workflow([step1, boom_last], fallback=fallback)
    result = fsm.run(TaskInput(task_name="last"), state={})

    assert not result.rejected
    assert result.completed is False
    assert result.final_state != "End"
    assert result.state.get("step1_done") is True
    last = result.steps[-1].results[-1]
    assert last.status == "failed"
    assert "last step exploded" in (last.error or "")


def test_failed_first_step_does_not_retry_until_max_cycles():
    fsm = Workflow([boom, boom_last], fallback=fallback)
    result = fsm.run(TaskInput(task_name="first"), state={}, max_cycles=8)

    assert result.completed is False
    assert result.final_state != "End"
    failed_runs = [
        item
        for step in result.steps
        for item in step.results
        if item.node_id == "Boom" and item.status == "failed"
    ]
    assert len(failed_runs) == 1


def test_successful_workflow_still_reaches_end():
    fsm = Workflow([step1, done], fallback=fallback)
    result = fsm.run(TaskInput(task_name="ok"), state={})

    assert not result.rejected
    assert result.completed is True
    assert result.final_state == "End"
    assert result.state.get("step1_done") is True
    assert result.state.get("done") is True
