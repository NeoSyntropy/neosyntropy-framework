"""Tests for @function_calling and @workflow decorators."""

from __future__ import annotations

from pydantic import BaseModel

from neosyntropy import ReasoningStep, ToolRegistry, function_calling, workflow


class SampleInput(BaseModel):
    user_request: str


class MockClient:
    def __init__(self):
        self.logs = []

    def log_run(self, project_id: str, input: str, output: str):
        self.logs.append({"project_id": project_id, "input": input, "output": output})


def test_function_calling_decorator_creation():
    client = MockClient()

    @function_calling(
        prompt="Extract parameters for processing order.",
        input_schema=SampleInput,
        client=client,
        project_id="test_project",
    )
    def my_order_func(num_items: int, item_name: str):
        return f"Ordered {num_items} x {item_name}"

    assert callable(my_order_func)


def test_workflow_decorator_creation():
    client = MockClient()

    @workflow(
        prompt="Determine inventory and extract order.",
        input_schema=SampleInput,
        reasoning_steps=[
            ReasoningStep(instruction="Check stock for item", tools=["check_stock"]),
        ],
        client=client,
        project_id="test_project",
        tools=ToolRegistry(),
    )
    def my_workflow_func(num_items: int, item_name: str):
        return f"Workflow ordered {num_items} x {item_name}"

    assert callable(my_workflow_func)


def test_async_function_calling_decorator_creation():
    client = MockClient()

    @function_calling(
        prompt="Extract parameters.",
        input_schema=SampleInput,
        client=client,
        project_id="test_project",
    )
    async def my_async_func(count: int):
        return count * 2

    assert callable(my_async_func)
