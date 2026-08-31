"""Tests for @function_calling and @workflow decorators."""

from __future__ import annotations

from pydantic import BaseModel

from neosyntropy import ReasoningStep, SchemaNode, ToolRegistry, Workflow, function_calling, workflow
from neosyntropy.monitor.graph.manifest import control_graph_manifest, graph_manifest


class SampleInput(BaseModel):
    user_request: str


class GreetParams(BaseModel):
    name: str


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


def test_graph_manifest_includes_decorator_and_function_source():
    schema_node = SchemaNode(
        id="greet_schema",
        input_schema=SampleInput,
        output_schema=GreetParams,
        prompt="Extract greeting parameters.",
    )
    fallback_node = SchemaNode(
        id="greet_fallback",
        input_schema=SampleInput,
        output_schema=schema_node.output_schema,
        prompt="Fallback node",
        is_fallback=True,
    )
    fsm = Workflow([schema_node], fallback=fallback_node)
    fsm.function_source = {
        "function_name": "greet",
        "function_module": "tests.test_decorators",
        "source_code": "def greet(params: GreetParams) -> str:\n    return f'Hello {params.name}'\n",
    }
    fsm.decorator = "function_calling"

    manifest = graph_manifest(fsm)
    assert manifest["decorator"] == "function_calling"
    assert manifest["function_source"]
    assert manifest["function_source"][0]["function_name"] == "greet"
    assert "def greet" in manifest["function_source"][0]["source_code"]

    control = control_graph_manifest(fsm)
    assert control["decorator"] == "function_calling"
    assert "def greet" in control["function_source"][0]["source_code"]
