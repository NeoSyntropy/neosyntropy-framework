from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from neosyntropy import (
    OpenInput,
    BoundTools,
    ControlManager,
    Edge,
    ExtractionError,
    Graph,
    Node,
    ProviderParameterExtractor,
    RunRequest,
    ToolCall,
    ToolCallingLoop,
    ToolRegistry,
    parse_tool_trigger,
    tool,
    tool_json_schema,
)


class LookupArgs(BaseModel):
    order_id: str
    include_history: bool = False


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(registry=registry)
    def lookup_order(args: LookupArgs) -> dict:
        """Look up an order by id."""
        if args.order_id == "boom":
            raise ValueError("order service unavailable")
        return {"order_id": args.order_id, "amount": 42.0}

    return registry


class ScriptedProvider:
    """Returns queued replies in order; falls back to a plain closing message."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, schema=None) -> str:
        self.prompts.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        return "Done."


def run_loop(provider, tools, **kwargs):
    loop = ToolCallingLoop(**kwargs)
    return asyncio.run(
        loop.run(
            provider=provider,
            messages=[{"role": "user", "content": "where is order ord_9?"}],
            tools=tools,
        )
    )


def bind(registry: ToolRegistry, allowed: tuple[str, ...]) -> BoundTools:
    return BoundTools(registry=registry, allowed=allowed, node_id="Support")


# --- trigger parsing ----------------------------------------------------------


def test_parse_tool_trigger():
    assert parse_tool_trigger("Let me check. <TOOL:lookup_order>") == (
        "Let me check.",
        "lookup_order",
    )
    assert parse_tool_trigger("No tools needed.") == ("No tools needed.", None)
    # Arguments never ride along with the trigger.
    visible, name = parse_tool_trigger('<TOOL:lookup_order>{"order_id": "x"}')
    assert (visible, name) == ("", "lookup_order")


def test_tool_json_schema_closes_the_object():
    schema = tool_json_schema(LookupArgs)
    assert schema["additionalProperties"] is False
    # Fields with defaults stay optional; only order_id must be supplied.
    assert set(schema["required"]) == {"order_id"}


# --- the loop -----------------------------------------------------------------


def test_loop_executes_tool_and_reinjects_result(registry):
    provider = ScriptedProvider(
        [
            "Checking that now. <TOOL:lookup_order>",
            '{"order_id": "ord_9", "include_history": false}',
            "Your order totals 42.0.",
        ]
    )
    result = run_loop(provider, bind(registry, ("lookup_order",)))

    assert [record.tool for record in result.records] == ["lookup_order"]
    record = result.records[0]
    assert record.ok
    assert record.arguments == {"order_id": "ord_9", "include_history": False}
    assert record.result == {"order_id": "ord_9", "amount": 42.0}
    assert "Your order totals 42.0." in result.text

    # The outcome was reinjected as a tool message the reasoner then saw.
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["ok"] is True


def test_extraction_prompt_carries_only_this_tool(registry):
    provider = ScriptedProvider(
        ["<TOOL:lookup_order>", '{"order_id": "ord_9", "include_history": false}']
    )
    run_loop(provider, bind(registry, ("lookup_order",)))
    extraction_prompt = provider.prompts[1]
    assert extraction_prompt.startswith("You extract tool arguments")
    assert "Tool: lookup_order" in extraction_prompt
    assert "Required JSON keys: order_id, include_history" in extraction_prompt
    assert extraction_prompt.endswith("JSON:")


def test_undeclared_tool_is_denied_and_never_executes(registry):
    provider = ScriptedProvider(["<TOOL:refund_everything>", "Sorry, I cannot."])
    result = run_loop(provider, bind(registry, ("lookup_order",)))

    assert len(result.records) == 1
    assert result.records[0].denied
    assert not result.records[0].ok
    # Nothing ran, and the refusal was fed back so the model could recover.
    assert registry.invocations == []
    assert any("not available here" in m["content"] for m in result.messages)


def test_invalid_arguments_never_reach_the_tool(registry):
    provider = ScriptedProvider(
        ["<TOOL:lookup_order>", '{"wrong_field": 1}', "I could not do that."]
    )
    result = run_loop(provider, bind(registry, ("lookup_order",)))

    assert not result.records[0].ok
    assert "failed lookup_order schema" in (result.records[0].error or "")
    assert registry.invocations == []


def test_non_json_extraction_is_recorded_not_raised(registry):
    provider = ScriptedProvider(["<TOOL:lookup_order>", "I think it is order nine."])
    result = run_loop(provider, bind(registry, ("lookup_order",)))
    assert not result.records[0].ok
    assert "no JSON object" in (result.records[0].error or "")


def test_failing_tool_is_recorded_as_a_failed_call(registry):
    provider = ScriptedProvider(
        [
            "<TOOL:lookup_order>",
            '{"order_id": "boom", "include_history": false}',
            "That lookup failed.",
        ]
    )
    result = run_loop(provider, bind(registry, ("lookup_order",)))
    record = result.records[0]
    assert not record.ok
    assert "order service unavailable" in (record.error or "")
    # The attempt is in the registry audit log.
    assert registry.invocations and not registry.invocations[0].ok


def test_low_confidence_arguments_are_not_executed(registry):
    class UnsureExtractor:
        async def extract(self, messages, tool) -> ToolCall:
            return ToolCall(tool=tool, arguments={"order_id": "ord_9"}, confidence=0.2)

    provider = ScriptedProvider(["<TOOL:lookup_order>", "I am not sure."])
    result = run_loop(
        provider, bind(registry, ("lookup_order",)), extractor=UnsureExtractor()
    )
    assert not result.records[0].ok
    assert "Low-confidence" in (result.records[0].error or "")
    assert registry.invocations == []


def test_duplicate_calls_are_rejected(registry):
    args = '{"order_id": "ord_9", "include_history": false}'
    provider = ScriptedProvider(
        ["<TOOL:lookup_order>", args, "<TOOL:lookup_order>", args, "All set."]
    )
    result = run_loop(provider, bind(registry, ("lookup_order",)))
    # Executed once; the replay was refused.
    assert len([r for r in result.records if r.ok]) == 1
    assert any("Duplicate tool call rejected." == m["content"] for m in result.messages)


def test_loop_is_bounded_by_max_tool_calls(registry):
    provider = ScriptedProvider(
        [
            "<TOOL:lookup_order>",
            '{"order_id": "a", "include_history": false}',
            "<TOOL:lookup_order>",
            '{"order_id": "b", "include_history": false}',
            "<TOOL:lookup_order>",
            '{"order_id": "c", "include_history": false}',
        ]
    )
    result = run_loop(provider, bind(registry, ("lookup_order",)), max_tool_calls=1)
    assert len([r for r in result.records if r.ok]) == 1
    assert any("Tool-call limit reached." == m["content"] for m in result.messages)


def test_extractor_reports_missing_tool(registry):
    extractor = ProviderParameterExtractor(ScriptedProvider([]), registry)
    with pytest.raises(KeyError):
        asyncio.run(extractor.extract([], "does_not_exist"))


def test_provider_extractor_raises_on_garbage(registry):
    extractor = ProviderParameterExtractor(ScriptedProvider(["nonsense"]), registry)
    with pytest.raises(ExtractionError):
        asyncio.run(extractor.extract([], "lookup_order"))


# --- end to end through the control manager -----------------------------------


def _graph_with_tool_node() -> Graph:
    from neosyntropy import TextOutput

    return Graph(
        nodes=[
            Node(
                id="Support",
                description="Answer order questions",
                provider="slm",
                prompt="Help the customer with their order.",
                tools=("lookup_order",),
                input_schema=OpenInput, output_schema=TextOutput,
            ),
            Node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput),
        ],
        edges=[
            Edge(source="Start", target="Support", kind="deterministic"),
            Edge(source="Support", target="End", kind="deterministic"),
        ],
    )


def test_provider_backed_node_calls_tools_end_to_end(registry):
    provider = ScriptedProvider(
        [
            "Let me look that up. <TOOL:lookup_order>",
            '{"order_id": "ord_9", "include_history": false}',
            "Your order totals 42.0.",
            '{"message": "Your order totals 42.0."}',
        ]
    )
    manager = ControlManager(
        _graph_with_tool_node(), providers={"slm": provider}, tools=registry
    )
    result = manager.run(RunRequest(intent="where is order ord_9?"))

    assert not result.rejected
    assert result.final_state == "Support"
    node_result = result.steps[0].results[0]
    assert node_result.tool_calls[0].ok
    assert node_result.tool_calls[0].result == {"order_id": "ord_9", "amount": 42.0}
    assert "42.0" in str(node_result.output)


def test_framework_seeds_provider_with_declared_prompt_only(registry):
    """Prompt assembly (tools catalog, evidence, state) is backend-owned."""
    provider = ScriptedProvider(
        ["No tools needed.", '{"message": "No tools needed."}']
    )
    manager = ControlManager(
        _graph_with_tool_node(), providers={"slm": provider}, tools=registry
    )
    manager.run(RunRequest(intent="hello"))
    prompt = provider.prompts[0]
    assert "Help the customer with their order." in prompt
    assert "Available tools:" not in prompt
    assert "Prior node findings" not in prompt
    assert "User Intent:" not in prompt
