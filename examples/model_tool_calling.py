"""A provider-backed node calling tools: trigger, extract, validate, reinject.

Run from the repository root::

    python examples/model_tool_calling.py

The node has no Python handler — it runs on a model. The stub provider below
stands in for a real one and emits the same wire format a trained reasoner
does: prose, then ``<TOOL:name>`` with no arguments. Arguments come from a
separate schema-constrained extraction step.

Three scenarios: a successful call, a tool the node was never granted, and
arguments that do not fit the schema. In both failure cases nothing executes
and the model is told why.
"""
from __future__ import annotations

from pydantic import BaseModel

from neosyntropy import (
    OpenInput,
    ControlManager,
    Edge,
    FSM,
    ReasoningNode,
    RunRequest,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    tool,
)

# --- Tools: the pydantic model is the extraction contract --------------------

registry = ToolRegistry()


class LookupOrderArgs(BaseModel):
    order_id: str
    include_history: bool = False


@tool(registry=registry)
def lookup_order(args: LookupOrderArgs) -> dict:
    """Look up an order by id and return its paid amount."""
    return {"order_id": args.order_id, "amount": 42.0}


class RefundArgs(BaseModel):
    order_id: str
    amount: float


@tool(registry=registry)
def issue_refund(args: RefundArgs) -> dict:
    """Issue a refund. Not granted to the support node."""
    return {"refunded": args.amount}


# --- A stub model speaking the trained wire format ---------------------------


class ScriptedProvider:
    """Replays queued replies so the example runs without a model."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def generate(self, prompt: str, *, schema: dict | None = None) -> str:
        del prompt, schema
        return self.replies.pop(0) if self.replies else "Anything else?"


# --- FSM: tools are granted per node, and the node prompt shows only those --


def build_graph() -> FSM:
    return FSM(
        nodes=[
            ReasoningNode(
                id="Support",
                description="Answer order questions",
                provider="inference",
                prompt="Help the customer with their order.",
                tools=("lookup_order",),
                input_schema=OpenInput,
            ),
            SchemaNode(
                id="OutOfScope",
                is_fallback=True,
                input_schema=OpenInput,
                output_schema=TextOutput,
                prompt="Politely refuse out-of-scope requests.",
            ),
        ],
        edges=[
            Edge(source="Start", target="Support", kind="deterministic"),
            Edge(source="Support", target="End", kind="deterministic"),
        ],
    )


def show(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(f"final:  {result.final_state} (rejected={result.rejected})")
    if result.rejection:
        print(f"REJECTED: {result.rejection}")
    for step in result.steps:
        for item in step.results:
            print(f"output: {item.output}")
            for record in item.tool_calls:
                verdict = "ok" if record.ok else ("denied" if record.denied else "failed")
                print(f"  [{verdict}] {record.tool} {record.arguments} {record.error or ''}")


def run(title: str, replies: list[str], intent: str) -> None:
    manager = ControlManager(
        build_graph(), providers={"inference": ScriptedProvider(replies)}, tools=registry
    )
    show(title, manager.run(RunRequest(intent=intent)))


def main() -> None:
    run(
        "Granted tool, valid arguments",
        [
            "Let me look that up. <TOOL:lookup_order>",
            '{"order_id": "ord_9", "include_history": false}',
            "Your order ord_9 totals 42.0.",
        ],
        "where is my order ord_9?",
    )

    # issue_refund exists in the registry but was never granted to this node.
    run(
        "Ungranted tool: denied, never executed",
        [
            "I will refund this. <TOOL:issue_refund>",
            "I am not able to issue refunds here.",
        ],
        "refund my order ord_9",
    )

    # The extractor's output does not fit LookupOrderArgs, so the tool is
    # never reached and the model is told why.
    run(
        "Invalid arguments: tool never reached",
        [
            "<TOOL:lookup_order>",
            '{"order": 9}',
            "I could not read that order id.",
        ],
        "check order nine",
    )


if __name__ == "__main__":
    main()
