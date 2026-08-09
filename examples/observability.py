"""Inspect the safe graph manifest or plug in a local lifecycle observer."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neosyntropy import (
    OpenInput,
    ControlManager,
    Edge,
    FSM,
    TextOutput,
    graph_manifest,
    node,
)


@node(id="Greet", input_schema=OpenInput, output_schema=TextOutput)
def greet(ctx):
    return ctx.result(output={"message": "Hello"}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope"})


graph = FSM(
        entry="ENTRY",
    nodes=[greet, out_of_scope],
    edges=[
        Edge(source="ENTRY", target="Greet", kind="deterministic"),
        Edge(source="Greet", target="End", kind="deterministic"),
    ]
)


class ConsoleObserver:
    async def run_started(
        self,
        *,
        request_id: str,
        initial_state: str,
        manifest: Mapping[str, Any],
        input: Mapping[str, Any] | None = None,
    ) -> str:
        print("run_started", request_id, initial_state, manifest, input)
        return request_id

    async def event(
        self, run_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        print(event_type, payload)

    async def run_finished(
        self,
        run_id: str,
        *,
        status: str,
        final_state: str,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        print("run_finished", status, final_state, output)


print(graph_manifest(graph))
result = ControlManager(graph, observer=ConsoleObserver()).run(
    {"input": {"text": "say hello"}}
)
print(result.final_state)
