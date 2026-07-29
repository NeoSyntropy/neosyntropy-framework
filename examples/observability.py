"""Inspect the safe graph manifest or plug in a local lifecycle observer."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neosyntropy import ControlManager, Edge, Graph, graph_manifest, node


@node(id="Greet")
def greet(ctx):
    return ctx.result(output="Hello", next_state="End")


@node(id="OutOfScope", is_fallback=True)
def out_of_scope(ctx):
    return ctx.result(output="Out of scope")


graph = Graph(
    nodes=[greet, out_of_scope],
    edges=[
        Edge(source="Start", target="Greet", label="first"),
        Edge(source="Greet", target="End", label="complete"),
    ],
)


class ConsoleObserver:
    async def run_started(
        self, *, request_id: str, initial_state: str, manifest: Mapping[str, Any]
    ) -> str:
        print("run_started", request_id, initial_state, manifest)
        return request_id

    async def event(
        self, run_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        print(event_type, payload)

    async def run_finished(
        self, run_id: str, *, status: str, final_state: str
    ) -> None:
        print("run_finished", status, final_state)


print(graph_manifest(graph))
result = ControlManager(graph, observer=ConsoleObserver()).run(
    {"intent": "say hello"}
)
print(result.final_state)
