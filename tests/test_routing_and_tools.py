from __future__ import annotations

import pytest
from pydantic import BaseModel

from neosyntropy import (
    Candidate,
    DeterministicRouter,
    RunContext,
    ToolNotAllowedError,
    ToolRegistry,
    Topology,
    tool,
)
from neosyntropy.tools.registry import BoundTools

from .conftest import build_graph


def make_context(current_state: str = "Start") -> RunContext:
    return RunContext(
        request_id="req-1",
        intent="refund my order",
        current_state=current_state,
    )


def make_candidates(graph) -> list[Candidate]:
    return [
        Candidate(
            node_id=item.id,
            name=item.name,
            prerequisites=item.prerequisites,
            is_fallback=item.is_fallback,
        )
        for item in graph.nodes.values()
    ]


# --- deterministic router -----------------------------------------------------


def test_deterministic_router_follows_edge_priority():
    graph = build_graph()
    candidates = make_candidates(graph)
    plan = _route(DeterministicRouter(graph), make_context("Start"), candidates)
    assert plan.topology == Topology.SEQUENTIAL
    chosen = candidates[plan.execution_plan[0][0]]
    assert chosen.node_id == "VerifyIdentity"


def test_deterministic_router_falls_back_when_nothing_is_legal():
    graph = build_graph()
    candidates = make_candidates(graph)
    plan = _route(DeterministicRouter(graph), make_context("End"), candidates)
    assert plan.topology == Topology.FALLBACK
    assert candidates[plan.execution_plan[0][0]].is_fallback


def _route(router, context, candidates):
    import asyncio

    return asyncio.run(router.route(context, candidates))


# --- tool registry -------------------------------------------------------------


class AddToCartArgs(BaseModel):
    product_id: str
    quantity: int


def test_tool_registration_and_invocation_log():
    registry = ToolRegistry()

    @tool(registry=registry)
    def add_to_cart(args: AddToCartArgs) -> dict:
        """Add a quantity of a product to the active cart."""
        return {"added": args.quantity}

    assert add_to_cart({"product_id": "p1", "quantity": 2}) == {"added": 2}
    assert registry.tools["add_to_cart"].json_schema["additionalProperties"] is False
    assert len(registry.invocations) == 1
    assert registry.invocations[0].ok

    # Invalid args are logged as failed invocations and raise.
    with pytest.raises(ValueError):
        add_to_cart({"product_id": "p1", "quantity": "many"})
    assert not registry.invocations[1].ok


def test_bound_tools_enforce_the_node_allow_list():
    registry = ToolRegistry()

    @tool(registry=registry)
    def lookup(args: AddToCartArgs) -> str:
        """Lookup."""
        return "ok"

    bound = BoundTools(registry=registry, allowed=(), node_id="N")
    with pytest.raises(ToolNotAllowedError, match="not allowed on node 'N'"):
        bound.invoke("lookup", {"product_id": "p", "quantity": 1})

    allowed = BoundTools(registry=registry, allowed=("lookup",), node_id="N")
    assert allowed.invoke("lookup", {"product_id": "p", "quantity": 1}) == "ok"
