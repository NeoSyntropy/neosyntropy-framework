from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from neosyntropy import (
    Candidate,
    DeterministicRouter,
    RouterError,
    RunContext,
    SlmRouter,
    ToolNotAllowedError,
    ToolRegistry,
    Topology,
    tool,
)
from neosyntropy.routing.slm import REJECTION_TEXT, build_instruction, build_output_schema
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


# --- SLM router wire contract --------------------------------------------------


def test_instruction_matches_trained_template():
    context = make_context("VerifyIdentity")
    names = [f"Node{i}" for i in range(9)] + [REJECTION_TEXT]
    instruction = build_instruction(context, names, category="commerce")
    assert instruction.startswith("Industry Category: [commerce]\n")
    assert "Current FSM State: [VerifyIdentity]" in instruction
    assert 'User Intent: "refund my order"' in instruction
    assert "[0]: Node0" in instruction
    assert f"[9]: {REJECTION_TEXT}" in instruction


def test_output_schema_matches_trained_contract():
    schema = build_output_schema(list(range(10)))
    assert schema["properties"]["topology"]["enum"] == [
        "parallel",
        "sequential",
        "fallback",
    ]
    assert schema["properties"]["execution_plan"]["items"]["items"]["enum"] == list(
        range(10)
    )
    assert schema["additionalProperties"] is False


class ScriptedProvider:
    def __init__(self, payload: dict):
        self.payload = payload
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, schema=None) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def test_slm_router_maps_slots_and_hybrid_shape():
    graph = build_graph()
    candidates = make_candidates(graph)
    provider = ScriptedProvider(
        {
            "reasoning": "verify and calculate in parallel, then issue",
            "topology": "sequential",
            "execution_plan": [[0, 1], [2]],
        }
    )
    plan = _route(SlmRouter(provider, category="commerce"), make_context(), candidates)
    # Wire "sequential with a parallel step" maps to internal HYBRID.
    assert plan.topology == Topology.HYBRID
    assert plan.execution_plan == [[0, 1], [2]]
    assert provider.prompts[0].startswith("### Instruction:\n")


def test_slm_router_maps_slot_nine_to_the_fallback():
    graph = build_graph()
    candidates = make_candidates(graph)
    provider = ScriptedProvider(
        {"reasoning": "out of scope", "topology": "fallback", "execution_plan": [[9]]}
    )
    plan = _route(SlmRouter(provider), make_context(), candidates)
    assert plan.topology == Topology.FALLBACK
    assert candidates[plan.execution_plan[0][0]].is_fallback


def test_slm_router_rejects_padding_slots():
    graph = build_graph()
    candidates = make_candidates(graph)
    provider = ScriptedProvider(
        {"reasoning": "?", "topology": "sequential", "execution_plan": [[7]]}
    )
    with pytest.raises(RouterError, match="padding candidate"):
        _route(SlmRouter(provider), make_context(), candidates)


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
