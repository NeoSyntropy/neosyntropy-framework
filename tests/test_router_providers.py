"""Router provider wire + shared routing-plan schema mapping."""

from __future__ import annotations

from neosyntropy import (
    FSM,
    OpenInput,
    SchemaNode,
    SemanticRouter,
    TextOutput,
    edge_deterministic,
)
from neosyntropy.backend import _control_api_graph
from neosyntropy.monitor.graph.manifest import _router_providers


def test_control_api_graph_includes_router_providers() -> None:
    wire = _control_api_graph(
        {
            "schema_version": 1,
            "entry": "PhaseRouter",
            "input_schema": {"type": "object"},
            "nodes": [
                {
                    "id": "DoWork",
                    "kind": "schema",
                    "is_fallback": False,
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "OutOfScope",
                    "kind": "schema",
                    "is_fallback": True,
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "PhaseRouter",
                    "kind": "router",
                    "output_schema": None,
                },
            ],
            "routers": ["PhaseRouter"],
            "router_providers": {"PhaseRouter": "gemini-2.5-flash"},
            "edges": [],
            "groups": [],
        }
    )
    assert wire["routers"] == ["PhaseRouter"]
    assert wire["router_providers"] == {"PhaseRouter": "gemini-2.5-flash"}
    assert "PhaseRouter" not in {n["id"] for n in wire["nodes"]}


def test_router_providers_from_semantic_router() -> None:
    leaf = SchemaNode(
        id="DoA",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="a",
        provider="gemini-2.5-flash",
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
        provider="gemini-2.5-flash",
    )
    router = SemanticRouter(
        id="PhaseRouter",
        input_schema=OpenInput,
        routes={"a": leaf},
        fallback_node=fallback,
        provider="gemini-2.5-flash",
    )
    fsm = FSM(
        entry=router,
        nodes=[leaf, fallback],
        routers=[router],
        edges=[edge_deterministic("DoA", "End")],
        validate_reachability=False,
    )
    assert _router_providers(fsm)["PhaseRouter"] == "gemini-2.5-flash"
