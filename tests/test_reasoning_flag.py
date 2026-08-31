"""Construction tests for low/high reasoning on validation and semantic routers."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from neosyntropy.core.edge import edge_deterministic
from neosyntropy.core.graph import FSM
from neosyntropy.core.node import (
    COMBINE_SCHEMA_SUFFIX,
    CombineNode,
    Node,
    SchemaNode,
    SemanticValidationNode,
    ValidationResult,
)
from neosyntropy.core.routing.semantic import SemanticRouter
from neosyntropy.core.schemas import OpenInput, TextOutput


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def test_semantic_validation_low_returns_schema_node() -> None:
    guard = SemanticValidationNode(
        "sql_check",
        input_schema=_In,
        prompt="Return valid=false if the SQL is unsafe.",
    )
    assert isinstance(guard, Node)
    assert not isinstance(guard, CombineNode)
    assert guard.kind == "schema"
    assert guard.mode == "schema_extraction"
    assert guard.tools == ()
    assert guard.output_model is ValidationResult


def test_semantic_validation_high_returns_combine_node() -> None:
    guard = SemanticValidationNode(
        "sql_check",
        input_schema=_In,
        prompt="Think through whether the SQL is safe, then decide.",
        reasoning="high",
    )
    assert isinstance(guard, CombineNode)
    nodes, links = guard.expand()
    assert [item.id for item in nodes] == ["sql_check", "sql_check.Schema"]
    assert nodes[0].kind == "combine_part"
    assert nodes[0].mode == "reasoning"
    assert nodes[1].output_model is ValidationResult
    assert len(links) == 1
    assert links[0].source == "sql_check"
    assert links[0].target == "sql_check.Schema"


def test_semantic_validation_tools_force_high() -> None:
    guard = SemanticValidationNode(
        "claim_check",
        input_schema=_In,
        prompt="Use fetch_order, then decide.",
        reasoning="low",
        tools=("fetch_order",),
    )
    assert isinstance(guard, CombineNode)
    assert tuple(guard.tools) == ("fetch_order",)


def test_semantic_validation_rejects_unknown_reasoning() -> None:
    with pytest.raises(ValueError, match="reasoning must be"):
        SemanticValidationNode(
            "bad",
            input_schema=_In,
            prompt="check",
            reasoning="medium",  # type: ignore[arg-type]
        )


def test_semantic_router_low_has_no_expand() -> None:
    leaf = SchemaNode(
        id="DoA",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="a",
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    router = SemanticRouter(
        id="PhaseRouter",
        input_schema=OpenInput,
        routes={"a": leaf},
        fallback_node=fallback,
    )
    assert router.reasoning == "low"
    assert router.router_state_id == "PhaseRouter"
    assert router.expand() == ([], [])
    sources = {edge.source for edge in router.compile()}
    assert sources == {"PhaseRouter"}


def test_semantic_router_high_expands_reasoning_node() -> None:
    leaf = SchemaNode(
        id="DoA",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="a",
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    router = SemanticRouter(
        id="PhaseRouter",
        input_schema=OpenInput,
        routes={"a": leaf},
        fallback_node=fallback,
        reasoning="high",
        prompt="Reason about the intent, then the router will pick a label.",
    )
    assert router.router_state_id == f"PhaseRouter{COMBINE_SCHEMA_SUFFIX}"
    nodes, links = router.expand()
    assert len(nodes) == 1
    assert nodes[0].id == "PhaseRouter"
    assert nodes[0].mode == "reasoning"
    assert links[0].source == "PhaseRouter"
    assert links[0].target == "PhaseRouter.Schema"
    sources = {edge.source for edge in router.compile()}
    assert sources == {"PhaseRouter.Schema"}


def test_semantic_router_tools_force_high_and_fsm_expands() -> None:
    leaf = SchemaNode(
        id="DoA",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="a",
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    router = SemanticRouter(
        id="PhaseRouter",
        input_schema=OpenInput,
        routes={"a": leaf},
        fallback_node=fallback,
        tools=("lookup_order",),
    )
    assert router.reasoning == "high"
    assert tuple(router.tools) == ("lookup_order",)

    fsm = FSM(
        entry=router,
        nodes=[leaf, fallback],
        routers=[router],
        edges=[edge_deterministic("DoA", "End")],
        validate_reachability=False,
    )
    assert "PhaseRouter" in fsm.nodes
    assert fsm.nodes["PhaseRouter"].mode == "reasoning"
    assert fsm.nodes["PhaseRouter"].tools == ("lookup_order",)
    assert "PhaseRouter.Schema" in fsm.router_ids
    assert "PhaseRouter" not in fsm.router_ids
    assert fsm.entry_id == "PhaseRouter"
