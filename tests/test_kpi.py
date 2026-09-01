"""Construction and behaviour tests for KPI nodes at all three levels."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from neosyntropy.core.node import (
    COMBINE_SCHEMA_SUFFIX,
    CombineNode,
    KpiResult,
    Node,
    SemanticKpiNode,
    functional_kpi_node,
)
from neosyntropy.core.kpi import (
    SemanticGroupPathKpi,
    SemanticFSMPathKpi,
    functional_fsm_path_kpi,
    functional_group_path_kpi,
)
from neosyntropy.core.group import Group
from neosyntropy.core.node.context import NodeContext


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


# ---------------------------------------------------------------------------
# KpiResult contract
# ---------------------------------------------------------------------------

def test_kpi_result_has_no_valid_field() -> None:
    kpi = KpiResult(name="completeness", score=0.8)
    assert not hasattr(kpi, "valid")


def test_kpi_result_defaults() -> None:
    kpi = KpiResult(name="foo", score=0.5)
    assert kpi.reason == ""


# ---------------------------------------------------------------------------
# SemanticKpiNode construction
# ---------------------------------------------------------------------------

def test_semantic_kpi_node_low_returns_schema_node() -> None:
    scorer = SemanticKpiNode(
        "answer_quality",
        input_schema=_In,
        prompt="Score the answer on a scale of 0–1. name='answer_quality'.",
    )
    assert isinstance(scorer, Node)
    assert not isinstance(scorer, CombineNode)
    assert scorer.kind == "schema"
    assert scorer.mode == "schema_extraction"
    assert scorer.tools == ()
    assert scorer.output_model is KpiResult


def test_semantic_kpi_node_high_returns_combine_node() -> None:
    scorer = SemanticKpiNode(
        "deep_quality",
        input_schema=_In,
        prompt="Reason through the quality, then score.",
        reasoning="high",
    )
    assert isinstance(scorer, CombineNode)
    nodes, links = scorer.expand()
    assert [n.id for n in nodes] == ["deep_quality", f"deep_quality{COMBINE_SCHEMA_SUFFIX}"]
    assert nodes[0].kind == "combine_part"
    assert nodes[0].mode == "reasoning"
    assert nodes[1].output_model is KpiResult
    assert links[0].source == "deep_quality"
    assert links[0].target == f"deep_quality{COMBINE_SCHEMA_SUFFIX}"


def test_semantic_kpi_node_tools_force_high() -> None:
    scorer = SemanticKpiNode(
        "tool_quality",
        input_schema=_In,
        prompt="Fetch order and score.",
        reasoning="low",
        tools=("fetch_order",),
    )
    assert isinstance(scorer, CombineNode)
    assert tuple(scorer.tools) == ("fetch_order",)


def test_semantic_kpi_node_rejects_unknown_reasoning() -> None:
    with pytest.raises(ValueError, match="reasoning must be"):
        SemanticKpiNode(
            "bad",
            input_schema=_In,
            prompt="score",
            reasoning="medium",  # type: ignore[arg-type]
        )


def test_semantic_kpi_node_requires_prompt() -> None:
    with pytest.raises(ValueError, match="requires a non-empty prompt"):
        SemanticKpiNode("no_prompt", input_schema=_In, prompt="")


# ---------------------------------------------------------------------------
# functional_kpi_node — construction
# ---------------------------------------------------------------------------

def test_functional_kpi_node_defaults() -> None:
    @functional_kpi_node()
    def my_kpi(ctx: NodeContext) -> float:
        return 0.75

    assert my_kpi.id == "my_kpi"
    assert my_kpi.kind == "handler"
    assert my_kpi.mode == "schema_extraction"
    assert my_kpi.group is None
    assert my_kpi.output_model is KpiResult


def test_functional_kpi_node_explicit_id() -> None:
    @functional_kpi_node(id="custom_id", output_key="my_score")
    def scorer(ctx: NodeContext) -> float:
        return 1.0

    assert scorer.id == "custom_id"


# ---------------------------------------------------------------------------
# functional_kpi_node — handler behaviour
# ---------------------------------------------------------------------------

def _make_ctx(state: dict) -> NodeContext:
    """Build a minimal NodeContext mock."""
    ctx = MagicMock(spec=NodeContext)
    ctx.state = state
    ctx.node = MagicMock()
    ctx.node.id = "test_kpi"

    def _result(output=None, *, state_updates=None, **_kw):
        from neosyntropy.core.models import NodeResult
        return NodeResult(
            node_id="test_kpi",
            output=output,
            state_updates=state_updates or {},
        )

    ctx.result.side_effect = _result
    return ctx


def test_functional_kpi_node_float_return_writes_state() -> None:
    @functional_kpi_node(id="float_kpi", output_key="my_score")
    def float_kpi(ctx: NodeContext) -> float:
        return 0.6

    ctx = _make_ctx({})
    result = asyncio.run(float_kpi.handler(ctx))
    assert result.state_updates["my_score"] == pytest.approx(0.6)
    assert result.state_updates["my_score_reason"] == ""
    assert result.state_updates["kpis"] == {"my_score": pytest.approx(0.6)}
    assert result.output == {"name": "my_score", "score": pytest.approx(0.6), "reason": ""}


def test_functional_kpi_node_kpi_result_return_writes_state() -> None:
    @functional_kpi_node(id="kpi_result_kpi", output_key="completeness")
    def kpi_result_kpi(ctx: NodeContext) -> KpiResult:
        return KpiResult(name="completeness", score=0.9, reason="all steps ran")

    ctx = _make_ctx({})
    result = asyncio.run(kpi_result_kpi.handler(ctx))
    assert result.state_updates["completeness"] == pytest.approx(0.9)
    assert result.state_updates["completeness_reason"] == "all steps ran"
    assert result.state_updates["kpis"] == {"completeness": pytest.approx(0.9)}


def test_functional_kpi_node_accumulates_kpis() -> None:
    @functional_kpi_node(id="kpi1", output_key="k1")
    def kpi1(ctx: NodeContext) -> KpiResult:
        return KpiResult(name="k1", score=0.5)

    @functional_kpi_node(id="kpi2", output_key="k2")
    def kpi2(ctx: NodeContext) -> KpiResult:
        return KpiResult(name="k2", score=0.8)

    ctx1 = _make_ctx({})
    result1 = asyncio.run(kpi1.handler(ctx1))

    ctx2 = _make_ctx({"kpis": result1.state_updates["kpis"]})
    result2 = asyncio.run(kpi2.handler(ctx2))

    assert result2.state_updates["kpis"] == {
        "k1": pytest.approx(0.5),
        "k2": pytest.approx(0.8),
    }


def test_functional_kpi_node_invalid_return_type() -> None:
    @functional_kpi_node(id="bad_kpi")
    def bad_kpi(ctx: NodeContext) -> str:  # type: ignore[return]
        return "not a number"

    ctx = _make_ctx({})
    with pytest.raises(TypeError, match="must return"):
        asyncio.run(bad_kpi.handler(ctx))


# ---------------------------------------------------------------------------
# Group-level factories
# ---------------------------------------------------------------------------

def test_semantic_group_path_kpi_registers_into_group() -> None:
    group = Group(name="billing")
    scorer = SemanticGroupPathKpi(
        "billing_quality",
        group=group,
        input_schema=_In,
        prompt="Score the billing flow. name='billing_quality'.",
    )
    assert "billing_quality" in group._nodes
    assert scorer.group == "billing"


def test_functional_group_path_kpi_registers_into_group() -> None:
    group = Group(name="triage")

    @functional_group_path_kpi(group=group, output_key="triage_quality")
    def triage_quality(ctx: NodeContext) -> float:
        return 0.7

    assert "triage_quality" in group._nodes
    assert triage_quality.group == "triage"


# ---------------------------------------------------------------------------
# FSM-level factories
# ---------------------------------------------------------------------------

def test_semantic_fsm_path_kpi_no_group() -> None:
    scorer = SemanticFSMPathKpi(
        "path_quality",
        input_schema=_In,
        prompt="Score the path. name='path_quality'.",
    )
    assert scorer.group is None


def test_functional_fsm_path_kpi_no_group() -> None:
    @functional_fsm_path_kpi(id="PathScore", input_schema=_In)
    def path_score(ctx: NodeContext) -> float:
        return 0.9

    assert path_score.group is None
