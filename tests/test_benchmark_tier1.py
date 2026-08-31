"""Compile-only checks for the Tier 1 neo-code FSM and NS vs LangGraph protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TIER1 = Path(__file__).resolve().parents[1] / "Benchmarks" / "tier1"
sys.path.insert(0, str(TIER1))

from fsm import PHASE_ROUTES, build_fsm  # noqa: E402
from protocol import (  # noqa: E402
    Case,
    Trace,
    format_table,
    landing_from_path,
    load_cases,
    map_phase_label,
    neosyntropy_usd,
    path_from_committed,
    score_trace,
    summarize,
    token_usd,
)


def test_tier1_fsm_compiles_phase_router_and_analyst_path() -> None:
    graph = build_fsm()
    assert graph.entry_id == "PhaseRouter"
    router = graph.routers["PhaseRouter"]
    assert set(router.routes) == set(PHASE_ROUTES)
    assert {PHASE_ROUTES[label] for label in router.routes} == {
        "AnalystPhase",
        "PlanStub",
        "SolutioningStub",
        "ImplementationStub",
        "CoreStub",
        "HelpStub",
    }

    deterministic = {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.kind == "deterministic"
    }
    assert ("AnalystPhase", "FinalizeNode") in deterministic
    assert ("FinalizeNode", "End") in deterministic

    semantic_targets = {
        edge.target
        for edge in graph.edges
        if edge.source == "PhaseRouter" and edge.kind == "semantic"
    }
    assert semantic_targets == set(PHASE_ROUTES.values())
    assert "OutOfScope" in graph.nodes
    assert graph.nodes["OutOfScope"].is_fallback is True
    assert graph.nodes["AnalystPhase"].tools == (
        "append_memlog",
        "read_workspace_file",
    )


def test_gold_cases_cover_every_phase_and_fallback() -> None:
    from protocol import PHASE_ROUTES as PROTOCOL_ROUTES

    assert PROTOCOL_ROUTES == PHASE_ROUTES
    cases = load_cases()
    landings = {case.expected_route for case in cases}
    assert landings == set(PHASE_ROUTES.values()) | {"OutOfScope"}
    assert all(case.user_request and case.expected_path[0] == "PhaseRouter" for case in cases)
    analysis = [case for case in cases if case.expected_route == "AnalystPhase"]
    assert analysis
    assert all(
        "append_memlog" in case.required_tools and "read_workspace_file" in case.required_tools
        for case in analysis
    )


def test_path_and_landing_from_committed_hops() -> None:
    path = path_from_committed(
        [
            "PhaseRouter->AnalystPhase",
            "AnalystPhase->FinalizeNode",
            "FinalizeNode->End",
        ]
    )
    assert path == ["PhaseRouter", "AnalystPhase", "FinalizeNode", "End"]
    assert landing_from_path(path) == "AnalystPhase"


def test_map_phase_label_counts_illegal_hops() -> None:
    assert map_phase_label("plan") == ("PlanStub", 0)
    assert map_phase_label("out_of_scope") == ("OutOfScope", 0)
    assert map_phase_label("pirate_mode") == ("OutOfScope", 1)


def test_cost_models_use_transition_vs_token_units() -> None:
    stub = ["PhaseRouter->PlanStub", "PlanStub->End"]
    analysis = [
        "PhaseRouter->AnalystPhase",
        "AnalystPhase->FinalizeNode",
        "FinalizeNode->End",
    ]
    assert neosyntropy_usd(stub) == 0.015
    assert neosyntropy_usd(analysis) == 0.025
    assert token_usd(1_000_000, 0, model="gpt-4.1-mini") == 0.40
    assert token_usd(0, 1_000_000, model="gpt-4.1-mini") == 1.60


def test_task_accuracy_requires_route_path_schema_and_tools() -> None:
    case = Case(
        id="a01",
        user_request="Analyze constraints.",
        expected_route="AnalystPhase",
        expected_path=("PhaseRouter", "AnalystPhase", "FinalizeNode", "End"),
        required_tools=("append_memlog", "read_workspace_file"),
        expected_output={"status": "completed"},
    )
    ok = Trace(
        case_id="a01",
        system="neosyntropy",
        landing="AnalystPhase",
        path=["PhaseRouter", "AnalystPhase", "FinalizeNode", "End"],
        output={"status": "completed", "analysis_report": "constraints listed"},
        tools_ok=["read_workspace_file", "append_memlog"],
        latency_ms=120.0,
        transitions=3,
        usd=0.025,
    )
    scored = score_trace(case, ok)
    assert scored.passed is True

    wrong_route = Trace(
        **{**ok.to_dict(), "landing": "PlanStub", "path": ["PhaseRouter", "PlanStub", "End"]}
    )
    assert score_trace(case, wrong_route).passed is False

    missing_tools = Trace(**{**ok.to_dict(), "tools_ok": ["append_memlog"]})
    assert score_trace(case, missing_tools).tools_ok is False
    assert score_trace(case, missing_tools).passed is False


def test_comparison_table_summarizes_accuracy_latency_and_money() -> None:
    ns_case = Case(
        id="p01",
        user_request="Write a PRD.",
        expected_route="PlanStub",
        expected_path=("PhaseRouter", "PlanStub", "End"),
        expected_output={"phase": "plan"},
    )
    ns = score_trace(
        ns_case,
        Trace(
            case_id="p01",
            system="neosyntropy",
            landing="PlanStub",
            path=["PhaseRouter", "PlanStub", "End"],
            output={"phase": "plan"},
            latency_ms=80,
            transitions=2,
            usd=0.015,
        ),
    )
    lg = score_trace(
        ns_case,
        Trace(
            case_id="p01",
            system="langgraph",
            landing="PlanStub",
            path=["PhaseRouter", "PlanStub", "End"],
            output={"phase": "plan"},
            latency_ms=900,
            tokens_in=1200,
            tokens_out=40,
            llm_calls=2,
            usd=0.0005,
            illegal_hops=0,
        ),
    )
    table = format_table([summarize("neosyntropy", [ns]), summarize("langgraph", [lg])])
    assert "neosyntropy" in table
    assert "langgraph" in table
    assert "100.0%" in table
    ns_summary = summarize("neosyntropy", [ns])
    lg_summary = summarize("langgraph", [lg])
    assert ns_summary.usd == 0.015
    assert lg_summary.mean_ms == 900
    assert ns_summary.p50_ms < lg_summary.p50_ms


def test_langgraph_scripted_plan_path() -> None:
    pytest.importorskip("langgraph")
    from langgraph_app import LangGraphHarness

    class _Msg:
        def __init__(self, content: str = "") -> None:
            self.content = content
            self.tool_calls: list[object] = []
            self.usage_metadata = {"input_tokens": 20, "output_tokens": 8}

    class ScriptedLLM:
        def __init__(self) -> None:
            self._schema: object | None = None

        def with_structured_output(self, schema: object, **_kwargs: object) -> ScriptedLLM:
            clone = ScriptedLLM()
            clone._schema = schema
            return clone

        def bind_tools(self, _tools: object) -> ScriptedLLM:
            return self

        def invoke(self, _messages: object) -> object:
            schema = self._schema
            name = getattr(schema, "__name__", "")
            if name == "RouteDecision" and schema is not None:
                return schema(phase="plan")  # type: ignore[misc]
            if name == "PhaseStubOutput" and schema is not None:
                return schema(phase="plan")  # type: ignore[misc]
            if name == "FinalizeOutput" and schema is not None:
                return schema(status="completed", analysis_report="n/a")  # type: ignore[misc]
            return _Msg("ok")

    result = LangGraphHarness(ScriptedLLM(), model_name="gpt-4.1-mini").invoke(
        {"user_request": "Write a PRD for the CI secret scanner."}
    )
    assert result["landing"] == "PlanStub"
    assert result["path"] == ["PhaseRouter", "PlanStub", "End"]
    assert result["output"]["phase"] == "plan"
    assert result["llm_calls"] >= 1
    assert result["usd"] > 0
