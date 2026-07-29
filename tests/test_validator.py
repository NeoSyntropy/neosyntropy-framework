from __future__ import annotations

import pytest

from neosyntropy import (
    Candidate,
    PlanValidationError,
    PlanValidator,
    RoutingPlan,
    RunContext,
    Topology,
)

from .conftest import build_graph


def make_context(current_state: str = "Start", prior=()) -> RunContext:
    return RunContext(
        request_id="req-1",
        intent="refund my order",
        current_state=current_state,
        prior_executions=list(prior),
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


@pytest.fixture
def graph():
    return build_graph()


@pytest.fixture
def candidates(graph):
    return make_candidates(graph)


def index_of(candidates, node_id: str) -> int:
    return next(i for i, c in enumerate(candidates) if c.node_id == node_id)


def test_legal_sequential_plan_passes(graph, candidates):
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[[index_of(candidates, "VerifyIdentity")]],
    )
    PlanValidator().validate(plan, candidates, graph, make_context())


def test_illegal_transition_is_rejected(graph, candidates):
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[[index_of(candidates, "IssueRefund")]],
    )
    prior = [
        {"node_id": "VerifyIdentity", "status": "succeeded"},
        {"node_id": "CalculateRefund", "status": "succeeded"},
    ]
    with pytest.raises(PlanValidationError, match="no legal transition"):
        PlanValidator().validate(plan, candidates, graph, make_context(prior=prior))


def test_unsatisfied_prerequisites_are_rejected(graph, candidates):
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[
            [index_of(candidates, "VerifyIdentity")],
            [index_of(candidates, "CalculateRefund")],
            [index_of(candidates, "IssueRefund")],
        ],
    )
    # Prerequisites satisfied by earlier plan steps: passes.
    PlanValidator().validate(plan, candidates, graph, make_context())

    lone = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[[index_of(candidates, "CalculateRefund")]],
    )
    with pytest.raises(PlanValidationError, match="unsatisfied prerequisites"):
        PlanValidator().validate(lone, candidates, graph, make_context("VerifyIdentity"))


def test_fallback_cannot_mix_with_actionable_nodes(graph, candidates):
    plan = RoutingPlan(
        topology=Topology.PARALLEL,
        execution_plan=[
            [index_of(candidates, "VerifyIdentity"), index_of(candidates, "OutOfScope")]
        ],
    )
    with pytest.raises(PlanValidationError, match="fallback cannot be mixed"):
        PlanValidator().validate(plan, candidates, graph, make_context())


def test_fallback_plan_selects_only_the_dedicated_fallback(graph, candidates):
    plan = RoutingPlan(
        topology=Topology.FALLBACK,
        execution_plan=[[index_of(candidates, "OutOfScope")]],
    )
    PlanValidator().validate(plan, candidates, graph, make_context())


def test_out_of_bounds_indices_are_rejected(graph, candidates):
    plan = RoutingPlan(topology=Topology.SEQUENTIAL, execution_plan=[[42]])
    with pytest.raises(PlanValidationError, match="out of bounds"):
        PlanValidator().validate(plan, candidates, graph, make_context())


def test_duplicate_candidate_use_is_rejected(graph, candidates):
    index = index_of(candidates, "VerifyIdentity")
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL, execution_plan=[[index], [index]]
    )
    with pytest.raises(PlanValidationError, match="only once"):
        PlanValidator().validate(plan, candidates, graph, make_context())


def test_topology_shape_rules(graph, candidates):
    verify = index_of(candidates, "VerifyIdentity")
    calculate = index_of(candidates, "CalculateRefund")
    # Sequential topology with a parallel step is rejected.
    bad_sequential = RoutingPlan(
        topology=Topology.SEQUENTIAL, execution_plan=[[verify, calculate]]
    )
    with pytest.raises(PlanValidationError, match="singleton steps"):
        PlanValidator().validate(bad_sequential, candidates, graph, make_context())
    # Parallel topology requires at least two nodes in its single step.
    bad_parallel = RoutingPlan(topology=Topology.PARALLEL, execution_plan=[[verify]])
    with pytest.raises(PlanValidationError, match="at least two nodes"):
        PlanValidator().validate(bad_parallel, candidates, graph, make_context())


def test_candidate_metadata_must_match_graph(graph, candidates):
    tampered = [
        candidate.model_copy(update={"prerequisites": ()})
        if candidate.node_id == "IssueRefund"
        else candidate
        for candidate in candidates
    ]
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[[index_of(tampered, "VerifyIdentity")]],
    )
    with pytest.raises(PlanValidationError, match="inconsistent prerequisites"):
        PlanValidator().validate(plan, tampered, graph, make_context())


def test_unlisted_transitions_allowed_when_explicitly_permissive(candidates):
    permissive = build_graph(allow_unlisted_transitions=True)
    candidates = make_candidates(permissive)
    plan = RoutingPlan(
        topology=Topology.SEQUENTIAL,
        execution_plan=[[index_of(candidates, "VerifyIdentity")]],
    )
    PlanValidator().validate(
        plan, candidates, permissive, make_context("SomewhereElse")
    )
