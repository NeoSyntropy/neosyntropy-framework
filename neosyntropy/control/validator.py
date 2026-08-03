"""Strict deterministic validation of router-produced plans.

Ported from ``neosyntropy_backend_cli/core/fsm/validator.py``. This is the
hard gate between proposal and execution: a plan that fails here never runs.
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.graph import FSM
from ..core.models import Candidate, RoutingPlan, Topology


class PlanValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid routing plan: " + "; ".join(errors))


class PlanValidator:
    def validate(
        self,
        plan: RoutingPlan,
        candidates: list[Candidate],
        graph: FSM,
        context: RunContext,
    ) -> None:
        errors: list[str] = []
        if not 1 <= len(candidates) <= 10:
            errors.append("candidate count must be between 1 and 10")
        candidate_ids = [candidate.node_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append("candidate node ids must be unique")
        unknown = sorted(set(candidate_ids) - set(graph.nodes))
        if unknown:
            errors.append(f"candidates reference unknown graph nodes: {unknown}")
        for candidate in candidates:
            definition = graph.nodes.get(candidate.node_id)
            if definition is None:
                continue
            if candidate.is_fallback != definition.is_fallback:
                errors.append(
                    f"candidate {candidate.node_id!r} has inconsistent fallback status"
                )
            if candidate.prerequisites != definition.prerequisites:
                errors.append(
                    f"candidate {candidate.node_id!r} has inconsistent prerequisites"
                )
        fallback_indices = [
            index for index, candidate in enumerate(candidates) if candidate.is_fallback
        ]
        if len(fallback_indices) != 1:
            errors.append("candidates must contain exactly one dedicated fallback")

        flat = [index for step in plan.execution_plan for index in step]
        if len(flat) != len(set(flat)):
            errors.append("a candidate may appear only once in an execution plan")
        invalid = sorted({index for index in flat if index < 0 or index >= len(candidates)})
        if invalid:
            errors.append(f"candidate indices out of bounds: {invalid}")
            raise PlanValidationError(errors)

        self._validate_topology(plan, candidates, fallback_indices, errors)
        self._validate_prerequisites(plan, candidates, context, errors)
        self._validate_transitions(plan, candidates, graph, context, errors)
        if errors:
            raise PlanValidationError(errors)

    @staticmethod
    def _validate_topology(
        plan: RoutingPlan,
        candidates: list[Candidate],
        fallback_indices: list[int],
        errors: list[str],
    ) -> None:
        steps = plan.execution_plan
        if plan.topology == Topology.PARALLEL:
            if len(steps) != 1 or len(steps[0]) < 2:
                errors.append("parallel topology requires one step with at least two nodes")
        elif plan.topology == Topology.SEQUENTIAL:
            if any(len(step) != 1 for step in steps):
                errors.append("sequential topology requires singleton steps")
        elif plan.topology == Topology.HYBRID:
            if len(steps) < 2 or not any(len(step) > 1 for step in steps):
                errors.append("hybrid topology requires multiple steps and a parallel step")
        elif plan.topology == Topology.FALLBACK:
            if len(steps) != 1 or len(steps[0]) != 1:
                errors.append("fallback topology requires exactly one singleton step")

        selected_fallback = [
            index for step in steps for index in step if candidates[index].is_fallback
        ]
        if plan.topology == Topology.FALLBACK:
            if selected_fallback != fallback_indices:
                errors.append("fallback plan must select only the dedicated fallback")
        elif selected_fallback:
            errors.append("fallback cannot be mixed with actionable nodes")

    @staticmethod
    def _validate_prerequisites(
        plan: RoutingPlan,
        candidates: list[Candidate],
        context: RunContext,
        errors: list[str],
    ) -> None:
        satisfied = {
            item.node_id for item in context.prior_executions if item.status == "succeeded"
        }
        for step in plan.execution_plan:
            for index in step:
                candidate = candidates[index]
                missing = set(candidate.prerequisites) - satisfied
                if missing:
                    errors.append(
                        f"{candidate.node_id} has unsatisfied prerequisites: "
                        f"{sorted(missing)}"
                    )
            satisfied.update(candidates[index].node_id for index in step)

    @staticmethod
    def _validate_transitions(
        plan: RoutingPlan,
        candidates: list[Candidate],
        graph: FSM,
        context: RunContext,
        errors: list[str],
    ) -> None:
        if plan.topology == Topology.FALLBACK:
            # The dedicated fallback is a safe stop, reachable from any state;
            # it does not move the workflow, so the transition table is not
            # consulted.
            return
        table = graph.transition_table()
        sources = {context.current_state}
        for step in plan.execution_plan:
            targets = {candidates[index].node_id for index in step}
            for target in targets:
                if not any(table.permits(source, target) for source in sources):
                    errors.append(
                        f"no legal transition to {target!r} from {sorted(sources)}"
                    )
            sources = targets
