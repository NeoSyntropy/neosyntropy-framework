"""Preferred-path runtime router: follows compiled FSM edges offline.

This is the control-loop adapter (not the developer authoring API).
Authoring uses :class:`~neosyntropy.routing.DeterministicRouter` /
:class:`~neosyntropy.routing.SemanticRouter` declarations that compile to edges.
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.graph import FSM
from ..core.models import Candidate, RoutingPlan, Topology
from .base import RouterError


class PreferredPathRouter:
    """Deterministic preferred-path over a compiled FSM (offline / no backend)."""

    def __init__(self, graph: FSM):
        self.graph = graph

    async def route(
        self, context: RunContext, candidates: list[Candidate]
    ) -> RoutingPlan:
        index_by_node = {candidate.node_id: i for i, candidate in enumerate(candidates)}

        matching = [
            edge
            for edge in self.graph.matching_deterministic(
                context.current_state, context.state
            )
            if edge.target in index_by_node
            and not candidates[index_by_node[edge.target]].is_fallback
        ]
        if len(matching) == 1:
            best = matching[0]
            return RoutingPlan(
                reasoning=(
                    f"Deterministic edge {context.current_state!r} -> "
                    f"{best.target!r}."
                ),
                topology=Topology.SEQUENTIAL,
                execution_plan=[[index_by_node[best.target]]],
            )

        scoped = self.graph.semantic_candidate_ids(context.current_state)
        if scoped is not None:
            actionable = [
                node_id
                for node_id in sorted(scoped)
                if node_id in index_by_node
                and not candidates[index_by_node[node_id]].is_fallback
                and self.graph.guard_allows(
                    context.current_state, node_id, context.state
                )
            ]
            if len(actionable) == 1:
                target = actionable[0]
                return RoutingPlan(
                    reasoning=(
                        f"Single semantic target {context.current_state!r} -> "
                        f"{target!r}."
                    ),
                    topology=Topology.SEQUENTIAL,
                    execution_plan=[[index_by_node[target]]],
                )

        fallback_id = self.graph.fallback_target(context.current_state)
        if fallback_id not in index_by_node:
            fallback_indices = [
                i for i, candidate in enumerate(candidates) if candidate.is_fallback
            ]
            if len(fallback_indices) != 1:
                raise RouterError(
                    "candidates must contain exactly one dedicated fallback"
                )
            fallback_index = fallback_indices[0]
        else:
            fallback_index = index_by_node[fallback_id]
        return RoutingPlan(
            reasoning=(
                f"No deterministic or unique semantic route from "
                f"{context.current_state!r}; using fallback edge to "
                f"{candidates[fallback_index].node_id!r}."
            ),
            topology=Topology.FALLBACK,
            execution_plan=[[fallback_index]],
        )
