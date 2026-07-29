"""Deterministic preferred-path router: no model, no guessing.

Walks the graph's outgoing edges from the current state using the verbatim
label priority table (`load < first < next < inferred-next < complete <
return < route < conditional`). When no legal, guard-allowed candidate
exists, it proposes the dedicated fallback — a safe stop, never improvisation.
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.graph import Graph
from ..core.models import Candidate, RoutingPlan, Topology
from .base import RouterError


class DeterministicRouter:
    def __init__(self, graph: Graph):
        self.graph = graph

    async def route(
        self, context: RunContext, candidates: list[Candidate]
    ) -> RoutingPlan:
        index_by_node = {candidate.node_id: i for i, candidate in enumerate(candidates)}

        choices = [
            edge
            for edge in self.graph.outgoing(context.current_state)
            if edge.target in index_by_node
            and not candidates[index_by_node[edge.target]].is_fallback
            and edge.guard_allows(context.state)
        ]
        if choices:
            best = min(choices, key=lambda edge: (edge.priority, edge.target))
            return RoutingPlan(
                reasoning=(
                    f"Preferred path: edge {context.current_state!r} -> "
                    f"{best.target!r} (label={best.label!r})."
                ),
                topology=Topology.SEQUENTIAL,
                execution_plan=[[index_by_node[best.target]]],
            )

        fallback_indices = [
            i for i, candidate in enumerate(candidates) if candidate.is_fallback
        ]
        if len(fallback_indices) != 1:
            raise RouterError("candidates must contain exactly one dedicated fallback")
        return RoutingPlan(
            reasoning=(
                f"No legal guard-allowed transition from {context.current_state!r}; "
                "routing to the dedicated fallback."
            ),
            topology=Topology.FALLBACK,
            execution_plan=[[fallback_indices[0]]],
        )
