"""Candidate selection: search is not permission.

Selection always runs — even with empty transitions — and finds relevant
nodes. Legality is decided later by the plan validator, which fail-closes
unless the edge is listed (or unlisted transitions are explicitly allowed).
"""
from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from ..core.context import RunContext
from ..core.graph import Graph
from ..core.models import Candidate

MAX_CANDIDATES = 10
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class CandidateSelector(Protocol):
    def select(
        self, context: RunContext, graph: Graph
    ) -> list[Candidate] | Awaitable[list[Candidate]]: ...


class LexicalCandidateSelector:
    """Simple deterministic in-memory selector.

    Scores nodes by token overlap between the intent and the node's
    name/description/group, keeps the best nine actionable nodes, and always
    appends the dedicated fallback last (the reserved slot in the trained
    router contract). Swap in a vector-based selector via the
    :class:`CandidateSelector` protocol without touching the control flow.
    """

    def __init__(self, max_candidates: int = MAX_CANDIDATES):
        if not 2 <= max_candidates <= MAX_CANDIDATES:
            raise ValueError("max_candidates must be between 2 and 10")
        self.max_candidates = max_candidates

    def select(self, context: RunContext, graph: Graph) -> list[Candidate]:
        intent_tokens = set(_TOKEN_RE.findall(context.intent.lower()))
        actionable: list[Candidate] = []
        for position, item in enumerate(graph.nodes.values()):
            if item.is_fallback:
                continue
            text = " ".join((item.name, item.description, item.group or "")).lower()
            node_tokens = set(_TOKEN_RE.findall(text))
            overlap = len(intent_tokens & node_tokens)
            score = overlap / len(intent_tokens) if intent_tokens else 0.0
            metadata: dict = {"position": position}
            if item.group:
                metadata["group"] = item.group
            actionable.append(
                Candidate(
                    node_id=item.id,
                    name=item.name,
                    description=item.description,
                    score=score,
                    prerequisites=item.prerequisites,
                    is_fallback=False,
                    metadata=metadata,
                )
            )
        actionable.sort(key=lambda c: (-c.score, c.metadata["position"]))
        selected = actionable[: self.max_candidates - 1]

        fallback = graph.fallback_node
        selected.append(
            Candidate(
                node_id=fallback.id,
                name=fallback.name,
                description=fallback.description,
                score=0.0,
                prerequisites=fallback.prerequisites,
                is_fallback=True,
            )
        )
        return selected
