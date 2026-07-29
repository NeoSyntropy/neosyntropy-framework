"""Edge: one permitted movement between states.

Node results may *propose* a next state; the transition table *permits* it.
The label priority table is ported verbatim from the backend's deterministic
preferred-path runner and drives the deterministic router.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Ported verbatim from neosyntropy_backend_cli (preferred-path edge priority).
EDGE_LABEL_PRIORITY: dict[str, int] = {
    "load": 0,
    "first": 1,
    "next": 2,
    "inferred-next": 3,
    "complete": 4,
    "return": 5,
    "route": 6,
    "conditional": 7,
}
DEFAULT_EDGE_PRIORITY = 99


class Edge(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    source: str = Field(validation_alias=AliasChoices("source", "from"))
    target: str = Field(validation_alias=AliasChoices("target", "to"))
    label: str = ""
    description: str = ""
    guard: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, exclude=True, repr=False
    )

    @property
    def priority(self) -> int:
        return EDGE_LABEL_PRIORITY.get(self.label, DEFAULT_EDGE_PRIORITY)

    def guard_allows(self, state: dict[str, Any]) -> bool:
        """Evaluate this edge's guard fail-closed.

        Missing guard allows; a guard returning falsy denies; a guard that
        raises denies (never fail-open on errors).
        """
        if self.guard is None:
            return True
        try:
            return bool(self.guard(state))
        except Exception:
            return False


@dataclass(frozen=True)
class TransitionTable:
    """Efficient immutable view of allowed transitions (guards excluded)."""

    allowed: frozenset[tuple[str, str]]
    permissive: bool = False

    @classmethod
    def from_edges(
        cls, edges: list[Edge], *, allow_unlisted_transitions: bool = False
    ) -> TransitionTable:
        return cls(
            frozenset((edge.source, edge.target) for edge in edges),
            allow_unlisted_transitions,
        )

    def permits(self, source: str, target: str) -> bool:
        return self.permissive or (source, target) in self.allowed
