"""Edge: one permitted movement between states.

Three edge kinds drive control:

- ``deterministic`` — a guard (or always-true) decides; when exactly one
  matches, the transition commits without calling the router.
- ``semantic`` — scopes hybrid candidate search to a node or group; the
  backend router then chooses among those candidates.
- ``fallback`` — used only when neither deterministic nor semantic yields a
  route; points at the dedicated fallback (or another safe stop).

Node results may *propose* a next state; the transition table *permits* it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

EdgeKind = Literal["deterministic", "semantic", "fallback"]
EdgeTargetKind = Literal["node", "group"]


class Edge(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    source: str = Field(validation_alias=AliasChoices("source", "from"))
    target: str = Field(validation_alias=AliasChoices("target", "to"))
    kind: EdgeKind = "semantic"
    target_kind: EdgeTargetKind = "node"
    description: str = ""
    guard: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, exclude=True, repr=False
    )

    @model_validator(mode="after")
    def validate_kind_constraints(self) -> Edge:
        if self.target_kind == "group" and self.kind != "semantic":
            raise ValueError(
                f"{self.kind} edges cannot target a group "
                f"(only semantic edges may); got {self.source!r} -> {self.target!r}"
            )
        if self.kind == "deterministic" and self.target_kind != "node":
            raise ValueError("deterministic edges must target a node")
        if self.kind == "fallback" and self.target_kind != "node":
            raise ValueError("fallback edges must target a node")
        return self

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


def edge_deterministic(
    source: str,
    target: str,
    *,
    guard: Callable[[dict[str, Any]], bool] | None = None,
    description: str = "",
) -> Edge:
    """Permitted auto-transition when ``guard`` (or always) passes."""
    return Edge(
        source=source,
        target=target,
        kind="deterministic",
        target_kind="node",
        description=description,
        guard=guard,
    )


def edge_semantic(
    source: str,
    target: str,
    *,
    target_kind: EdgeTargetKind = "node",
    description: str = "",
) -> Edge:
    """Scopes the semantic router to a node or group; the router chooses."""
    return Edge(
        source=source,
        target=target,
        kind="semantic",
        target_kind=target_kind,
        description=description,
    )


def edge_fallback(
    source: str,
    target: str,
    *,
    description: str = "",
) -> Edge:
    """Safe-stop edge when deterministic and semantic routing both miss."""
    return Edge(
        source=source,
        target=target,
        kind="fallback",
        target_kind="node",
        description=description,
    )


@dataclass(frozen=True)
class TransitionTable:
    """Efficient immutable view of allowed transitions (guards excluded)."""

    allowed: frozenset[tuple[str, str]]
    group_edges: frozenset[tuple[str, str]]
    node_groups: Mapping[str, str | None]
    permissive: bool = False

    @classmethod
    def from_edges(
        cls,
        edges: list[Edge],
        *,
        node_groups: Mapping[str, str | None] | None = None,
        allow_unlisted_transitions: bool = False,
    ) -> TransitionTable:
        groups = node_groups or {}
        allowed = frozenset(
            (edge.source, edge.target)
            for edge in edges
            if edge.target_kind == "node"
        )
        group_edges = frozenset(
            (edge.source, edge.target)
            for edge in edges
            if edge.target_kind == "group"
        )
        return cls(
            allowed,
            group_edges,
            dict(groups),
            allow_unlisted_transitions,
        )

    def permits(self, source: str, target: str) -> bool:
        if self.permissive or (source, target) in self.allowed:
            return True
        group = self.node_groups.get(target)
        return group is not None and (source, group) in self.group_edges
