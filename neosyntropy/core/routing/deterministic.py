"""Deterministic routing.

Developer API: :class:`DeterministicRouter` declarations (rules → nodes/routers).
Runtime offline adapter: :class:`PreferredPathRouter`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..edge import Edge, edge_deterministic
from .declarations import RouteTarget, _normalize_input_schema, _target_id
from .preferred import PreferredPathRouter

# Backward-compatible name for the runtime preferred-path adapter.
# Prefer PreferredPathRouter in new framework code.
GraphDeterministicRouter = PreferredPathRouter

@dataclass
class _RuleContext:
    """Minimal context exposed to deterministic rule predicates."""

    state: dict[str, Any]

def _wrap_guard(
    predicate: Callable[..., bool],
) -> Callable[[dict[str, Any]], bool]:
    """Adapt ``(ctx) -> bool`` or ``(state) -> bool`` to an edge guard."""

    def guard(state: dict[str, Any]) -> bool:
        try:
            return bool(predicate(_RuleContext(state)))
        except TypeError:
            return bool(predicate(state))

    return guard

@dataclass
class DeterministicRouter:
    """Hard business rules: first matching rule wins.

    Example::

        auth = DeterministicRouter(
            id="CheckAuth",
            input_schema=OpenInput,
            rules=[
                (lambda ctx: ctx.state.get("token_valid") is True, intent_router),
                (lambda ctx: ctx.state.get("token_valid") is False, login_node),
            ],
        )
    """

    id: str
    rules: Sequence[tuple[Callable[..., bool], RouteTarget]]
    description: str = ""
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    #: Owning group name when this router was authored inside a :class:`Group`.
    group: str | None = None
    # Resolved JSON Schema + optional Pydantic model (set in __post_init__).
    json_schema: dict[str, Any] | None = field(init=False, default=None, repr=False)
    input_model: type[BaseModel] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DeterministicRouter requires id")
        if not self.rules:
            raise ValueError(f"DeterministicRouter {self.id!r} requires rules")
        schema, model = _normalize_input_schema(
            self.input_schema, owner=f"DeterministicRouter {self.id!r}"
        )
        object.__setattr__(self, "json_schema", schema)
        object.__setattr__(self, "input_model", model)

    def compile(self) -> list[Edge]:
        edges: list[Edge] = []
        for index, (predicate, target) in enumerate(self.rules):
            target_id, kind = _target_id(target)
            if kind == "group":
                raise ValueError(
                    f"DeterministicRouter {self.id!r} rule {index} cannot target a "
                    f"group; use SemanticRouter for group routes"
                )
            edges.append(
                edge_deterministic(
                    self.id,
                    target_id,
                    guard=_wrap_guard(predicate),
                    description=f"{self.id} rule[{index}] -> {target_id}",
                )
            )
        return edges

__all__ = [
    "DeterministicRouter",
    "PreferredPathRouter",
    "GraphDeterministicRouter",
]
