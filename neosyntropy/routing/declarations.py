"""Developer-facing routers: declare routing instead of hand-writing edges.

``DeterministicRouter`` and ``SemanticRouter`` are authored units that compile
into FSM edges. Targets may be nodes, groups, or other routers.

When a router is the FSM ``entry``, it must declare ``input_schema`` — that
contract becomes the workflow entry gate.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..core.edge import Edge, edge_deterministic, edge_fallback, edge_semantic
from ..core.group import Group
from ..core.node import CombineNode, Node
from ..core.schemas import input_model_schema

# Node | CombineNode | Group | DeterministicRouter | SemanticRouter | str
RouteTarget = Any


@dataclass
class _RuleContext:
    """Minimal context exposed to deterministic rule predicates."""

    state: dict[str, Any]


def _normalize_input_schema(
    source: type[BaseModel] | dict[str, Any] | None,
    *,
    owner: str,
) -> tuple[dict[str, Any] | None, type[BaseModel] | None]:
    """Normalize an optional router input contract to a closed JSON Schema."""
    if source is None:
        return None, None
    if isinstance(source, type) and issubclass(source, BaseModel):
        return input_model_schema(source), source
    if isinstance(source, dict) and source:
        return dict(source), None
    raise ValueError(
        f"{owner} input_schema must be a pydantic BaseModel class or a "
        "non-empty JSON Schema object"
    )


def _target_id(target: RouteTarget) -> tuple[str, str]:
    """Return ``(id, kind)`` where kind is ``node`` | ``group`` | ``router``."""
    if isinstance(target, str):
        return target, "node"
    if isinstance(target, Group):
        return target.name, "group"
    if isinstance(target, (DeterministicRouter, SemanticRouter)):
        return target.id, "router"
    if isinstance(target, CombineNode):
        return target.id, "node"
    if isinstance(target, Node):
        return target.id, "node"
    raise TypeError(f"unsupported route target: {type(target)!r}")


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


@dataclass
class SemanticRouter:
    """SLM / intent routing over labeled targets (nodes or groups).

    Example::

        intent = SemanticRouter(
            id="CustomerIntent",
            input_schema=OpenInput,
            routes={
                "wants_to_pay": billing_group,
                "needs_support": support_group,
            },
            fallback_node=general_chat,
        )
    """

    id: str
    routes: Mapping[str, RouteTarget]
    fallback_node: Node | CombineNode | str | None = None
    description: str = ""
    category: str = "general"
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    #: Owning group name when this router was authored inside a :class:`Group`.
    group: str | None = None
    json_schema: dict[str, Any] | None = field(init=False, default=None, repr=False)
    input_model: type[BaseModel] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SemanticRouter requires id")
        if not self.routes:
            raise ValueError(f"SemanticRouter {self.id!r} requires routes")
        schema, model = _normalize_input_schema(
            self.input_schema, owner=f"SemanticRouter {self.id!r}"
        )
        object.__setattr__(self, "json_schema", schema)
        object.__setattr__(self, "input_model", model)

    def compile(self) -> list[Edge]:
        edges: list[Edge] = []
        for label, target in self.routes.items():
            target_id, kind = _target_id(target)
            if kind == "router":
                # Enter another router state after the semantic choice — rare;
                # treat as a node-kind edge to the router id.
                edges.append(
                    edge_semantic(
                        self.id,
                        target_id,
                        target_kind="node",
                        description=f"{self.id}:{label} -> router:{target_id}",
                    )
                )
            elif kind == "group":
                edges.append(
                    edge_semantic(
                        self.id,
                        target_id,
                        target_kind="group",
                        description=f"{self.id}:{label} -> group:{target_id}",
                    )
                )
            else:
                edges.append(
                    edge_semantic(
                        self.id,
                        target_id,
                        target_kind="node",
                        description=f"{self.id}:{label} -> {target_id}",
                    )
                )
        if self.fallback_node is not None:
            fb_id, fb_kind = _target_id(self.fallback_node)
            if fb_kind == "group":
                raise ValueError(
                    f"SemanticRouter {self.id!r} fallback_node cannot be a group"
                )
            edges.append(
                edge_fallback(
                    self.id,
                    fb_id,
                    description=f"{self.id} fallback -> {fb_id}",
                )
            )
        return edges


RouterDecl = DeterministicRouter | SemanticRouter


def collect_router_ids(routers: Sequence[RouterDecl]) -> set[str]:
    return {item.id for item in routers}


def compile_routers(routers: Sequence[RouterDecl]) -> list[Edge]:
    edges: list[Edge] = []
    seen: set[str] = set()
    for item in routers:
        if item.id in seen:
            raise ValueError(f"duplicate router id {item.id!r}")
        seen.add(item.id)
        edges.extend(item.compile())
    return edges


def collect_nested_routers(root: RouterDecl) -> list[RouterDecl]:
    """Walk rule/route targets and return root + nested router declarations."""
    found: dict[str, RouterDecl] = {}

    def visit(item: RouterDecl) -> None:
        if item.id in found:
            return
        found[item.id] = item
        targets: list[RouteTarget] = []
        if isinstance(item, DeterministicRouter):
            targets.extend(target for _, target in item.rules)
        else:
            targets.extend(item.routes.values())
            if item.fallback_node is not None:
                targets.append(item.fallback_node)
        for target in targets:
            if isinstance(target, (DeterministicRouter, SemanticRouter)):
                visit(target)

    visit(root)
    return list(found.values())
