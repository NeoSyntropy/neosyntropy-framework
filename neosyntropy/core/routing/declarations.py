"""Developer-facing routers: declare routing instead of hand-writing edges.

``DeterministicRouter`` and ``SemanticRouter`` are authored units that compile
into FSM edges. Targets may be nodes, groups, or other routers.

When a router is the FSM ``entry``, it must declare ``input_schema`` — that
contract becomes the workflow entry gate.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..edge import Edge
from ..group import Group
from ..node import CombineNode, Node
from ..schemas import input_model_schema

# Node | CombineNode | Group | DeterministicRouter | SemanticRouter | str
RouteTarget = Any


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
    if type(target).__name__ in ("DeterministicRouter", "SemanticRouter"):
        return target.id, "router"
    if isinstance(target, CombineNode):
        return target.id, "node"
    if isinstance(target, Node):
        return target.id, "node"
    raise TypeError(f"unsupported route target: {type(target)!r}")


import typing
if typing.TYPE_CHECKING:
    from ..protocols import Compilable
    RouterDecl = Compilable
else:
    RouterDecl = Any


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
        if type(item).__name__ == "DeterministicRouter":
            targets.extend(target for _, target in item.rules)
        else:
            targets.extend(item.routes.values())
            if item.fallback_node is not None:
                targets.append(item.fallback_node)
        for target in targets:
            if type(target).__name__ in ("DeterministicRouter", "SemanticRouter"):
                visit(target)

    visit(root)
    return list(found.values())
