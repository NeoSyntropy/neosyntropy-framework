"""Semantic routing.

Developer API: :class:`SemanticRouter` declarations (labeled routes → nodes/groups).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..edge import Edge, edge_fallback, edge_semantic
from ..node import CombineNode, Node
from .declarations import RouteTarget, _normalize_input_schema, _target_id


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
    #: Backend inference provider id (``neosyntropy/base`` or a Vertex model).
    provider: str = "neosyntropy/base"
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
        from ..node import Node
        if isinstance(self.fallback_node, Node) and not getattr(self.fallback_node, "is_fallback", False):
            object.__setattr__(self, "fallback_node", self.fallback_node.model_copy(update={"is_fallback": True}))

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

__all__ = [
    "SemanticRouter",
]
