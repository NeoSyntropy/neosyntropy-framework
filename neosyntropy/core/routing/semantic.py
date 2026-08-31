"""Semantic routing.

Developer API: :class:`SemanticRouter` declarations (labeled routes → nodes/groups).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..edge import Edge, edge_deterministic, edge_fallback, edge_semantic
from ..node import COMBINE_SCHEMA_SUFFIX, CombineNode, Node, ReasoningLevel, ReasoningNode
from ..node._utils import _resolve_reasoning_level
from .declarations import RouteTarget, _normalize_input_schema, _target_id


@dataclass
class SemanticRouter:
    """SLM / intent routing over labeled targets (nodes or groups).

    ``reasoning="low"`` (default) is a single router state that picks a
    labeled route.  ``reasoning="high"`` prepends a reasoning node at
    ``{id}`` and moves the router to ``{id}.Schema`` (the CombineNode
    pattern).  Passing any *tools* automatically upgrades to ``high``.

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

        # High reasoning with tools:
        researched = SemanticRouter(
            id="ResearchedIntent",
            input_schema=OpenInput,
            routes={"refund": refund_node, "status": status_node},
            fallback_node=general_chat,
            tools=("lookup_order",),
            prompt="Look up the order, then choose refund or status.",
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
    #: ``"low"`` (router only) or ``"high"`` (reason then route).  Tools force high.
    reasoning: ReasoningLevel = "low"
    #: Tool names for the reasoning half.  Non-empty implies ``reasoning="high"``.
    tools: Sequence[str] = ()
    #: Reasoning prompt when ``reasoning="high"``.  A default is generated
    #: from the route labels when omitted.
    prompt: str = ""
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
        level = _resolve_reasoning_level(
            self.reasoning, self.tools, owner=f"SemanticRouter {self.id!r}"
        )
        object.__setattr__(self, "reasoning", level)
        object.__setattr__(self, "tools", tuple(self.tools))
        from ..node import Node
        if isinstance(self.fallback_node, Node) and not getattr(self.fallback_node, "is_fallback", False):
            object.__setattr__(self, "fallback_node", self.fallback_node.model_copy(update={"is_fallback": True}))

    @property
    def is_high_reasoning(self) -> bool:
        return self.reasoning == "high"

    @property
    def schema_id(self) -> str:
        """Router state id when high (``{id}.Schema``)."""
        return f"{self.id}{COMBINE_SCHEMA_SUFFIX}"

    @property
    def router_state_id(self) -> str:
        """FSM state that performs semantic routing."""
        return self.schema_id if self.is_high_reasoning else self.id

    def expand(self) -> tuple[list[Node], list[Edge]]:
        """When high, return the reasoning node and the link to the router."""
        if not self.is_high_reasoning:
            return [], []
        labels = ", ".join(self.routes)
        reasoning_prompt = self.prompt or (
            f"Reason about which route best matches the input. "
            f"Candidate labels: {labels}."
        )
        input_schema = self.input_schema or {"type": "object"}
        reasoning = ReasoningNode(
            id=self.id,
            input_schema=input_schema,
            tools=self.tools,
            prompt=reasoning_prompt,
            name=self.id,
            description=self.description,
            group=self.group,
            metadata={
                "combine_id": self.id,
                "combine_role": "reasoning",
                "router": True,
            },
            provider=self.provider,
        )
        object.__setattr__(reasoning, "kind", "combine_part")
        return [reasoning], [edge_deterministic(self.id, self.router_state_id)]

    def compile(self) -> list[Edge]:
        source = self.router_state_id
        edges: list[Edge] = []
        for label, target in self.routes.items():
            target_id, kind = _target_id(target)
            if kind == "router":
                # Enter another router state after the semantic choice — rare;
                # treat as a node-kind edge to the router id.
                edges.append(
                    edge_semantic(
                        source,
                        target_id,
                        target_kind="node",
                        description=f"{self.id}:{label} -> router:{target_id}",
                    )
                )
            elif kind == "group":
                edges.append(
                    edge_semantic(
                        source,
                        target_id,
                        target_kind="group",
                        description=f"{self.id}:{label} -> group:{target_id}",
                    )
                )
            else:
                edges.append(
                    edge_semantic(
                        source,
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
                    source,
                    fb_id,
                    description=f"{self.id} fallback -> {fb_id}",
                )
            )
        return edges

__all__ = [
    "SemanticRouter",
]
