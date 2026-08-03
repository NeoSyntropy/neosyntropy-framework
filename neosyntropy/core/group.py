"""Group: a named collection of nodes, with optional internal routing.

Groups organize nodes and, when targeted by a semantic edge, scope hybrid
candidate search (or land on ``entry`` when one is set). Authoring helpers
(``@group.node``, ``routers``, ``entry``, ``add_edge``) compile into the
parent FSM — groups are not a second control path at runtime.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr

from .edge import Edge, edge_deterministic
from .models import RuntimeModel


class Group(RuntimeModel):
    """Named node collection; optional authored subgraph for the parent FSM.

    Example::

        billing = Group(name="billing")

        @billing.node(id="ValidateCard", input_schema=OpenInput, output_schema=EmptyOutput)
        def validate(ctx):
            return ctx.result(output={}, state_updates={"card_valid": True})

        logic = DeterministicRouter(
            id="BillingLogic",
            rules=[(lambda ctx: ctx.state.get("card_valid") is True, "ProcessPayment")],
        )
        billing.routers = [logic]
        billing.entry = "ValidateCard"
        billing.add_edge("ValidateCard", "BillingLogic")
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    entry: str | None = None
    routers: list[Any] = Field(default_factory=list)

    _nodes: dict[str, Any] = PrivateAttr(default_factory=dict)
    _edges: list[tuple[str, str]] = PrivateAttr(default_factory=list)

    def node(
        self,
        id: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        provider: str = "backend",
        prompt: str = "",
        prerequisites: tuple[str, ...] | list[str] = (),
        tools: tuple[str, ...] | list[str] = (),
        input_schema: type[Any] | dict[str, Any] | None = None,
        output_schema: type[Any] | dict[str, Any] | None = None,
        is_fallback: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Any]:
        """Declare a handler node that belongs to this group."""
        from .node import node as node_decorator

        def decorator(fn: Callable[..., Any]) -> Any:
            item = node_decorator(
                id=id,
                name=name,
                description=description,
                provider=provider,
                prompt=prompt,
                prerequisites=prerequisites,
                tools=tools,
                input_schema=input_schema,
                output_schema=output_schema,
                group=self.name,
                is_fallback=is_fallback,
                metadata=metadata,
            )(fn)
            self._register_node(item)
            return item

        return decorator

    def add_node(self, item: Any) -> Any:
        """Register an existing node (or CombineNode) into this group."""
        from .node import CombineNode, Node

        if isinstance(item, CombineNode):
            if item.group not in (None, self.name):
                raise ValueError(
                    f"node {item.id!r} belongs to group {item.group!r}, "
                    f"not {self.name!r}"
                )
            item.group = self.name
            self._register_node(item)
            return item
        if isinstance(item, Node):
            if item.group not in (None, self.name):
                raise ValueError(
                    f"node {item.id!r} belongs to group {item.group!r}, "
                    f"not {self.name!r}"
                )
            # Node is a pydantic model; prefer a copy with the group set.
            bound = item.model_copy(update={"group": self.name})
            self._register_node(bound)
            return bound
        raise TypeError(
            f"Group.add_node expects Node or CombineNode; got {type(item)!r}"
        )

    def add_edge(self, source: str | Any, target: str | Any) -> None:
        """Connect a group node or router to another node or router.

        Compiles to a deterministic FSM edge when the group is attached to an
        ``FSM``. Typical use: node → internal router after the node finishes.
        """
        source_id = _endpoint_id(source)
        target_id = _endpoint_id(target)
        self._edges.append((source_id, target_id))

    @property
    def nodes(self) -> dict[str, Any]:
        """Nodes authored on this group (id → node)."""
        return dict(self._nodes)

    @property
    def declared_edges(self) -> list[tuple[str, str]]:
        """``(source, target)`` pairs from :meth:`add_edge`."""
        return list(self._edges)

    def authored_nodes(self) -> list[Any]:
        return list(self._nodes.values())

    def authored_routers(self) -> list[Any]:
        return list(self.routers or [])

    def compiled_edges(self) -> list[Edge]:
        """Deterministic edges from :meth:`add_edge`."""
        return [
            edge_deterministic(
                source,
                target,
                description=f"group {self.name}: {source} -> {target}",
            )
            for source, target in self._edges
        ]

    def entry_id(self) -> str | None:
        """Resolved entry state id, or ``None`` when unset."""
        if self.entry is None:
            return None
        return _endpoint_id(self.entry)

    def _register_node(self, item: Any) -> None:
        node_id = getattr(item, "id", None)
        if not node_id:
            raise ValueError("group node requires an id")
        if node_id in self._nodes:
            raise ValueError(
                f"duplicate node id {node_id!r} in group {self.name!r}"
            )
        self._nodes[node_id] = item


def _endpoint_id(endpoint: str | Any) -> str:
    if isinstance(endpoint, str):
        if not endpoint:
            raise ValueError("edge endpoint id must be non-empty")
        return endpoint
    for attr in ("id", "name"):
        value = getattr(endpoint, attr, None)
        if isinstance(value, str) and value:
            return value
    raise TypeError(
        f"edge endpoint must be a str or object with id; got {type(endpoint)!r}"
    )


def expand_authored_groups(
    groups: Sequence[Group],
) -> tuple[list[Any], list[Any], list[Edge]]:
    """Collect nodes, routers, and edges authored on groups."""
    nodes: list[Any] = []
    routers: list[Any] = []
    edges: list[Edge] = []
    seen_nodes: set[str] = set()
    seen_routers: set[str] = set()
    for group in groups:
        for item in group.authored_nodes():
            node_id = item.id
            if node_id in seen_nodes:
                raise ValueError(
                    f"duplicate authored node id {node_id!r} across groups"
                )
            seen_nodes.add(node_id)
            nodes.append(item)
        for item in group.authored_routers():
            router_id = getattr(item, "id", None)
            if not router_id:
                raise ValueError(
                    f"group {group.name!r} router is missing an id"
                )
            if router_id in seen_routers:
                continue
            seen_routers.add(router_id)
            routers.append(item)
        edges.extend(group.compiled_edges())
    return nodes, routers, edges
