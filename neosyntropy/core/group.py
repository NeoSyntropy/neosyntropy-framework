"""Group: a named collection of nodes, with optional internal routing.

Groups organize nodes and, when targeted by a semantic edge, scope hybrid
candidate search (or land on ``entry`` when one is set). Authoring helpers
(``@group.node``, ``routers``, ``entry``, ``add_edge``) compile into the
parent FSM — groups are not a second control path at runtime.

FSM-like constructor (same shape as ``FSM``)::

    group = Group(
        name="billing",
        entry=validate_node,
        nodes=[validate_node, pay_node],
        edges=[
            edge_deterministic("ValidateCard", "ProcessPayment"),
            edge_deterministic("ProcessPayment", "End"),
        ],
    )
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .edge import Edge, edge_deterministic
from .models import RuntimeModel

_END = "End"


class Group(RuntimeModel):
    """Named node collection; optional authored subgraph for the parent FSM.

    Prefer the FSM-like constructor when the subgraph is known up front::

        group = Group(
            name="billing",
            entry=validate_node,
            nodes=[validate_node, pay_node],
            edges=[edge_deterministic("ValidateCard", "ProcessPayment")],
        )

    Or build incrementally with ``@group.node`` / ``add_edge`` / ``routers``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    entry: Any = None
    routers: list[Any] = Field(default_factory=list)
    #: Parent group name when this group is nested inside another (groups-in-groups).
    parent: str | None = None

    _nodes: dict[str, Any] = PrivateAttr(default_factory=dict)
    _edges: list[tuple[str, str]] = PrivateAttr(default_factory=list)
    _edge_objs: list[Edge] = PrivateAttr(default_factory=list)
    _namespace: bool = PrivateAttr(default=False)
    _child_groups: dict[str, Group] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        entry: Any = None,
        routers: Sequence[Any] | None = None,
        nodes: Sequence[Any] | None = None,
        edges: Sequence[Any] | None = None,
        groups: Sequence[Group] | None = None,
        namespace: bool | None = None,
        parent: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct a group; optional ``nodes`` / ``edges`` / ``groups`` mirror ``FSM``.

        ``groups`` nests child :class:`Group` subgraphs (groups-in-groups). Child
        groups are also auto-collected from :class:`SemanticRouter` route targets
        authored on this group.
        """
        if kwargs:
            raise TypeError(
                f"Group() got unexpected keyword argument(s): {sorted(kwargs)}"
            )
        do_namespace = bool(nodes) if namespace is None else bool(namespace)
        super().__init__(  # type: ignore
            name=name,
            description=description,
            metadata=dict(metadata or {}),
            entry=None,
            routers=[],
            parent=parent,
        )
        object.__setattr__(self, "_namespace", do_namespace)
        object.__setattr__(self, "_child_groups", {})
        self._ingest(
            entry=entry,
            routers=list(routers or []),
            nodes=list(nodes or []),
            edges=list(edges or []),
            groups=list(groups or []),
        )

    @model_validator(mode="after")
    def _coerce_entry_id(self) -> Group:
        # Incremental authoring may assign entry to a node/router object.
        if self.entry is not None and not isinstance(self.entry, str):
            object.__setattr__(self, "entry", _endpoint_id(self.entry))
        return self

    def _ingest(
        self,
        *,
        entry: Any,
        routers: list[Any],
        nodes: list[Any],
        edges: list[Any],
        groups: list[Group],
    ) -> None:
        id_map: dict[str, str] = {}
        node_map: dict[str, Any] = {}
        for item in nodes:
            bound, old_id, new_id = self._bind_node(item, clear_fallback=True)
            id_map[old_id] = new_id
            node_map[old_id] = bound
            node_map[new_id] = bound

        rebound_routers = [
            self._rebind_router(item, id_map, node_map) for item in routers
        ]
        for item in rebound_routers:
            rid = getattr(item, "id", None)
            if isinstance(rid, str) and rid:
                id_map.setdefault(rid, rid)
        object.__setattr__(self, "routers", rebound_routers)

        if entry is not None:
            object.__setattr__(self, "entry", self._rewrite_endpoint(entry, id_map))

        for edge in edges:
            self._ingest_edge(edge, id_map)

        for child in groups:
            self._add_child(child)
        # Semantic route targets that are Groups become nested children.
        from .routing.semantic import SemanticRouter

        for item in rebound_routers:
            if not isinstance(item, SemanticRouter):
                continue
            for target in item.routes.values():
                if isinstance(target, Group):
                    self._add_child(target)

    def _add_child(self, child: Group) -> None:
        """Register ``child`` as a nested subgroup of this group."""
        if not isinstance(child, Group):
            raise TypeError(
                f"Group groups= expects Group; got {type(child)!r}"
            )
        if child.name == self.name:
            raise ValueError(f"group {self.name!r} cannot nest itself")
        existing_parent = child.parent
        if existing_parent not in (None, self.name):
            raise ValueError(
                f"group {child.name!r} already nested under {existing_parent!r}, "
                f"cannot also nest under {self.name!r}"
            )
        object.__setattr__(child, "parent", self.name)
        self._child_groups[child.name] = child

    def _bind_node(self, item: Any, *, clear_fallback: bool) -> tuple[Any, str, str]:
        from .node import CombineNode, Node

        if isinstance(item, CombineNode):
            old_id = item.id
            new_id = self._namespaced(old_id)
            if item.group not in (None, self.name):
                raise ValueError(
                    f"node {old_id!r} belongs to group {item.group!r}, "
                    f"not {self.name!r}"
                )
            item.group = self.name
            if new_id != old_id:
                item.id = new_id
            if clear_fallback:
                item.is_fallback = False
            self._register_node(item)
            return item, old_id, new_id

        if isinstance(item, Node):
            old_id = item.id
            new_id = self._namespaced(old_id)
            if item.group not in (None, self.name):
                raise ValueError(
                    f"node {old_id!r} belongs to group {item.group!r}, "
                    f"not {self.name!r}"
                )
            updates: dict[str, Any] = {"group": self.name}
            if new_id != old_id:
                updates["id"] = new_id
            if clear_fallback and item.is_fallback:
                updates["is_fallback"] = False
            bound = item.model_copy(update=updates)
            self._register_node(bound)
            return bound, old_id, new_id

        raise TypeError(
            f"Group nodes= expects Node or CombineNode; got {type(item)!r}"
        )

    def _namespaced(self, node_id: str) -> str:
        if not self._namespace or node_id == _END:
            return node_id
        prefix = f"{self.name}__"
        if node_id.startswith(prefix):
            return node_id
        return f"{prefix}{node_id}"

    def _rewrite_endpoint(self, endpoint: Any, id_map: dict[str, str]) -> str:
        if isinstance(endpoint, str):
            if endpoint == _END:
                return endpoint
            return id_map.get(endpoint, self._namespaced(endpoint))
        endpoint_id = _endpoint_id(endpoint)
        if endpoint_id == _END:
            return endpoint_id
        return id_map.get(endpoint_id, self._namespaced(endpoint_id))

    def _rebind_router(
        self,
        router: Any,
        id_map: dict[str, str],
        node_map: dict[str, Any],
    ) -> Any:
        from .routing.deterministic import DeterministicRouter
        from .routing.semantic import SemanticRouter

        def map_target(target: Any) -> Any:
            if target is None:
                return None
            if isinstance(target, (DeterministicRouter, SemanticRouter)):
                return self._rebind_router(target, id_map, node_map)
            if isinstance(target, Group):
                return target
            if isinstance(target, str):
                if target == _END:
                    return target
                return id_map.get(target, self._namespaced(target))
            tid = getattr(target, "id", None)
            if isinstance(tid, str) and tid in node_map:
                return node_map[tid]
            if isinstance(tid, str) and tid in id_map:
                return node_map.get(id_map[tid], target)
            return target

        old_id = getattr(router, "id", None)
        new_id = self._namespaced(old_id) if isinstance(old_id, str) else old_id
        if isinstance(old_id, str) and new_id != old_id:
            id_map[old_id] = new_id

        if isinstance(router, SemanticRouter):
            return SemanticRouter(
                id=new_id or router.id,
                routes={label: map_target(t) for label, t in router.routes.items()},
                fallback_node=map_target(router.fallback_node),
                description=router.description,
                category=router.category,
                input_schema=router.input_model or router.input_schema,
                group=self.name,
                provider=getattr(router, "provider", "neosyntropy/base")
                or "neosyntropy/base",
                reasoning=getattr(router, "reasoning", "low"),
                tools=tuple(getattr(router, "tools", ()) or ()),
                prompt=getattr(router, "prompt", "") or "",
            )
        if isinstance(router, DeterministicRouter):
            return DeterministicRouter(
                id=new_id or router.id,
                rules=[(pred, map_target(t)) for pred, t in router.rules],
                description=router.description,
                input_schema=router.input_model or router.input_schema,
                group=self.name,
            )
        return router

    def _ingest_edge(self, edge: Any, id_map: dict[str, str]) -> None:
        if isinstance(edge, Edge):
            source = self._rewrite_endpoint(edge.source, id_map)
            target = self._rewrite_endpoint(edge.target, id_map)
            self._edge_objs.append(
                edge.model_copy(
                    update={
                        "source": source,
                        "target": target,
                        "description": edge.description
                        or f"group {self.name}: {source} -> {target}",
                    }
                )
            )
            return
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            self.add_edge(
                self._rewrite_endpoint(edge[0], id_map),
                self._rewrite_endpoint(edge[1], id_map),
            )
            return
        raise TypeError(
            f"Group edges= expects Edge or (source, target); got {type(edge)!r}"
        )

    def node(
        self,
        id: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        provider: str = "neosyntropy/base",
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
        """Edges from ``edges=`` / :meth:`add_edge`."""
        from_pairs = [
            edge_deterministic(
                source,
                target,
                description=f"group {self.name}: {source} -> {target}",
            )
            for source, target in self._edges
        ]
        return [*self._edge_objs, *from_pairs]

    def compile(self) -> list[Edge]:
        """Compile the group's declared edges. Returns compiled_edges()."""
        return self.compiled_edges()

    def entry_id(self) -> str | None:
        """Resolved entry state id, or ``None`` when unset."""
        if self.entry is None:
            return None
        return _endpoint_id(self.entry)

    def child_groups(self) -> list[Group]:
        """Nested subgroup instances (groups-in-groups), in insertion order."""
        return list(self._child_groups.values())

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
    """Collect nodes, routers, and edges authored on groups (including nested)."""
    nodes: list[Any] = []
    routers: list[Any] = []
    edges: list[Edge] = []
    seen_nodes: set[str] = set()
    seen_routers: set[str] = set()
    seen_groups: set[str] = set()

    def walk(group: Group) -> None:
        if group.name in seen_groups:
            return
        seen_groups.add(group.name)
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
        for child in group.child_groups():
            walk(child)

    for group in groups:
        walk(group)
    return nodes, routers, edges


def flatten_group_tree(groups: Sequence[Group]) -> list[Group]:
    """Depth-first list of every group in ``groups`` including nested children."""
    out: list[Group] = []
    seen: set[str] = set()

    def walk(group: Group) -> None:
        if group.name in seen:
            return
        seen.add(group.name)
        out.append(group)
        for child in group.child_groups():
            walk(child)

    for group in groups:
        walk(group)
    return out
