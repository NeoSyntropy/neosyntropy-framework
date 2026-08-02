"""Graph: nodes + edges + groups, validated at construction.

The graph is the single source of permission. Search/selection is not
permission, router proposals are not permission — only edges listed here
(or an explicit ``allow_unlisted_transitions=True``) permit a transition.

Edge kinds:

- ``deterministic`` — auto-commit when exactly one guard matches
- ``semantic`` — scopes the semantic router to a node or group
- ``fallback`` — used when neither of the above yields a route

``input_schema`` declares the entry contract: the state a run must supply
when it starts at ``Start``. It is the counterpart to a node's
``output_schema`` — the workflow says what it takes in, not just what each
node hands back — and it is enforced fail-closed before any selection or
routing happens.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel

from .edge import Edge, TransitionTable
from .group import Group
from .node import Node
from .schemas import input_model_schema

START = "Start"
END = "End"


class GraphValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid graph: " + "; ".join(errors))


def _entry_schema(source: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Normalize an entry contract to a closed JSON Schema object."""
    if isinstance(source, type) and issubclass(source, BaseModel):
        return input_model_schema(source)
    if isinstance(source, dict) and source:
        return dict(source)
    raise GraphValidationError(
        [
            "input_schema must be a pydantic BaseModel class or a "
            "non-empty JSON Schema object"
        ]
    )


class Graph:
    def __init__(
        self,
        *,
        nodes: Iterable[Node],
        edges: Iterable[Edge | dict[str, Any]] = (),
        groups: Iterable[Group | str] = (),
        input_schema: type[BaseModel] | dict[str, Any] | None = None,
        allow_unlisted_transitions: bool = False,
        validate_reachability: bool = True,
    ):
        self.input_model: type[BaseModel] | None = (
            input_schema
            if isinstance(input_schema, type) and issubclass(input_schema, BaseModel)
            else None
        )
        self.input_schema: dict[str, Any] | None = (
            _entry_schema(input_schema) if input_schema is not None else None
        )
        self.nodes: dict[str, Node] = {}
        for item in nodes:
            if item.id in self.nodes:
                raise GraphValidationError([f"duplicate node id {item.id!r}"])
            self.nodes[item.id] = item

        self.edges: list[Edge] = [
            edge if isinstance(edge, Edge) else Edge.model_validate(edge) for edge in edges
        ]
        self.allow_unlisted_transitions = allow_unlisted_transitions

        self.groups: dict[str, Group] = {}
        for group in groups:
            resolved = group if isinstance(group, Group) else Group(name=group)
            self.groups[resolved.name] = resolved
        # Groups referenced by nodes are auto-registered.
        for item in self.nodes.values():
            if item.group and item.group not in self.groups:
                self.groups[item.group] = Group(name=item.group)

        self._validate(validate_reachability)

    # -- validation ---------------------------------------------------------

    def _validate(self, validate_reachability: bool) -> None:
        errors: list[str] = []
        if not self.nodes:
            errors.append("graph must define at least one node")

        known_nodes = set(self.nodes) | {START, END}
        for edge in self.edges:
            if edge.source not in known_nodes:
                errors.append(
                    f"edge source {edge.source!r} is not a known node "
                    f"(or Start/End)"
                )
            if edge.target_kind == "group":
                if edge.target not in self.groups:
                    errors.append(
                        f"semantic edge targets unknown group {edge.target!r}"
                    )
            elif edge.target not in known_nodes:
                errors.append(
                    f"edge target {edge.target!r} is not a known node "
                    f"(or Start/End)"
                )

        fallback_count = sum(item.is_fallback for item in self.nodes.values())
        if fallback_count != 1:
            errors.append("graph must define exactly one dedicated fallback node")

        missing_output = sorted(
            item.id for item in self.nodes.values() if not item.output_schema
        )
        if missing_output:
            errors.append(
                f"every node requires output_schema; missing on: {missing_output}"
            )
        missing_input = sorted(
            item.id for item in self.nodes.values() if not item.input_schema
        )
        if missing_input:
            errors.append(
                f"every node requires input_schema; missing on: {missing_input}"
            )

        if errors:
            raise GraphValidationError(errors)

        if validate_reachability and self.edges:
            self._validate_reachability(errors)
        if errors:
            raise GraphValidationError(errors)

    def _concrete_targets(self, edge: Edge) -> set[str]:
        if edge.target_kind == "group":
            return {node.id for node in self.nodes_in_group(edge.target)}
        return {edge.target}

    def _validate_reachability(self, errors: list[str]) -> None:
        """Start/End discipline with group-target expansion.

        Applies only to vertices that participate in edges: every such vertex
        must be reachable from Start, and (when End is used) must be able to
        reach End. The fallback node is exempt — it is a safe stop, not a
        path member. Capability-only nodes (no edges) are exempt as well;
        the plan validator fail-closes on them unless unlisted transitions
        are explicitly allowed.
        """
        participants: set[str] = set()
        forward: dict[str, set[str]] = {}
        backward: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.kind == "fallback":
                # Fallback edges are safe-stop exits, not primary path members.
                continue
            participants.add(edge.source)
            for target in self._concrete_targets(edge):
                participants.add(target)
                forward.setdefault(edge.source, set()).add(target)
                backward.setdefault(target, set()).add(edge.source)

        fallback_id = self.fallback_node.id
        if START not in participants:
            errors.append("edges are defined but none originate from 'Start'")
            return

        reachable = _flood(START, forward)
        unreachable = sorted(
            vertex
            for vertex in participants
            if vertex not in reachable and vertex not in {START, fallback_id}
        )
        if unreachable:
            errors.append(f"states unreachable from Start: {unreachable}")

        if END in participants:
            can_finish = _flood(END, backward)
            stuck = sorted(
                vertex
                for vertex in reachable
                if vertex not in can_finish and vertex not in {END, fallback_id}
            )
            if stuck:
                errors.append(f"states that cannot reach End: {stuck}")

    # -- entry contract -----------------------------------------------------

    def entry_input_error(self, state: Mapping[str, Any]) -> str | None:
        """Return why ``state`` fails the entry contract, or None when it holds.

        Only meaningful for a run starting at ``Start``: later cycles resume
        with state the workflow itself produced.
        """
        if self.input_schema is None:
            return None
        try:
            jsonschema.validate(instance=dict(state), schema=self.input_schema)
        except jsonschema.exceptions.ValidationError as exc:
            return f"entry input does not match the graph input schema: {exc.message}"
        except jsonschema.exceptions.SchemaError as exc:
            return f"graph input schema is invalid: {exc.message}"
        return None

    # -- permissions --------------------------------------------------------

    @property
    def fallback_node(self) -> Node:
        return next(item for item in self.nodes.values() if item.is_fallback)

    def transition_table(self) -> TransitionTable:
        return TransitionTable.from_edges(
            self.edges,
            node_groups={node_id: node.group for node_id, node in self.nodes.items()},
            allow_unlisted_transitions=self.allow_unlisted_transitions,
        )

    def allows(self, source: str, target: str) -> bool:
        return self.transition_table().permits(source, target)

    def guard_allows(self, source: str, target: str, state: dict[str, Any]) -> bool:
        """Evaluate guards for a transition, fail-closed per edge.

        Deterministic edges own the gate for a concrete target: when any
        deterministic edge lists ``source -> target``, at least one of those
        guards must allow. Semantic edges cannot bypass a failing
        deterministic guard. When no listed edge matches (permissive graphs
        or default self-transitions), guards are vacuously satisfied.
        """
        matching = [
            edge
            for edge in self.edges
            if edge.source == source
            and (
                (edge.target_kind == "node" and edge.target == target)
                or (
                    edge.target_kind == "group"
                    and self.nodes.get(target) is not None
                    and self.nodes[target].group == edge.target
                )
            )
        ]
        if not matching:
            return True
        deterministic = [edge for edge in matching if edge.kind == "deterministic"]
        if deterministic:
            return any(edge.guard_allows(state) for edge in deterministic)
        return any(edge.guard_allows(state) for edge in matching)

    def outgoing(self, source: str, *, kind: str | None = None) -> list[Edge]:
        edges = [edge for edge in self.edges if edge.source == source]
        if kind is None:
            return edges
        return [edge for edge in edges if edge.kind == kind]

    def matching_deterministic(
        self, source: str, state: dict[str, Any]
    ) -> list[Edge]:
        """Deterministic edges whose guards allow ``state`` (fail-closed)."""
        return [
            edge
            for edge in self.outgoing(source, kind="deterministic")
            if edge.guard_allows(state)
        ]

    def semantic_candidate_ids(self, source: str) -> set[str] | None:
        """Node ids in scope for the semantic router from ``source``.

        Returns ``None`` when there are no outgoing semantic edges (caller
        should use the fallback edge). Returns an empty set when semantic
        edges exist but resolve to no actionable nodes.
        """
        semantic = self.outgoing(source, kind="semantic")
        if not semantic:
            return None
        scoped: set[str] = set()
        for edge in semantic:
            for target in self._concrete_targets(edge):
                if target in self.nodes and not self.nodes[target].is_fallback:
                    scoped.add(target)
        return scoped

    def fallback_target(self, source: str) -> str:
        """Target of the fallback edge from ``source``, else the fallback node."""
        edges = self.outgoing(source, kind="fallback")
        if len(edges) == 1:
            return edges[0].target
        if len(edges) > 1:
            # Ambiguous fallback declarations fail closed to the dedicated node.
            return self.fallback_node.id
        return self.fallback_node.id

    # -- organization -------------------------------------------------------

    def nodes_in_group(self, name: str) -> list[Node]:
        return [item for item in self.nodes.values() if item.group == name]


def _flood(origin: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {origin}
    queue = deque([origin])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency.get(vertex, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen
