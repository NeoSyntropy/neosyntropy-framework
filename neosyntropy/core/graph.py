"""Graph: nodes + edges + groups + axioms, validated at construction.

The graph is the single source of permission. Search/selection is not
permission, router proposals are not permission — only edges listed here
(or an explicit ``allow_unlisted_transitions=True``) permit a transition.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from .axiom import Axiom
from .edge import Edge, TransitionTable
from .group import Group
from .node import Node

START = "Start"
END = "End"


class GraphValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid graph: " + "; ".join(errors))


class Graph:
    def __init__(
        self,
        *,
        nodes: Iterable[Node],
        edges: Iterable[Edge | dict[str, Any]] = (),
        groups: Iterable[Group | str] = (),
        axioms: Iterable[Axiom] = (),
        allow_unlisted_transitions: bool = False,
        validate_reachability: bool = True,
    ):
        self.nodes: dict[str, Node] = {}
        for item in nodes:
            if item.id in self.nodes:
                raise GraphValidationError([f"duplicate node id {item.id!r}"])
            self.nodes[item.id] = item

        self.edges: list[Edge] = [
            edge if isinstance(edge, Edge) else Edge.model_validate(edge) for edge in edges
        ]
        self.allow_unlisted_transitions = allow_unlisted_transitions
        self.axioms: list[Axiom] = list(axioms)

        self.groups: dict[str, Group] = {}
        for group in groups:
            resolved = group if isinstance(group, Group) else Group(name=group)
            self.groups[resolved.name] = resolved
        # Groups referenced by nodes are auto-registered; they organize, never control.
        for item in self.nodes.values():
            if item.group and item.group not in self.groups:
                self.groups[item.group] = Group(name=item.group)

        self._validate(validate_reachability)

    # -- validation ---------------------------------------------------------

    def _validate(self, validate_reachability: bool) -> None:
        errors: list[str] = []
        if not self.nodes:
            errors.append("graph must define at least one node")

        known = set(self.nodes) | {START, END}
        unknown = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source, edge.target)
                if endpoint not in known
            }
        )
        if unknown:
            errors.append(f"transitions reference unknown nodes: {unknown}")

        fallback_count = sum(item.is_fallback for item in self.nodes.values())
        if fallback_count != 1:
            errors.append("graph must define exactly one dedicated fallback node")

        if errors:
            raise GraphValidationError(errors)

        if validate_reachability and self.edges:
            self._validate_reachability(errors)
        if errors:
            raise GraphValidationError(errors)

    def _validate_reachability(self, errors: list[str]) -> None:
        """Start/End discipline, ported from the backend's compile-time gate.

        Applies only to vertices that participate in edges: every such vertex
        must be reachable from Start, and (when End is used) must be able to
        reach End. The fallback node is exempt — it is a safe stop, not a
        path member. Capability-only nodes (no edges) are exempt as well;
        the plan validator fail-closes on them unless unlisted transitions
        are explicitly allowed.
        """
        participants = {
            endpoint for edge in self.edges for endpoint in (edge.source, edge.target)
        }
        fallback_id = self.fallback_node.id
        if START not in participants:
            errors.append("edges are defined but none originate from 'Start'")
            return

        forward: dict[str, set[str]] = {}
        backward: dict[str, set[str]] = {}
        for edge in self.edges:
            forward.setdefault(edge.source, set()).add(edge.target)
            backward.setdefault(edge.target, set()).add(edge.source)

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

    # -- permissions --------------------------------------------------------

    @property
    def fallback_node(self) -> Node:
        return next(item for item in self.nodes.values() if item.is_fallback)

    def transition_table(self) -> TransitionTable:
        return TransitionTable.from_edges(
            self.edges, allow_unlisted_transitions=self.allow_unlisted_transitions
        )

    def allows(self, source: str, target: str) -> bool:
        if self.allow_unlisted_transitions:
            return True
        return any(edge.source == source and edge.target == target for edge in self.edges)

    def guard_allows(self, source: str, target: str, state: dict[str, Any]) -> bool:
        """Evaluate guards for a transition, fail-closed per edge.

        When several edges connect the same pair, any edge whose guard allows
        is sufficient. When no listed edge matches (permissive graphs or
        default self-transitions), guards are vacuously satisfied.
        """
        matching = [
            edge for edge in self.edges if edge.source == source and edge.target == target
        ]
        if not matching:
            return True
        return any(edge.guard_allows(state) for edge in matching)

    def outgoing(self, source: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == source]

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
