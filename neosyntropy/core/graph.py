"""FSM: nodes + edges + groups, validated at construction.

The FSM is the single source of permission. Search/selection is not
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

from .edge import Edge, TransitionTable, edge_deterministic, edge_fallback
from .group import Group, expand_authored_groups
from .node import CombineNode, Node
from .schemas import input_model_schema

START = "Start"
END = "End"

FSMNode = Node | CombineNode


def _flatten_nodes(
    items: Iterable[FSMNode],
) -> tuple[list[Node], list[Edge]]:
    """Expand CombineNode authoring units into concrete nodes + link edges."""
    nodes: list[Node] = []
    auto_edges: list[Edge] = []
    for item in items:
        if isinstance(item, CombineNode):
            expanded, links = item.expand()
            nodes.extend(expanded)
            auto_edges.extend(links)
        elif isinstance(item, Node):
            nodes.append(item)
        else:
            raise FSMValidationError(
                [f"FSM nodes must be Node or CombineNode; got {type(item)!r}"]
            )
    return nodes, auto_edges


def _resolve_routers(
    routers: Iterable[Any],
    entry: Any,
) -> tuple[list[Any], set[str], list[Edge]]:
    """Collect router declarations (including nested) and compile edges."""
    from ..routing.declarations import (
        DeterministicRouter,
        SemanticRouter,
        collect_nested_routers,
        compile_routers,
    )

    roots = list(routers)
    if entry is not None and isinstance(entry, (DeterministicRouter, SemanticRouter)):
        if entry not in roots and not any(r.id == entry.id for r in roots):
            roots.insert(0, entry)

    if not roots:
        return [], set(), []

    collected: dict[str, Any] = {}
    for root in roots:
        if not isinstance(root, (DeterministicRouter, SemanticRouter)):
            raise FSMValidationError(
                [f"routers must be DeterministicRouter or SemanticRouter; got {type(root)!r}"]
            )
        for item in collect_nested_routers(root):
            collected[item.id] = item

    ordered = list(collected.values())
    return ordered, set(collected), compile_routers(ordered)


class FSMValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid FSM: " + "; ".join(errors))


def _entry_schema(source: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Normalize an entry contract to a closed JSON Schema object."""
    if isinstance(source, type) and issubclass(source, BaseModel):
        return input_model_schema(source)
    if isinstance(source, dict) and source:
        return dict(source)
    raise FSMValidationError(
        [
            "input_schema must be a pydantic BaseModel class or a "
            "non-empty JSON Schema object"
        ]
    )


class FSM:
    def __init__(
        self,
        *,
        nodes: Iterable[FSMNode],
        edges: Iterable[Edge | dict[str, Any]] = (),
        groups: Iterable[Group | str] = (),
        routers: Iterable[Any] = (),
        entry: Any = None,
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

        from ..routing.declarations import DeterministicRouter, SemanticRouter

        self.groups: dict[str, Group] = {}
        resolved_groups: list[Group] = []
        for group in groups:
            resolved = group if isinstance(group, Group) else Group(name=group)
            self.groups[resolved.name] = resolved
            resolved_groups.append(resolved)

        # Groups used as semantic route targets may also author nodes/routers.
        for root in [*routers, entry]:
            if not isinstance(root, (DeterministicRouter, SemanticRouter)):
                continue
            from ..routing.declarations import collect_nested_routers

            for item in collect_nested_routers(root):
                if not isinstance(item, SemanticRouter):
                    continue
                for target in item.routes.values():
                    if isinstance(target, Group) and target.name not in self.groups:
                        self.groups[target.name] = target
                        resolved_groups.append(target)

        try:
            group_nodes, group_routers, group_edges = expand_authored_groups(
                resolved_groups
            )
        except ValueError as exc:
            raise FSMValidationError([str(exc)]) from exc

        flat_nodes, auto_edges = _flatten_nodes([*nodes, *group_nodes])
        self.nodes: dict[str, Node] = {}
        for item in flat_nodes:
            if item.id in self.nodes:
                raise FSMValidationError([f"duplicate node id {item.id!r}"])
            self.nodes[item.id] = item

        resolved_routers, router_ids, router_edges = _resolve_routers(
            [*routers, *group_routers], entry
        )
        self.routers = {item.id: item for item in resolved_routers}
        self.router_ids = set(router_ids)

        overlap = self.router_ids & set(self.nodes)
        if overlap:
            raise FSMValidationError(
                [f"router id clashes with node id: {sorted(overlap)}"]
            )

        declared = [
            edge if isinstance(edge, Edge) else Edge.model_validate(edge)
            for edge in edges
        ]
        entry_edges: list[Edge] = []
        if entry is not None:
            from ..core.node import Node as NodeType

            if isinstance(entry, (DeterministicRouter, SemanticRouter)):
                entry_id = entry.id
            elif isinstance(entry, CombineNode):
                entry_id = entry.id
            elif isinstance(entry, NodeType):
                entry_id = entry.id
            elif isinstance(entry, str):
                entry_id = entry
            else:
                raise FSMValidationError(
                    [f"entry must be a router, node, or id; got {type(entry)!r}"]
                )
            entry_edges.append(edge_deterministic(START, entry_id))

        # Router/Combine/group auto edges first; author edges may still add more.
        self.edges: list[Edge] = [
            *auto_edges,
            *router_edges,
            *group_edges,
            *entry_edges,
            *declared,
        ]
        self.allow_unlisted_transitions = allow_unlisted_transitions

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

        known_nodes = set(self.nodes) | set(self.router_ids) | {START, END}
        for edge in self.edges:
            if edge.source not in known_nodes:
                errors.append(
                    f"edge source {edge.source!r} is not a known node, "
                    f"router, Start, or End"
                )
            if edge.target_kind == "group":
                if edge.target not in self.groups:
                    errors.append(
                        f"semantic edge targets unknown group {edge.target!r}"
                    )
            elif edge.target not in known_nodes:
                errors.append(
                    f"edge target {edge.target!r} is not a known node, "
                    f"router, Start, or End"
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

        for group in self.groups.values():
            entry_id = group.entry_id()
            if entry_id is not None and entry_id not in known_nodes:
                errors.append(
                    f"group {group.name!r} entry {entry_id!r} is not a known "
                    f"node or router"
                )

        if errors:
            raise FSMValidationError(errors)

        if validate_reachability and self.edges:
            self._validate_reachability(errors)
        if errors:
            raise FSMValidationError(errors)

    def _concrete_targets(self, edge: Edge) -> set[str]:
        if edge.target_kind == "group":
            group = self.groups.get(edge.target)
            entry_id = group.entry_id() if group is not None else None
            if entry_id is not None:
                return {entry_id}
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
        group_entries = {
            name: entry
            for name, group in self.groups.items()
            if (entry := group.entry_id()) is not None
        }
        return TransitionTable.from_edges(
            self.edges,
            node_groups={node_id: node.group for node_id, node in self.nodes.items()},
            group_entries=group_entries,
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

    def is_router_state(self, state_id: str) -> bool:
        """True when ``state_id`` is a compiled router (not an executable node)."""
        return state_id in self.router_ids

    # -- run (control cycle) ------------------------------------------------

    def run(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        intent: str | None = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Run the FSM. Workflows always begin at ``Start`` — do not pass it.

        Application code::

            result = fsm.run(intent="...", state={...}, client=client)

        By default ``until_end=True`` advances cycles until ``End`` or rejection.
        Pass ``until_end=False`` for a single control cycle (resume via
        ``current_state`` only when you intentionally continue mid-path).
        """
        return self._run_loop(
            request,
            client=client,
            tools=tools,
            intent=intent,
            state=state,
            until_end=until_end,
            max_cycles=max_cycles,
            **kwargs,
        )

    async def arun(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        intent: str | None = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Async form of :meth:`run`."""
        import asyncio

        # Reuse the sync loop via a thread when callers already use arun with
        # until_end; single-cycle path stays fully async.
        payload = self._normalize_request(request, intent=intent, state=state)
        manager = self._control_manager(client=client, tools=tools, **kwargs)
        if not until_end:
            return await manager.arun(payload)

        return await asyncio.to_thread(
            self._run_loop,
            payload,
            client=client,
            tools=tools,
            until_end=True,
            max_cycles=max_cycles,
            **kwargs,
        )

    def _normalize_request(
        self,
        request: Any,
        *,
        intent: str | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .models import RunRequest

        if request is None:
            if not intent:
                raise ValueError("fsm.run requires intent= or a RunRequest")
            payload: dict[str, Any] = {"intent": intent, "state": dict(state or {})}
        elif isinstance(request, RunRequest):
            payload = request.model_dump()
            if intent is not None:
                payload["intent"] = intent
            if state is not None:
                payload["state"] = dict(state)
        elif isinstance(request, Mapping):
            payload = dict(request)
            if intent is not None:
                payload["intent"] = intent
            if state is not None:
                payload["state"] = dict(state)
        else:
            raise TypeError(
                f"request must be RunRequest, mapping, or omitted; got {type(request)!r}"
            )

        # Workflows always enter at Start unless the caller resumes mid-path.
        payload.setdefault("current_state", START)
        if not payload.get("current_state"):
            payload["current_state"] = START
        payload.setdefault("prior_executions", [])
        payload.setdefault("state", {})
        return payload

    def _follow_terminal_edge(self, current: str, state: Mapping[str, Any]) -> str:
        """If the only deterministic exit is End, advance without another node run."""
        if current in {START, END}:
            return current
        matching = self.matching_deterministic(current, dict(state))
        if len(matching) == 1 and matching[0].target == END:
            return END
        return current

    def _run_loop(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        intent: str | None = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        payload = self._normalize_request(request, intent=intent, state=state)
        manager = self._control_manager(client=client, tools=tools, **kwargs)

        if not until_end:
            return manager.run(payload)

        last = None
        prior: list[dict[str, Any]] = list(payload.get("prior_executions") or [])
        current = str(payload.get("current_state") or START)
        initial_state = current
        snapshot = dict(payload.get("state") or {})
        intent_text = str(payload["intent"])
        all_steps: list[Any] = []
        all_transitions: list[str] = []
        all_gates: list[Any] = []
        step_offset = 0

        for _ in range(max_cycles):
            cycle_request = {
                **payload,
                "intent": intent_text,
                "current_state": current,
                "state": snapshot,
                "prior_executions": prior,
            }
            last = manager.run(cycle_request)
            snapshot = dict(last.state)
            current = last.final_state
            for step in last.steps:
                all_steps.append(
                    step.model_copy(update={"step": step_offset + step.step})
                )
            step_offset += len(last.steps)
            all_transitions.extend(last.audit.committed_transitions)
            all_gates.extend(last.audit.gate_checks)
            prior = prior + [
                {
                    "node_id": item.node_id,
                    "status": item.status,
                    "output": item.output,
                    "state_updates": item.state_updates,
                }
                for step in last.steps
                for item in step.results
            ]
            if last.rejected:
                return last.model_copy(
                    update={
                        "steps": all_steps,
                        "audit": last.audit.model_copy(
                            update={
                                "initial_state": initial_state,
                                "steps": all_steps,
                                "committed_transitions": all_transitions,
                                "gate_checks": all_gates,
                            }
                        ),
                    }
                )
            previous = current
            current = self._follow_terminal_edge(current, snapshot)
            if current == END and previous != END:
                all_transitions.append(f"{previous}->{END}")
            if current == END:
                return last.model_copy(
                    update={
                        "final_state": END,
                        "completed": True,
                        "steps": all_steps,
                        "audit": last.audit.model_copy(
                            update={
                                "initial_state": initial_state,
                                "final_state": END,
                                "steps": all_steps,
                                "committed_transitions": all_transitions,
                                "gate_checks": all_gates,
                            }
                        ),
                    }
                )

        raise RuntimeError(
            f"FSM did not reach End within {max_cycles} cycles "
            f"(last state {current!r})"
        )

    def _control_manager(
        self,
        *,
        client: Any = None,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        from ..control.manager import ControlManager

        return ControlManager(self, client=client, tools=tools, **kwargs)


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


def _entry_id(item: FSMNode) -> str:
    """FSM state id to enter this authored unit."""
    return item.id


def _exit_id(item: FSMNode) -> str:
    """FSM state id that leaves this authored unit (CombineNode → ``{id}.Schema``)."""
    if isinstance(item, CombineNode):
        return item.schema_id
    return item.id


def Workflow(
    sequence: list[FSMNode] | tuple[FSMNode, ...],
    *,
    fallback: FSMNode | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
) -> FSM:
    """Build a linear FSM from an ordered node list.

    Developers pass the happy-path sequence; ``Workflow`` wires
    ``Start → … → End`` and the optional fallback. No manual edges required.

    ``CombineNode`` units expand as usual: the next step links from
    ``{id}.Schema``, not from the reasoning entry.
    """
    steps = list(sequence)
    if not steps:
        raise FSMValidationError(["Workflow sequence must contain at least one node"])
    if fallback is None:
        raise FSMValidationError(
            ["Workflow requires a fallback node (exactly one per FSM)"]
        )
    if isinstance(fallback, Node) and not fallback.is_fallback:
        raise FSMValidationError(
            [f"fallback node {fallback.id!r} must set is_fallback=True"]
        )
    if isinstance(fallback, CombineNode):
        raise FSMValidationError(
            ["fallback cannot be a CombineNode; use SchemaNode(..., is_fallback=True)"]
        )

    edges: list[Edge] = [edge_deterministic(START, _entry_id(steps[0]))]
    for index in range(len(steps) - 1):
        edges.append(
            edge_deterministic(_exit_id(steps[index]), _entry_id(steps[index + 1]))
        )
    edges.append(edge_deterministic(_exit_id(steps[-1]), END))
    edges.append(edge_fallback(START, fallback.id))

    return FSM(
        nodes=[*steps, fallback],
        edges=edges,
        input_schema=input_schema,
    )


# Backward-compatible aliases
Graph = FSM
GraphValidationError = FSMValidationError
