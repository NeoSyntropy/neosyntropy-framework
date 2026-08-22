"""FSM: nodes + edges + groups, validated at construction.

The FSM is the single source of permission. Search/selection is not
permission, router proposals are not permission — only edges listed here
(or an explicit ``allow_unlisted_transitions=True``) permit a transition.

Edge kinds:

- ``deterministic`` — auto-commit when exactly one guard matches
- ``semantic`` — scopes the semantic router to a node or group
- ``fallback`` — used when neither of the above yields a route

``entry`` is required. It is a node or router with ``input_schema``; that
schema is the workflow entry contract (derived onto ``FSM.input_schema``).
Runs begin at ``entry.id`` — there is no synthetic Start vertex.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel

from .edge import Edge, TransitionTable, edge_deterministic, edge_fallback
from .group import Group, expand_authored_groups, flatten_group_tree
from .node import CombineNode, Node

END = "End"
_FORBIDDEN_START = "Start"

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


def _resolve_entry_id(entry: Any) -> str:
    """Return the state id for an authored entry (router, node, or id)."""
    from ..routing.declarations import DeterministicRouter, SemanticRouter

    if isinstance(entry, (DeterministicRouter, SemanticRouter)):
        return entry.id
    if isinstance(entry, CombineNode):
        return entry.id
    if isinstance(entry, Node):
        return entry.id
    if isinstance(entry, str) and entry.strip():
        return entry
    raise FSMValidationError(
        [f"entry must be a router, node, or id; got {type(entry)!r}"]
    )


def _entry_contract(
    entry: Any,
    entry_id: str,
    nodes: dict[str, Node],
    routers: dict[str, Any],
) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """Derive the FSM entry input_schema from the entry node or router."""
    from ..routing.declarations import DeterministicRouter, SemanticRouter

    if isinstance(entry, (DeterministicRouter, SemanticRouter)):
        schema = entry.json_schema
        if not schema:
            raise FSMValidationError(
                [
                    f"entry router {entry_id!r} requires input_schema "
                    "(it is the workflow entry contract)"
                ]
            )
        return dict(schema), entry.input_model

    if isinstance(entry, CombineNode):
        node = nodes.get(entry.id)
        if node is None or not node.input_schema:
            raise FSMValidationError(
                [f"entry CombineNode {entry_id!r} requires input_schema"]
            )
        return dict(node.input_schema), node.input_model

    if isinstance(entry, Node):
        if not entry.input_schema:
            raise FSMValidationError(
                [f"entry node {entry_id!r} requires input_schema"]
            )
        return dict(entry.input_schema), entry.input_model

    # String id — resolve against assembled nodes/routers.
    if entry_id in routers:
        router = routers[entry_id]
        schema = getattr(router, "json_schema", None)
        if not schema:
            raise FSMValidationError(
                [
                    f"entry router {entry_id!r} requires input_schema "
                    "(it is the workflow entry contract)"
                ]
            )
        return dict(schema), getattr(router, "input_model", None)
    node = nodes.get(entry_id)
    if node is None:
        raise FSMValidationError(
            [f"entry {entry_id!r} is not a known node or router"]
        )
    if not node.input_schema:
        raise FSMValidationError(
            [f"entry node {entry_id!r} requires input_schema"]
        )
    return dict(node.input_schema), node.input_model


class FSM:
    def __init__(
        self,
        *,
        nodes: Iterable[FSMNode],
        entry: Any,
        edges: Iterable[Edge | dict[str, Any]] = (),
        groups: Iterable[Group | str] = (),
        routers: Iterable[Any] = (),
        allow_unlisted_transitions: bool = False,
        validate_reachability: bool = True,
    ):
        if entry is None:
            raise FSMValidationError(
                ["FSM requires entry= (a node or router with input_schema)"]
            )

        from ..routing.declarations import DeterministicRouter, SemanticRouter

        self.groups: dict[str, Group] = {}
        resolved_groups: list[Group] = []
        for group in groups:
            resolved = group if isinstance(group, Group) else Group(name=group)
            for item in flatten_group_tree([resolved]):
                if item.name not in self.groups:
                    self.groups[item.name] = item
                    resolved_groups.append(item)

        # Groups used as semantic route targets may also author nodes/routers.
        for root in [*routers, entry]:
            if not isinstance(root, (DeterministicRouter, SemanticRouter)):
                continue
            from ..routing.declarations import collect_nested_routers

            for item in collect_nested_routers(root):
                if not isinstance(item, SemanticRouter):
                    continue
                for target in item.routes.values():
                    if isinstance(target, Group):
                        for nested in flatten_group_tree([target]):
                            if nested.name not in self.groups:
                                self.groups[nested.name] = nested
                                resolved_groups.append(nested)

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

        self.entry_id = _resolve_entry_id(entry)
        if self.entry_id == _FORBIDDEN_START:
            raise FSMValidationError(
                ["entry cannot be the removed synthetic 'Start' state"]
            )
        if (
            self.entry_id not in self.nodes
            and self.entry_id not in self.router_ids
        ):
            raise FSMValidationError(
                [f"entry {self.entry_id!r} is not a known node or router"]
            )

        self.input_schema, self.input_model = _entry_contract(
            entry, self.entry_id, self.nodes, self.routers
        )

        declared = [
            edge if isinstance(edge, Edge) else Edge.model_validate(edge)
            for edge in edges
        ]
        # Router/Combine/group auto edges first; author edges may still add more.
        self.edges: list[Edge] = [
            *auto_edges,
            *router_edges,
            *group_edges,
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
        if not self.entry_id:
            errors.append("FSM requires entry=")
        if not self.input_schema:
            errors.append(
                "FSM entry must declare input_schema"
            )

        known_nodes = set(self.nodes) | set(self.router_ids) | {END}
        for edge in self.edges:
            if edge.source == _FORBIDDEN_START or edge.target == _FORBIDDEN_START:
                errors.append(
                    "edges must not use the removed synthetic 'Start' state; "
                    "set entry= instead"
                )
                continue
            if edge.source not in known_nodes:
                errors.append(
                    f"edge source {edge.source!r} is not a known node, "
                    f"router, or End"
                )
            if edge.target_kind == "group":
                if edge.target not in self.groups:
                    errors.append(
                        f"semantic edge targets unknown group {edge.target!r}"
                    )
            elif edge.target not in known_nodes:
                errors.append(
                    f"edge target {edge.target!r} is not a known node, "
                    f"router, or End"
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
            group_entry = group.entry_id()
            if group_entry is not None and group_entry not in known_nodes:
                errors.append(
                    f"group {group.name!r} entry {group_entry!r} is not a known "
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
            group_entry = group.entry_id() if group is not None else None
            if group_entry is not None:
                return {group_entry}
            return {node.id for node in self.nodes_in_group(edge.target)}
        return {edge.target}

    def _validate_reachability(self, errors: list[str]) -> None:
        """Entry/End discipline with group-target expansion.

        Applies only to vertices that participate in edges: every such vertex
        must be reachable from the declared entry, and (when End is used) must
        be able to reach End. The fallback node is exempt — it is a safe stop,
        not a path member. Capability-only nodes (no edges) are exempt as well;
        the plan validator fail-closes on them unless unlisted transitions
        are explicitly allowed.
        """
        participants: set[str] = {self.entry_id}
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
        reachable = _flood(self.entry_id, forward)
        unreachable = sorted(
            vertex
            for vertex in participants
            if vertex not in reachable
            and vertex not in {self.entry_id, fallback_id}
        )
        if unreachable:
            errors.append(
                f"states unreachable from entry {self.entry_id!r}: {unreachable}"
            )

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

    def entry_input_error(self, payload: Mapping[str, Any]) -> str | None:
        """Return why ``payload`` fails the entry input contract, or None when it holds.

        Only meaningful for a run starting at ``entry_id``. The contract is
        derived from the entry node/router ``input_schema`` and validates the
        run ``input`` channel — not workflow ``state``.
        """
        if not self.input_schema:
            return "graph is missing required entry input_schema"
        try:
            jsonschema.validate(instance=dict(payload), schema=self.input_schema)
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
        """Deterministic edges whose guards allow ``state`` (fail-closed).

        Order matches compile / authoring order (DeterministicRouter: first
        matching rule wins — use :meth:`first_matching_deterministic`).
        """
        return [
            edge
            for edge in self.outgoing(source, kind="deterministic")
            if edge.guard_allows(state)
        ]

    def first_matching_deterministic(
        self, source: str, state: dict[str, Any]
    ) -> Edge | None:
        """First guard-allowed deterministic edge (DeterministicRouter semantics)."""
        for edge in self.outgoing(source, kind="deterministic"):
            if edge.guard_allows(state):
                return edge
        return None

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
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Run the FSM. Workflows begin at ``entry`` — do not pass a Start state.

        Application code::

            result = fsm.run(EntryInput(...), state={...}, client=client)

        ``request`` (or the first positional) is the run ``input`` and must match
        the entry ``input_schema``. ``state`` is a separate mutable workflow bag.

        By default ``until_end=True`` advances cycles until ``End`` or rejection.
        Pass ``until_end=False`` for a single control cycle (resume via
        ``current_state`` only when you intentionally continue mid-path).
        """
        return self._run_loop(
            request,
            client=client,
            tools=tools,
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
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Async form of :meth:`run`."""
        import asyncio

        # Reuse the sync loop via a thread when callers already use arun with
        # until_end; single-cycle path stays fully async.
        payload = self._normalize_request(request, state=state)
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

    def run_batch(
        self,
        requests: list[Any],
        *,
        batch_size: int = 50,
        client: Any = None,
        tools: Any = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> list[Any]:
        """Run the FSM over a batch of requests concurrently using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        # Limit the number of concurrent threads to batch_size
        max_workers = min(batch_size, len(requests)) if requests else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.run,
                    req,
                    client=client,
                    tools=tools,
                    until_end=until_end,
                    max_cycles=max_cycles,
                    **kwargs
                )
                for req in requests
            ]
            for future in futures:
                results.append(future.result())
        return results

    async def arun_batch(
        self,
        requests: list[Any],
        *,
        batch_size: int = 50,
        client: Any = None,
        tools: Any = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> list[Any]:
        """Async run the FSM over a batch of requests concurrently."""
        import asyncio
        
        async def _bounded_arun(semaphore: asyncio.Semaphore, req: Any):
            async with semaphore:
                return await self.arun(
                    req,
                    client=client,
                    tools=tools,
                    until_end=until_end,
                    max_cycles=max_cycles,
                    **kwargs
                )
                
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [_bounded_arun(semaphore, req) for req in requests]
        return await asyncio.gather(*tasks)

    def _normalize_request(
        self,
        request: Any,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from pydantic import BaseModel
        from .models import RunRequest

        payload: dict[str, Any] = {}
        if request is None:
            payload = {"input": {}, "state": dict(state or {})}
        elif isinstance(request, RunRequest):
            payload = request.model_dump()
            if state is not None:
                payload["state"] = dict(state)
        elif isinstance(request, BaseModel):
            payload = {"input": request.model_dump(), "state": dict(state or {})}
        elif isinstance(request, Mapping):
            req_dict = dict(request)
            if "prior_executions" in req_dict or "current_state" in req_dict:
                payload = req_dict
                if "input" not in payload and "intent" in payload:
                    # Legacy resume payloads used intent= as free text.
                    legacy = payload.pop("intent")
                    payload["input"] = (
                        legacy if isinstance(legacy, dict) else {"text": str(legacy)}
                    )
                if state is not None:
                    payload["state"] = dict(state)
            elif "input" in req_dict or "state" in req_dict:
                payload = {
                    "input": dict(req_dict.get("input") or {}),
                    "state": dict(state if state is not None else req_dict.get("state") or {}),
                }
                for key in ("current_state", "prior_executions", "request_id", "metadata", "history"):
                    if key in req_dict:
                        payload[key] = req_dict[key]
            else:
                payload = {"input": req_dict, "state": dict(state or {})}
        else:
            raise TypeError(
                f"request must be RunRequest, BaseModel, mapping, or omitted; got {type(request)!r}"
            )

        # Workflows enter at the declared entry unless the caller resumes mid-path.
        payload.setdefault("current_state", self.entry_id)
        if not payload.get("current_state"):
            payload["current_state"] = self.entry_id
        payload.setdefault("prior_executions", [])
        if "input" not in payload:
            payload["input"] = {}
        elif not isinstance(payload["input"], dict):
            payload["input"] = {"text": str(payload["input"])}
        payload.setdefault("state", {})
        payload.pop("intent", None)
        return payload

    def _follow_terminal_edge(self, current: str, state: Mapping[str, Any]) -> str:
        """If the first matching deterministic exit is End, advance without another node run."""
        if current == END:
            return current
        matching = self.first_matching_deterministic(current, dict(state))
        if matching is not None and matching.target == END:
            return END
        return current

    def _run_loop(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        payload = self._normalize_request(request, state=state)
        manager = self._control_manager(client=client, tools=tools, **kwargs)

        if not until_end:
            return manager.run(payload)

        last = None
        prior: list[dict[str, Any]] = list(payload.get("prior_executions") or [])
        current = str(payload.get("current_state") or self.entry_id)
        initial_state = current
        snapshot = dict(payload.get("state") or {})
        run_input = dict(payload.get("input") or {})
        all_steps: list[Any] = []
        all_transitions: list[str] = []
        all_gates: list[Any] = []
        step_offset = 0

        for _ in range(max_cycles):
            cycle_request = {
                **payload,
                "input": run_input,
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
    entry: Any = None,
) -> FSM:
    """Build a linear FSM from an ordered node list.

    Developers pass the happy-path sequence; ``Workflow`` sets ``entry`` to
    the first step, wires ``entry → … → End``, and the optional fallback.
    The entry node's ``input_schema`` is the workflow contract.

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
    if isinstance(fallback, CombineNode):
        raise FSMValidationError(
            ["fallback cannot be a CombineNode; use SchemaNode(..., is_fallback=True)"]
        )
    if isinstance(fallback, Node) and not fallback.is_fallback:
        fallback = fallback.model_copy(update={"is_fallback": True})

    first = steps[0]
    edges: list[Edge] = []
    for index in range(len(steps) - 1):
        edges.append(
            edge_deterministic(_exit_id(steps[index]), _entry_id(steps[index + 1]))
        )
    edges.append(edge_deterministic(_exit_id(steps[-1]), END))
    if entry is None:
        first = steps[0]
        edges.append(edge_fallback(_entry_id(first), fallback.id))
        entry_target = first
    else:
        edges.append(edge_fallback(_entry_id(entry), fallback.id))
        entry_target = entry

    is_router = hasattr(entry_target, "routes") # Simple check for router
    return FSM(
        nodes=[*steps, fallback],
        routers=[entry_target] if is_router else [],
        entry=entry_target,
        edges=edges,
    )


# Backward-compatible aliases
Graph = FSM
GraphValidationError = FSMValidationError
