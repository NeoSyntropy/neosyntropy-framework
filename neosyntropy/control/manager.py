"""ControlManager: the whole control cycle as one object.

The pipeline the docs describe, end to end::

    input -> deterministic edge | semantic-edge candidates -> router proposal
          -> plan validation -> execution -> guards / transition checks
          -> one state commit -> audit record

Rules the manager guarantees:

- Proposal is not permission: the router only proposes; the validator and
  the transition table decide.
- Fail-closed gates run before commit; an illegal transition or failed
  guard rejects the step with no state change ("no billable transition").
- At most one state commit per plan step, applied atomically.
- Every cycle emits an :class:`AuditRecord` so reviews check a graph path,
  not a transcript.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from ..backend import (
    BackendClient,
    BackendProvider,
    Client,
)
from ..core.context import ContextBuilder, RunContext
from ..core.graph import END, FSM
from ..core.models import (
    AuditRecord,
    Candidate,
    ExecutionStepResult,
    GateCheck,
    NodeResult,
    RoutingPlan,
    RunRequest,
    RunResult,
    Topology,
)
from ..core.state import StateConflictError, StateManager
from ..observability import (
    BackendTelemetryReporter,
    RunObserver,
    best_effort_call,
    control_graph_manifest,
    graph_manifest,
)
from ..providers.base import Provider, ProviderRegistry
from ..routing.backend_route import BackendSemanticRouter
from ..routing.base import Router
from ..routing.preferred import PreferredPathRouter
from ..tools.calling import ParameterExtractor
from ..tools.registry import ToolNotAllowedError, ToolRegistry
from .executor import TopologyExecutor
from .logging import DecisionLogger
from .validator import PlanValidationError, PlanValidator

# Trained semantic-router wire contract: 9 actionable slots + fallback.
_MAX_ACTIONABLE_CANDIDATES = 9


def _resolve_backend(
    client: Client | BackendClient | None = None,
    *,
    backend: BackendClient | None = None,
) -> BackendClient | None:
    """Accept the public :class:`Client` or a low-level ``BackendClient``."""
    if client is not None and backend is not None:
        raise ValueError("pass client= or backend=, not both")
    if isinstance(client, Client):
        return client._as_backend()
    if isinstance(client, BackendClient):
        return client
    if client is not None:
        raise TypeError(
            f"client must be Client or BackendClient; got {type(client)!r}"
        )
    if backend is not None:
        return backend
    return BackendClient.from_env()


class ControlManager:
    def __init__(
        self,
        graph: FSM,
        *,
        client: Client | BackendClient | None = None,
        backend: BackendClient | None = None,
        router: Router | None = None,
        providers: ProviderRegistry | Mapping[str, Provider] | None = None,
        tools: ToolRegistry | None = None,
        validator: PlanValidator | None = None,
        executor: TopologyExecutor | None = None,
        extractor: ParameterExtractor | None = None,
        context_builder: ContextBuilder | None = None,
        decision_logger: DecisionLogger | None = None,
        observer: RunObserver | None = None,
        telemetry_timeout: float | None = None,
        capture_payloads: bool = True,
    ):
        self.graph = graph
        resolved_backend = _resolve_backend(client, backend=backend)
        self._backend = resolved_backend
        if telemetry_timeout is None:
            # Prefer the client's telemetry budget so slow DBs don't create
            # orphan runs (run_started commits after the observer timed out).
            telemetry_timeout = (
                float(getattr(resolved_backend, "telemetry_timeout", 2.0) or 2.0)
                if resolved_backend is not None
                else 2.0
            )
        backend_providers: dict[str, Provider] = {}
        if resolved_backend is not None:
            backend_provider = BackendProvider(resolved_backend)
            backend_providers = {
                "neosyntropy/base": backend_provider,
            }
        self.providers = (
            providers
            if isinstance(providers, ProviderRegistry)
            else ProviderRegistry({**backend_providers, **dict(providers or {})})
        )
        if isinstance(providers, ProviderRegistry):
            for name, provider in backend_providers.items():
                self.providers.register(name, provider)
        self.tools = tools or ToolRegistry()
        # When a backend is configured, ControlManager uses the opaque
        # /control/runs API. Local router remains an offline fallback.
        self.router = router or (
            BackendSemanticRouter(resolved_backend)
            if resolved_backend is not None
            else PreferredPathRouter(graph)
        )
        self.validator = validator or PlanValidator()
        self.executor = executor or TopologyExecutor(
            self.providers, self.tools, extractor=extractor
        )
        self.context_builder = context_builder or ContextBuilder()
        self.decision_logger = decision_logger
        self.observer = (
            observer
            if observer is not None
            else (
                BackendTelemetryReporter(resolved_backend)
                if resolved_backend is not None
                else None
            )
        )
        if telemetry_timeout <= 0:
            raise ValueError("telemetry_timeout must be positive")
        self.telemetry_timeout = telemetry_timeout
        # When True, telemetry includes the run input plus each step's input
        # state and node outputs so developers can debug the FSM step by step
        # (and the data can later feed training). Set False for sanitized
        # lifecycle-only telemetry.
        self.capture_payloads = capture_payloads

    # -- public API ----------------------------------------------------------

    def run(self, request: RunRequest | Mapping[str, Any]) -> RunResult:
        """Run one control cycle synchronously."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(request))
        raise RuntimeError(
            "ControlManager.run() cannot be called from a running event loop; "
            "use `await ControlManager.arun(...)` instead"
        )

    async def arun(self, request: RunRequest | Mapping[str, Any]) -> RunResult:
        typed_request = (
            request
            if isinstance(request, RunRequest)
            else RunRequest.model_validate(request)
        )
        if not typed_request.current_state:
            typed_request = typed_request.model_copy(
                update={"current_state": self.graph.entry_id}
            )
        context = self.context_builder.build(typed_request)
        run_id = await self._observation_started(context)
        entry_check = self._entry_input_check(context)
        try:
            if entry_check is not None and not entry_check.passed:
                # The entry contract is the first gate: nothing is selected,
                # routed, or executed on input the graph never accepted.
                result = self._result(
                    context,
                    None,
                    [],
                    [],
                    context.current_state,
                    dict(context.state),
                    [entry_check],
                    completed=False,
                    rejection=entry_check.message,
                    committed=[],
                )
            else:
                # A satisfied entry gate still belongs in the audit trail.
                opening = [entry_check] if entry_check is not None else None
                result = (
                    await self._run_remote_control(context, run_id, opening)
                    if self._backend is not None
                    else await self._run_control_cycle(context, run_id, opening)
                )
        except Exception as exc:
            await self._observe(
                run_id, "run_failed", {"error_type": type(exc).__name__}
            )
            await self._observation_finished(
                run_id, status="failed", final_state=context.current_state
            )
            raise

        if result.rejected:
            await self._observe(run_id, "run_rejected", {})
            status = "rejected"
        elif result.completed:
            status = "completed"
        else:
            await self._observe(run_id, "run_failed", {})
            status = "failed"
        warm_blobs = [result.rejection or ""]
        for step in result.steps:
            for item in step.results:
                if item.error:
                    warm_blobs.append(item.error)
        if any(
            "warming" in blob.lower() or "still loading the gpu model" in blob.lower()
            for blob in warm_blobs
        ):
            print(
                "neosyntropy: inference is still warming up "
                "(GPU cold start, often 1–2 minutes). Retry shortly.",
                file=sys.stderr,
            )
        output: dict[str, Any] | None = None
        if self.capture_payloads:
            output = {
                "state": dict(result.state),
                "committed_transitions": list(result.audit.committed_transitions),
            }
            if result.rejection:
                output["rejection"] = result.rejection
        await self._observation_finished(
            run_id, status=status, final_state=result.final_state, output=output
        )
        return result

    def _apply_local_router_hops(
        self, context: RunContext
    ) -> tuple[RunContext, list[tuple[str, str]]]:
        """Follow unique guard-matching hops to routers/End before remote calls.

        DeterministicRouter guards are not on the wire; resolve them locally so
        the backend receives a concrete current_state. Returns the hops taken so
        telemetry can show router stations on the run path.

        Never hop off an actionable node that has not succeeded yet — entry
        nodes with ``→ End`` must still execute before the terminal transition.
        """
        state = context.current_state
        hops: list[tuple[str, str]] = []
        already_ran = {
            item.node_id
            for item in context.prior_executions
            if item.status == "succeeded"
        }
        for _ in range(16):
            standing = self.graph.nodes.get(state)
            if (
                standing is not None
                and not standing.is_fallback
                and state not in already_ran
            ):
                break
            matching = self.graph.matching_deterministic(state, context.state)
            if len(matching) != 1:
                break
            target = matching[0].target
            if target == END or self.graph.is_router_state(target):
                hops.append((state, target))
                state = target
                if target == END:
                    break
                continue
            break
        if state == context.current_state:
            return context, []
        return context.model_copy(update={"current_state": state}), hops

    def _router_hop_payload(
        self,
        source: str,
        target: str,
        *,
        reason: str,
        context: RunContext,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Telemetry for a router hop, including the input the router evaluated."""
        payload: dict[str, Any] = {
            "source": source,
            "target": target,
            "reason": reason,
            "router_hop": True,
        }
        if self.capture_payloads:
            payload["input"] = {
                "input": dict(context.input),
                "current_state": source,
                "state": dict(state if state is not None else context.state),
            }
        return payload

    async def _observe_transitions(
        self,
        run_id: str | None,
        transitions: list[tuple[str, str]],
        *,
        reason: str,
        context: RunContext,
    ) -> None:
        """Emit transition_committed for local router hops (not node executions)."""
        for source, target in transitions:
            await self._observe(
                run_id,
                "transition_committed",
                self._router_hop_payload(
                    source, target, reason=reason, context=context
                ),
            )

    async def _run_remote_control(
        self,
        context: RunContext,
        observation_run_id: str | None,
        initial_checks: list[GateCheck] | None = None,
    ) -> RunResult:
        """Backend owns select/route/validate/commit; client only executes handlers."""
        assert self._backend is not None
        context, local_hops = self._apply_local_router_hops(context)
        await self._observe_transitions(
            observation_run_id,
            local_hops,
            reason="deterministic_router",
            context=context,
        )
        if context.current_state == END:
            checks = list(initial_checks or [])
            return self._result(
                context,
                None,
                [],
                [],
                END,
                dict(context.state),
                checks,
                completed=True,
                rejection=None,
                committed=[f"{source}->{target}" for source, target in local_hops],
            )
        # Multi-rule DeterministicRouter: evaluate guards locally, then short-
        # circuit through the offline cycle (backend cannot see guards).
        if self.graph.is_router_state(context.current_state):
            matching = self.graph.matching_deterministic(
                context.current_state, context.state
            )
            if len(matching) == 1 and matching[0].target in self.graph.nodes:
                return await self._run_control_cycle(
                    context, observation_run_id, initial_checks
                )

        request_payload = {
            "input": dict(context.input),
            "request_id": context.request_id,
            "current_state": context.current_state,
            "history": [
                {"role": message.role, "content": message.content}
                for message in context.history
            ],
            "prior_executions": [
                record.model_dump(mode="json") for record in context.prior_executions
            ],
            "state": dict(context.state),
            "metadata": dict(context.metadata),
        }
        view = await self._backend.start_control_run(
            control_graph_manifest(self.graph, self.tools),
            request_payload,
        )
        await self._observe(
            observation_run_id,
            "plan_proposed",
            {"mode": "backend_owned"},
        )
        steps: list[ExecutionStepResult] = []
        checks: list[GateCheck] = list(initial_checks or [])
        observed_transitions = 0
        while view.get("status") == "awaiting_execution":
            step = view.get("step") or {}
            node_ids = list(step.get("nodes") or [])
            step_number = int(step.get("step", len(steps)))
            step_context = context.model_copy(
                update={
                    "current_state": view.get("current_state", context.current_state),
                    "state": dict(view.get("state") or {}),
                }
            )
            step_payload = self._step_payload(step_number, node_ids, step_context)
            await self._observe(observation_run_id, "step_started", step_payload)
            guard_check = self._remote_guard_check(node_ids, step_context)
            checks.append(guard_check)
            if not guard_check.passed:
                rejection = guard_check.message or "edge guard denied"
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=rejection,
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection
                    ),
                )
                break
            candidates = [
                Candidate(
                    node_id=node_id,
                    name=self.graph.nodes[node_id].name,
                    description=self.graph.nodes[node_id].description,
                    prerequisites=self.graph.nodes[node_id].prerequisites,
                    is_fallback=self.graph.nodes[node_id].is_fallback,
                )
                for node_id in node_ids
            ]
            indices = list(range(len(candidates)))
            try:
                results = await self.executor.execute_step(
                    indices, candidates, self.graph, step_context
                )
            except ToolNotAllowedError as exc:
                checks.append(
                    GateCheck(
                        name="ToolAllowList",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=str(exc),
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=str(exc)
                    ),
                )
                break

            try:
                preview_state, _ = StateManager(step_context).preview(results)
            except StateConflictError as exc:
                checks.append(
                    GateCheck(
                        name="StateConflict",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=str(exc),
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=str(exc), results=results
                    ),
                )
                break

            result_checks = self._result_stage_checks(
                step_context,
                results,
                preview_state,
                step_context.current_state,
                skip_backend_only=True,
            )
            checks.extend(result_checks)
            failed = [check for check in result_checks if not check.passed]
            if failed:
                rejection = failed[0].message or f"gate '{failed[0].name}' failed"
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=rejection,
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection, results=results
                    ),
                )
                break

            steps.append(ExecutionStepResult(step=step_number, results=results))
            view = await self._backend.submit_control_results(
                str(view["run_id"]),
                results=[_wire_node_result(result) for result in results],
            )
            transitions = list(view.get("committed_transitions") or [])
            for transition in transitions[observed_transitions:]:
                source, separator, target = str(transition).partition("->")
                if separator:
                    source = source.strip()
                    target = target.strip()
                    router_hop = self.graph.is_router_state(source)
                    if router_hop:
                        hop_payload = self._router_hop_payload(
                            source,
                            target,
                            reason="semantic_router",
                            context=context,
                            state=dict(view.get("state") or step_context.state),
                        )
                        hop_payload["step"] = step_number
                        await self._observe(
                            observation_run_id,
                            "transition_committed",
                            hop_payload,
                        )
                    else:
                        await self._observe(
                            observation_run_id,
                            "transition_committed",
                            {
                                "step": step_number,
                                "source": source,
                                "target": target,
                                "reason": "transition",
                                "router_hop": False,
                            },
                        )
            observed_transitions = len(transitions)
            await self._observe(
                observation_run_id,
                "step_completed",
                self._step_completed_payload(
                    step_payload,
                    (
                        "completed"
                        if view.get("status") in {"awaiting_execution", "completed"}
                        else str(view.get("status") or "failed")
                    ),
                    results=results,
                    state=dict(view.get("state") or {}),
                ),
            )

        final_state = str(view.get("current_state") or context.current_state)
        rejection = view.get("rejection")
        if isinstance(rejection, str) and rejection:
            checks.append(
                GateCheck(
                    name="BackendControl",
                    stage="result",
                    passed=False,
                    message=rejection,
                )
            )
        remote_committed = list(view.get("committed_transitions") or [])
        local_committed = [f"{source}->{target}" for source, target in local_hops]
        return self._result(
            context,
            None,
            [],
            steps,
            final_state,
            dict(view.get("state") or {}),
            checks,
            completed=bool(view.get("completed")),
            rejection=rejection if isinstance(rejection, str) else None,
            committed=[*local_committed, *remote_committed],
        )

    def _remote_guard_check(
        self, node_ids: list[str], context: RunContext
    ) -> GateCheck:
        frontier = {context.current_state}
        for node_id in node_ids:
            definition = self.graph.nodes[node_id]
            if definition.is_fallback:
                continue
            # Already standing on the planned node (entry / first visit).
            if node_id in frontier:
                continue
            allowed = any(
                self.graph.allows(source, node_id)
                and self.graph.guard_allows(source, node_id, context.state)
                for source in frontier
            )
            if not allowed:
                return GateCheck(
                    name="EdgeGuard",
                    stage="result",
                    passed=False,
                    node_id=node_id,
                    message=(
                        f"edge guard denied transition to {node_id!r} "
                        f"from {sorted(frontier)}"
                    ),
                )
        return GateCheck(name="EdgeGuard", stage="result", passed=True)

    async def _run_control_cycle(
        self,
        context: RunContext,
        run_id: str | None,
        initial_checks: list[GateCheck] | None = None,
    ) -> RunResult:
        # Router / End hop: exactly one deterministic edge to a non-executable
        # state (compiled router id or End) — commit the transition only.
        # Never skip an actionable node that has not run yet (entry).
        hop = self._deterministic_hop_target(context)
        if hop is not None:
            standing = self.graph.nodes.get(context.current_state)
            if standing is not None and not standing.is_fallback:
                already_ran = {
                    item.node_id
                    for item in context.prior_executions
                    if item.status == "succeeded"
                }
                if context.current_state not in already_ran:
                    hop = None
        if hop is not None:
            checks = list(initial_checks or [])
            checks.append(
                GateCheck(
                    name="RouterHop",
                    stage="result",
                    passed=True,
                    message=f"{context.current_state!r} -> {hop!r}",
                )
            )
            await self._observe(
                run_id,
                "transition_committed",
                self._router_hop_payload(
                    context.current_state,
                    hop,
                    reason="router_hop",
                    context=context,
                ),
            )
            return self._result(
                context,
                None,
                [],
                [],
                hop,
                dict(context.state),
                checks,
                completed=(hop == END),
                rejection=None,
                committed=[f"{context.current_state}->{hop}"],
            )

        # Standing on an executable node that has not run yet (typical for
        # entry): execute that node first. Outgoing edges apply on later cycles.
        short_circuit = self._current_node_plan(context)
        if short_circuit is None:
            # Deterministic short-circuit: exactly one matching edge commits
            # without the semantic router.
            short_circuit = self._deterministic_plan(context)
        if short_circuit is not None:
            candidates, plan = short_circuit
        elif self.graph.semantic_candidate_ids(context.current_state) is None:
            # No semantic edges from this state → fallback edge.
            candidates, plan = self._fallback_plan(context)
        else:
            # Candidates are the concrete targets of outgoing semantic edges
            # (plus the dedicated fallback). The semantic router chooses.
            candidates = self._semantic_candidates(context)
            actionable = [c for c in candidates if not c.is_fallback]
            if not actionable:
                candidates, plan = self._fallback_plan(context)
            else:
                plan = await self.router.route(context, candidates)
        if self.decision_logger is not None:
            self.decision_logger.log_router_decision(context, candidates, plan)
        await self._observe(
            run_id,
            "plan_proposed",
            {
                "topology": plan.topology.value,
                "steps": [
                    [candidates[index].node_id for index in indices]
                    for indices in plan.execution_plan
                ],
            },
        )

        checks: list[GateCheck] = list(initial_checks or [])

        # Gate 1: deterministic plan validation (topology, prerequisites, edges).
        try:
            self.validator.validate(plan, candidates, self.graph, context)
            checks.append(GateCheck(name="PlanValidator", stage="plan", passed=True))
        except PlanValidationError as exc:
            checks.append(
                GateCheck(
                    name="PlanValidator", stage="plan", passed=False, message=str(exc)
                )
            )
            return self._result(context, plan, candidates, [], context.current_state,
                                dict(context.state), checks, completed=False,
                                rejection=str(exc), committed=[])

        # Execution: run each step, gate its results, then commit at most one
        # state change — in that order, always.
        state_manager = StateManager(context)
        steps: list[ExecutionStepResult] = []
        committed: list[str] = []
        frontier: set[str] = {context.current_state}
        completed = True
        rejection: str | None = None

        # Mutable work queue so unique deterministic edges to the next node
        # (Investigate → ExtractTicket) append as further steps in this cycle.
        candidates = list(candidates)
        work: list[list[int]] = [list(step) for step in plan.execution_plan]
        step_number = 0
        while step_number < len(work):
            indices = work[step_number]
            step_targets = {candidates[index].node_id for index in indices}
            step_context = context.model_copy(
                update={
                    "current_state": state_manager.current_state,
                    "state": state_manager.snapshot(),
                }
            )
            step_payload = self._step_payload(
                step_number, sorted(step_targets), step_context
            )
            await self._observe(run_id, "step_started", step_payload)

            # Gate: edge guards run against the pre-step state, before the
            # side effect of executing the node (fail-closed).
            guard_check = self._step_guard_check(
                frontier, step_targets, candidates, indices, step_context.state
            )
            checks.append(guard_check)
            if not guard_check.passed:
                completed, rejection = False, guard_check.message
                await self._observe(
                    run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection
                    ),
                )
                break

            try:
                results = await self.executor.execute_step(
                    list(indices), candidates, self.graph, step_context
                )
            except ToolNotAllowedError as exc:
                checks.append(
                    GateCheck(
                        name="ToolAllowList",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                completed, rejection = False, str(exc)
                await self._observe(
                    run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection
                    ),
                )
                break

            try:
                preview_state, merged_next = state_manager.preview(results)
            except StateConflictError as exc:
                checks.append(
                    GateCheck(
                        name="StateConflict",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                completed, rejection = False, str(exc)
                await self._observe(
                    run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection, results=results
                    ),
                )
                break

            # Executor defaults next_state to the node itself. Still follow a
            # unique deterministic edge to End / a router. Edges to another
            # actionable node are chained as an extra step after commit (below).
            if len(step_targets) == 1:
                only = next(iter(step_targets))
                if merged_next in (None, only):
                    matching = self.graph.matching_deterministic(only, preview_state)
                    if len(matching) == 1:
                        auto_target = matching[0].target
                        if auto_target == END or self.graph.is_router_state(
                            auto_target
                        ):
                            results = [
                                (
                                    result.model_copy(
                                        update={"next_state": auto_target}
                                    )
                                    if result.next_state in (None, only)
                                    else result
                                )
                                for result in results
                            ]
                            preview_state, merged_next = state_manager.preview(
                                results
                            )

            step_checks = self._result_stage_checks(
                context, results, preview_state, state_manager.current_state
            )
            # A proposed next state must be legal from the step's own nodes
            # (the plan-validated frontier) or from the current state.
            step_checks.append(
                self._transition_check(
                    frontier | step_targets, merged_next, state_manager, preview_state
                )
            )
            checks.extend(step_checks)
            step_failed = [check for check in step_checks if not check.passed]
            if step_failed:
                completed = False
                rejection = step_failed[0].message or f"gate '{step_failed[0].name}' failed"
                await self._observe(
                    run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload, "rejected", rejection=rejection, results=results
                    ),
                )
                break

            previous_state = state_manager.current_state
            await state_manager.apply_step(results)
            hop_targets = _attribution_hop_targets(
                previous_state,
                state_manager.current_state,
                results,
            )
            cursor = previous_state
            for target in hop_targets:
                transition = f"{cursor}->{target}"
                committed.append(transition)
                await self._observe(
                    run_id,
                    "transition_committed",
                    {
                        "step": step_number,
                        "source": cursor,
                        "target": target,
                    },
                )
                cursor = target
            steps.append(ExecutionStepResult(step=step_number, results=results))
            frontier = step_targets

            if any(result.status == "failed" for result in results):
                completed = False
                await self._observe(
                    run_id,
                    "step_completed",
                    self._step_completed_payload(
                        step_payload,
                        "failed",
                        results=results,
                        state=state_manager.snapshot(),
                    ),
                )
                break
            await self._observe(
                run_id,
                "step_completed",
                self._step_completed_payload(
                    step_payload,
                    "completed",
                    results=results,
                    state=state_manager.snapshot(),
                ),
            )
            step_number += 1
            if step_number < len(work):
                continue
            chained = self._chain_deterministic_node_step(
                state_manager.current_state,
                state_manager.snapshot(),
                candidates,
                {
                    item.node_id
                    for item in context.prior_executions
                    if item.status == "succeeded"
                }
                | {
                    result.node_id
                    for step in steps
                    for result in step.results
                    if result.status == "succeeded"
                },
            )
            if chained is not None:
                candidates, next_indices = chained
                work.append(next_indices)

        return self._result(
            context,
            plan,
            candidates,
            steps,
            state_manager.current_state,
            state_manager.snapshot(),
            checks,
            completed=completed,
            rejection=rejection,
            committed=committed,
        )

    # -- observability -------------------------------------------------------

    def _run_input(self, context: RunContext) -> dict[str, Any] | None:
        """Debug payload describing what the whole run received."""
        if not self.capture_payloads:
            return None
        return {
            "input": dict(context.input),
            "current_state": context.current_state,
            "history": [
                {"role": message.role, "content": message.content}
                for message in context.history
            ],
            "state": dict(context.state),
            "metadata": dict(context.metadata),
        }

    def _step_payload(
        self, step_number: int, node_ids: Iterable[str], step_context: RunContext
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": step_number,
            "node_ids": sorted(node_ids),
        }
        if self.capture_payloads:
            payload["input"] = {
                "current_state": step_context.current_state,
                "state": dict(step_context.state),
            }
        return payload

    def _step_completed_payload(
        self,
        step_payload: dict[str, Any],
        status: str,
        *,
        results: list[NodeResult] | None = None,
        state: dict[str, Any] | None = None,
        rejection: str | None = None,
    ) -> dict[str, Any]:
        payload = {**step_payload, "status": status}
        if not self.capture_payloads:
            return payload
        if results is not None:
            output: dict[str, Any] = {
                "results": [_wire_node_result(result) for result in results]
            }
            if state is not None:
                output["state"] = dict(state)
            payload["output"] = output
        if rejection:
            payload["rejection"] = rejection
        return payload

    async def _observation_started(self, context: RunContext) -> str | None:
        if self.observer is None:
            return None
        try:
            operation = self.observer.run_started(
                request_id=context.request_id,
                initial_state=context.current_state,
                manifest=graph_manifest(self.graph, self.tools),
                input=self._run_input(context),
            )
        except Exception:
            return None
        run_id = await best_effort_call(operation, timeout=self.telemetry_timeout)
        return str(run_id) if run_id else None

    async def _observe(
        self, run_id: str | None, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        if self.observer is None or run_id is None:
            return
        try:
            operation = self.observer.event(run_id, event_type, payload)
        except Exception:
            return
        await best_effort_call(operation, timeout=self.telemetry_timeout)

    async def _observation_finished(
        self,
        run_id: str | None,
        *,
        status: str,
        final_state: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        if self.observer is None or run_id is None:
            return
        try:
            operation = self.observer.run_finished(
                run_id, status=status, final_state=final_state, output=output
            )
        except Exception:
            return
        await best_effort_call(operation, timeout=self.telemetry_timeout)

    # -- gates ---------------------------------------------------------------

    def _result_stage_checks(
        self,
        context: RunContext,
        results: list[NodeResult],
        preview_state: dict[str, Any],
        current_state: str,
        *,
        skip_backend_only: bool = False,
    ) -> list[GateCheck]:
        return []

    def _entry_input_check(self, context: RunContext) -> GateCheck | None:
        """Built-in entry gate: run input must match entry input_schema."""
        if context.current_state != self.graph.entry_id:
            return None
        message = self.graph.entry_input_error(context.input)
        return GateCheck(
            name="InputSchema",
            stage="plan",
            passed=message is None,
            message=message or "",
        )

    def _step_guard_check(
        self,
        frontier: set[str],
        step_targets: set[str],
        candidates: list[Candidate],
        indices: list[int],
        state: dict[str, Any],
    ) -> GateCheck:
        """Built-in edge-guard gate, evaluated before a step executes.

        For every planned hop into this step, at least one permitting edge's
        guard must allow it against the pre-step state. The dedicated
        fallback is exempt (safe stop). Fail-closed: a raising guard denies.
        """
        for index in indices:
            candidate = candidates[index]
            if candidate.is_fallback:
                continue
            target = candidate.node_id
            # Already standing on the planned node (entry / resume) — no hop.
            if target in frontier:
                continue
            allowed = any(
                (
                    self.graph.allows(source, target)
                    or any(
                        self.graph.allows(source, r) and self.graph.allows(r, target)
                        for r in self.graph.router_ids
                    )
                )
                and self.graph.guard_allows(source, target, state)
                for source in frontier
            )
            if not allowed:
                return GateCheck(
                    name="EdgeGuard",
                    stage="result",
                    passed=False,
                    node_id=target,
                    message=(
                        f"edge guard denied transition to {target!r} "
                        f"from {sorted(frontier)}"
                    ),
                )
        return GateCheck(name="EdgeGuard", stage="result", passed=True)

    def _transition_check(
        self,
        sources: set[str],
        target: str | None,
        state_manager: StateManager,
        preview_state: dict[str, Any],
    ) -> GateCheck:
        """Built-in transition-legality + guard gate, evaluated at commit time.

        The plan validator already approved hops between planned nodes; this
        gate re-verifies the *actual* proposed target (handlers may deviate)
        and evaluates edge guards against the previewed state, fail-closed.
        """
        if target is None or target == state_manager.current_state:
            return GateCheck(name="TransitionLegality", stage="result", passed=True)
        if target in sources:
            # Moving onto a plan-validated step node's own state.
            return GateCheck(name="TransitionLegality", stage="result", passed=True)
        for source in sources | {state_manager.current_state}:
            if self.graph.allows(source, target) and self.graph.guard_allows(
                source, target, preview_state
            ):
                return GateCheck(name="TransitionLegality", stage="result", passed=True)
        return GateCheck(
            name="TransitionLegality",
            stage="result",
            passed=False,
            message=(
                f"no legal guard-allowed transition to {target!r} "
                f"from {sorted(sources)}"
            ),
        )

    # -- helpers -------------------------------------------------------------

    def _deterministic_hop_target(self, context: RunContext) -> str | None:
        """Target id when the only matching edge is a router state or End."""
        matching = self.graph.matching_deterministic(
            context.current_state, context.state
        )
        if len(matching) != 1:
            return None
        target_id = matching[0].target
        if target_id == END or self.graph.is_router_state(target_id):
            return target_id
        return None

    def _chain_deterministic_node_step(
        self,
        current_state: str,
        state: Mapping[str, Any],
        candidates: list[Candidate],
        already_ran: set[str],
    ) -> tuple[list[Candidate], list[int]] | None:
        """Append the unique deterministic successor node as the next step."""
        matching = self.graph.matching_deterministic(current_state, dict(state))
        if len(matching) != 1:
            return None
        target = matching[0].target
        if target == END or self.graph.is_router_state(target):
            return None
        definition = self.graph.nodes.get(target)
        if definition is None or definition.is_fallback:
            return None
        if target in already_ran:
            return None
        if not self.graph.guard_allows(current_state, target, dict(state)):
            return None
        idx = next(
            (i for i, item in enumerate(candidates) if item.node_id == target),
            None,
        )
        if idx is None:
            candidates.append(
                Candidate(
                    node_id=definition.id,
                    name=definition.name,
                    description=definition.description,
                    prerequisites=definition.prerequisites,
                    is_fallback=definition.is_fallback,
                )
            )
            idx = len(candidates) - 1
        return candidates, [idx]

    def _current_node_plan(
        self, context: RunContext
    ) -> tuple[list[Candidate], RoutingPlan] | None:
        """Execute the current node when it is actionable and not yet succeeded."""
        definition = self.graph.nodes.get(context.current_state)
        if definition is None or definition.is_fallback:
            return None
        already_ran = {
            item.node_id
            for item in context.prior_executions
            if item.status == "succeeded"
        }
        if context.current_state in already_ran:
            return None
        target = Candidate(
            node_id=definition.id,
            name=definition.name,
            description=definition.description,
            prerequisites=definition.prerequisites,
            is_fallback=definition.is_fallback,
        )
        fallback = self.graph.fallback_node
        candidates = [
            target,
            Candidate(
                node_id=fallback.id,
                name=fallback.name,
                description=fallback.description,
                prerequisites=fallback.prerequisites,
                is_fallback=True,
            ),
        ]
        plan = RoutingPlan(
            reasoning=f"Execute current node {definition.id!r}.",
            topology=Topology.SEQUENTIAL,
            execution_plan=[[0]],
        )
        return candidates, plan

    def _deterministic_plan(
        self, context: RunContext
    ) -> tuple[list[Candidate], RoutingPlan] | None:
        """Return a plan when exactly one deterministic edge matches a node."""
        matching = self.graph.matching_deterministic(
            context.current_state, context.state
        )
        if len(matching) != 1:
            return None
        target_id = matching[0].target
        if target_id not in self.graph.nodes:
            return None
        definition = self.graph.nodes[target_id]
        target = Candidate(
            node_id=definition.id,
            name=definition.name,
            description=definition.description,
            prerequisites=definition.prerequisites,
            is_fallback=definition.is_fallback,
        )
        fallback = self.graph.fallback_node
        candidates = [
            target,
            Candidate(
                node_id=fallback.id,
                name=fallback.name,
                description=fallback.description,
                prerequisites=fallback.prerequisites,
                is_fallback=True,
            ),
        ]
        plan = RoutingPlan(
            reasoning=(
                f"Deterministic edge {context.current_state!r} -> {target_id!r}."
            ),
            topology=Topology.SEQUENTIAL,
            execution_plan=[[0]],
        )
        return candidates, plan

    def _fallback_plan(
        self, context: RunContext
    ) -> tuple[list[Candidate], RoutingPlan]:
        """Build a fallback-topology plan from the fallback edge."""
        fallback_id = self.graph.fallback_target(context.current_state)
        fallback = self.graph.nodes.get(fallback_id, self.graph.fallback_node)
        candidates = [
            Candidate(
                node_id=fallback.id,
                name=fallback.name,
                description=fallback.description,
                prerequisites=fallback.prerequisites,
                is_fallback=True,
            )
        ]
        plan = RoutingPlan(
            reasoning=(
                f"No deterministic or semantic route from "
                f"{context.current_state!r}; using fallback edge to "
                f"{fallback.id!r}."
            ),
            topology=Topology.FALLBACK,
            execution_plan=[[0]],
        )
        return candidates, plan

    def _semantic_candidates(self, context: RunContext) -> list[Candidate]:
        """Build router candidates from outgoing semantic edge targets."""
        scoped = self.graph.semantic_candidate_ids(context.current_state) or set()
        actionable: list[Candidate] = []
        for item in self.graph.nodes.values():
            if item.is_fallback or item.id not in scoped:
                continue
            metadata: dict[str, Any] = {}
            if item.group:
                metadata["group"] = item.group
            actionable.append(
                Candidate(
                    node_id=item.id,
                    name=item.name,
                    description=item.description,
                    prerequisites=item.prerequisites,
                    is_fallback=False,
                    metadata=metadata,
                )
            )
            if len(actionable) >= _MAX_ACTIONABLE_CANDIDATES:
                break
        fallback = self.graph.fallback_node
        actionable.append(
            Candidate(
                node_id=fallback.id,
                name=fallback.name,
                description=fallback.description,
                prerequisites=fallback.prerequisites,
                is_fallback=True,
            )
        )
        return actionable

    def _result(
        self,
        context: RunContext,
        plan: RoutingPlan | None,
        candidates: list[Candidate],
        steps: list[ExecutionStepResult],
        final_state: str,
        state: dict[str, Any],
        checks: list[GateCheck],
        *,
        completed: bool,
        rejection: str | None,
        committed: list[str],
    ) -> RunResult:
        rejected = rejection is not None
        audit = AuditRecord(
            request_id=context.request_id,
            input=dict(context.input),
            initial_state=context.current_state,
            final_state=final_state,
            plan=plan,
            candidates=candidates,
            gate_checks=checks,
            steps=steps,
            committed_transitions=committed,
            rejected=rejected,
            rejection=rejection,
        )
        return RunResult(
            request_id=context.request_id,
            plan=plan,
            candidates=candidates,
            steps=steps,
            final_state=final_state,
            state=state,
            completed=completed and not rejected,
            rejected=rejected,
            rejection=rejection,
            audit=audit,
        )


def _attribution_hop_targets(
    previous: str,
    next_state: str | None,
    results: list[NodeResult],
) -> list[str]:
    """Ordered state targets to record for this step.

    Chained nodes often execute while ``previous`` is still the predecessor.
    When the result jumps straight to ``End`` (or another outbound state),
    land on the executed node first so transitions read
    ``predecessor → N → next`` instead of collapsing to ``predecessor → next``.
    """
    if not next_state or next_state == previous:
        return []
    succeeded = [item for item in results if item.status == "succeeded"]
    if len(succeeded) == 1:
        executed = succeeded[0].node_id
        if previous != executed and next_state != executed:
            return [executed, next_state]
    return [next_state]


def _wire_node_result(result: NodeResult) -> dict[str, Any]:
    """Serialize only the fields the control-run API accepts."""
    return {
        "node_id": result.node_id,
        "status": result.status,
        "output": result.output,
        "state_updates": result.state_updates,
        "next_state": result.next_state,
        "error": result.error,
    }
