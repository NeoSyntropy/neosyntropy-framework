"""ControlManager: the whole control cycle as one object.

The pipeline the docs describe, end to end::

    input -> candidate selection -> router proposal -> plan validation
          -> plan axioms -> execution -> result axioms -> one state commit
          -> audit record

Rules the manager guarantees:

- Proposal is not permission: the router only proposes; the validator, the
  axiom engine, and the transition table decide.
- Fail-closed gates run before commit; a broken axiom or illegal transition
  rejects the step with no state change ("no billable transition").
- At most one state commit per plan step, applied atomically.
- Every cycle emits an :class:`AuditRecord` so reviews check a graph path,
  not a transcript.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from ..backend import (
    BackendCandidateSelector,
    BackendClient,
    BackendProvider,
    BackendRouter,
)
from ..core.axiom import Axiom, AxiomEngine, AxiomViolation, Proposal
from ..core.context import ContextBuilder, RunContext
from ..core.graph import Graph
from ..core.models import (
    AuditRecord,
    AxiomCheck,
    Candidate,
    ExecutionStepResult,
    NodeResult,
    RoutingPlan,
    RunRequest,
    RunResult,
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
from ..routing.base import Router
from ..routing.deterministic import DeterministicRouter
from ..tools.calling import ParameterExtractor
from ..tools.registry import ToolRegistry
from .executor import TopologyExecutor
from .logging import DecisionLogger
from .selector import CandidateSelector, LexicalCandidateSelector
from .validator import PlanValidationError, PlanValidator


class ControlManager:
    def __init__(
        self,
        graph: Graph,
        *,
        backend: BackendClient | None = None,
        router: Router | None = None,
        providers: ProviderRegistry | Mapping[str, Provider] | None = None,
        tools: ToolRegistry | None = None,
        selector: CandidateSelector | None = None,
        validator: PlanValidator | None = None,
        executor: TopologyExecutor | None = None,
        extractor: ParameterExtractor | None = None,
        axioms: Iterable[Axiom] = (),
        context_builder: ContextBuilder | None = None,
        decision_logger: DecisionLogger | None = None,
        observer: RunObserver | None = None,
        telemetry_timeout: float = 2.0,
    ):
        self.graph = graph
        resolved_backend = backend if backend is not None else BackendClient.from_env()
        self._backend = resolved_backend
        backend_providers: dict[str, Provider] = {}
        if resolved_backend is not None:
            backend_provider = BackendProvider(resolved_backend)
            backend_providers = {
                "backend": backend_provider,
                # Compatibility for graphs authored before provider selection
                # moved behind the backend.
                "slm": backend_provider,
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
        # /control/runs API. Local router/selector remain offline fallbacks.
        self.router = router or (
            BackendRouter(resolved_backend)
            if resolved_backend is not None
            else DeterministicRouter(graph)
        )
        self.selector = selector or (
            BackendCandidateSelector(resolved_backend)
            if resolved_backend is not None
            else LexicalCandidateSelector()
        )
        self.validator = validator or PlanValidator()
        self.executor = executor or TopologyExecutor(
            self.providers, self.tools, extractor=extractor
        )
        self.context_builder = context_builder or ContextBuilder()
        self.axiom_engine = AxiomEngine([*graph.axioms, *axioms])
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
        context = self.context_builder.build(typed_request)
        run_id = await self._observation_started(context)
        try:
            result = (
                await self._run_remote_control(context, run_id)
                if self._backend is not None
                else await self._run_control_cycle(context, run_id)
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
        await self._observation_finished(
            run_id, status=status, final_state=result.final_state
        )
        return result

    async def _run_remote_control(
        self, context: RunContext, observation_run_id: str | None
    ) -> RunResult:
        """Backend owns select/route/validate/commit; client only executes handlers."""
        assert self._backend is not None
        request_payload = {
            "intent": context.intent,
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
            control_graph_manifest(self.graph),
            request_payload,
        )
        await self._observe(
            observation_run_id,
            "plan_proposed",
            {"mode": "backend_owned"},
        )
        steps: list[ExecutionStepResult] = []
        checks: list[AxiomCheck] = []
        observed_transitions = 0
        while view.get("status") == "awaiting_execution":
            step = view.get("step") or {}
            node_ids = list(step.get("nodes") or [])
            step_number = int(step.get("step", len(steps)))
            step_payload = {"step": step_number, "node_ids": sorted(node_ids)}
            await self._observe(observation_run_id, "step_started", step_payload)
            step_context = context.model_copy(
                update={
                    "current_state": view.get("current_state", context.current_state),
                    "state": dict(view.get("state") or {}),
                }
            )
            guard_check = self._remote_guard_check(node_ids, step_context)
            checks.append(guard_check)
            if not guard_check.passed:
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=guard_check.message or "edge guard denied",
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    {**step_payload, "status": "rejected"},
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
            except AxiomViolation as exc:
                checks.append(
                    AxiomCheck(
                        name=exc.axiom or "AxiomViolation",
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
                    {**step_payload, "status": "rejected"},
                )
                break

            try:
                preview_state, _ = StateManager(step_context).preview(results)
            except StateConflictError as exc:
                checks.append(
                    AxiomCheck(
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
                    {**step_payload, "status": "rejected"},
                )
                break

            result_checks = self._result_stage_checks(
                step_context,
                results,
                preview_state,
                step_context.current_state,
            )
            checks.extend(result_checks)
            failed = [check for check in result_checks if not check.passed]
            if failed:
                view = await self._backend.submit_control_results(
                    str(view["run_id"]),
                    client_rejection=failed[0].message
                    or f"axiom '{failed[0].name}' violated",
                )
                await self._observe(
                    observation_run_id,
                    "step_completed",
                    {**step_payload, "status": "rejected"},
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
                    await self._observe(
                        observation_run_id,
                        "transition_committed",
                        {
                            "step": step_number,
                            "source": source,
                            "target": target,
                        },
                    )
            observed_transitions = len(transitions)
            await self._observe(
                observation_run_id,
                "step_completed",
                {
                    **step_payload,
                    "status": (
                        "completed"
                        if view.get("status") in {"awaiting_execution", "completed"}
                        else str(view.get("status") or "failed")
                    ),
                },
            )

        final_state = str(view.get("current_state") or context.current_state)
        rejection = view.get("rejection")
        if isinstance(rejection, str) and rejection:
            checks.append(
                AxiomCheck(
                    name="BackendControl",
                    stage="result",
                    passed=False,
                    message=rejection,
                )
            )
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
            committed=list(view.get("committed_transitions") or []),
        )

    def _remote_guard_check(
        self, node_ids: list[str], context: RunContext
    ) -> AxiomCheck:
        frontier = {context.current_state}
        for node_id in node_ids:
            definition = self.graph.nodes[node_id]
            if definition.is_fallback:
                continue
            allowed = any(
                self.graph.allows(source, node_id)
                and self.graph.guard_allows(source, node_id, context.state)
                for source in frontier
            )
            if not allowed:
                return AxiomCheck(
                    name="EdgeGuard",
                    stage="result",
                    passed=False,
                    node_id=node_id,
                    message=(
                        f"edge guard denied transition to {node_id!r} "
                        f"from {sorted(frontier)}"
                    ),
                )
        return AxiomCheck(name="EdgeGuard", stage="result", passed=True)

    async def _run_control_cycle(
        self, context: RunContext, run_id: str | None
    ) -> RunResult:
        # Selection always runs; search is not permission.
        candidates = self.selector.select(context, self.graph)
        if inspect.isawaitable(candidates):
            candidates = await candidates
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

        checks: list[AxiomCheck] = []

        # Gate 1: deterministic plan validation (topology, prerequisites, edges).
        try:
            self.validator.validate(plan, candidates, self.graph, context)
            checks.append(AxiomCheck(name="PlanValidator", stage="plan", passed=True))
        except PlanValidationError as exc:
            checks.append(
                AxiomCheck(
                    name="PlanValidator", stage="plan", passed=False, message=str(exc)
                )
            )
            return self._result(context, plan, candidates, [], context.current_state,
                                dict(context.state), checks, completed=False,
                                rejection=str(exc), committed=[])

        # Gate 2: plan-stage axioms (global/state scoped, then per planned node).
        plan_proposal = Proposal(
            kind="plan",
            state=dict(context.state),
            current_state=context.current_state,
            plan=plan,
        )
        checks.extend(self.axiom_engine.evaluate("plan", context, plan_proposal))
        for node_id in self._planned_node_ids(plan, candidates):
            checks.extend(
                self.axiom_engine.evaluate(
                    "plan",
                    context,
                    plan_proposal.model_copy(update={"node_id": node_id}),
                    node_scoped_only=True,
                )
            )
        failed = [check for check in checks if not check.passed]
        if failed:
            rejection = failed[0].message or f"axiom '{failed[0].name}' violated"
            return self._result(context, plan, candidates, [], context.current_state,
                                dict(context.state), checks, completed=False,
                                rejection=rejection, committed=[])

        # Execution: run each step, gate its results, then commit at most one
        # state change — in that order, always.
        state_manager = StateManager(context)
        steps: list[ExecutionStepResult] = []
        committed: list[str] = []
        frontier: set[str] = {context.current_state}
        completed = True
        rejection: str | None = None

        for step_number, indices in enumerate(plan.execution_plan):
            step_targets = {candidates[index].node_id for index in indices}
            step_payload = {"step": step_number, "node_ids": sorted(step_targets)}
            await self._observe(run_id, "step_started", step_payload)
            step_context = context.model_copy(
                update={
                    "current_state": state_manager.current_state,
                    "state": state_manager.snapshot(),
                }
            )

            # Gate: edge guards run against the pre-step state, before the
            # side effect of executing the node (fail-closed).
            guard_check = self._step_guard_check(
                frontier, step_targets, candidates, indices, step_context.state
            )
            checks.append(guard_check)
            if not guard_check.passed:
                completed, rejection = False, guard_check.message
                await self._observe(
                    run_id, "step_completed", {**step_payload, "status": "rejected"}
                )
                break

            try:
                results = await self.executor.execute_step(
                    list(indices), candidates, self.graph, step_context
                )
            except AxiomViolation as exc:
                checks.append(
                    AxiomCheck(
                        name=exc.axiom or "AxiomViolation",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                completed, rejection = False, str(exc)
                await self._observe(
                    run_id, "step_completed", {**step_payload, "status": "rejected"}
                )
                break

            try:
                preview_state, merged_next = state_manager.preview(results)
            except StateConflictError as exc:
                checks.append(
                    AxiomCheck(
                        name="StateConflict",
                        stage="result",
                        passed=False,
                        message=str(exc),
                    )
                )
                completed, rejection = False, str(exc)
                await self._observe(
                    run_id, "step_completed", {**step_payload, "status": "rejected"}
                )
                break

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
                rejection = step_failed[0].message or f"axiom '{step_failed[0].name}' violated"
                await self._observe(
                    run_id, "step_completed", {**step_payload, "status": "rejected"}
                )
                break

            previous_state = state_manager.current_state
            await state_manager.apply_step(results)
            if state_manager.current_state != previous_state:
                transition = f"{previous_state}->{state_manager.current_state}"
                committed.append(transition)
                await self._observe(
                    run_id,
                    "transition_committed",
                    {
                        "step": step_number,
                        "source": previous_state,
                        "target": state_manager.current_state,
                    },
                )
            steps.append(ExecutionStepResult(step=step_number, results=results))
            frontier = step_targets

            if any(result.status == "failed" for result in results):
                completed = False
                await self._observe(
                    run_id, "step_completed", {**step_payload, "status": "failed"}
                )
                break
            await self._observe(
                run_id, "step_completed", {**step_payload, "status": "completed"}
            )

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

    async def _observation_started(self, context: RunContext) -> str | None:
        if self.observer is None:
            return None
        try:
            operation = self.observer.run_started(
                request_id=context.request_id,
                initial_state=context.current_state,
                manifest=graph_manifest(self.graph),
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
        self, run_id: str | None, *, status: str, final_state: str
    ) -> None:
        if self.observer is None or run_id is None:
            return
        try:
            operation = self.observer.run_finished(
                run_id, status=status, final_state=final_state
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
    ) -> list[AxiomCheck]:
        checks: list[AxiomCheck] = []
        for result in results:
            proposal = Proposal(
                kind="result",
                state=preview_state,
                current_state=current_state,
                next_state=result.next_state,
                node_id=result.node_id,
                result=result,
            )
            checks.extend(self.axiom_engine.evaluate("result", context, proposal))
        return checks

    def _step_guard_check(
        self,
        frontier: set[str],
        step_targets: set[str],
        candidates: list[Candidate],
        indices: list[int],
        state: dict[str, Any],
    ) -> AxiomCheck:
        """Built-in edge-guard axiom, evaluated before a step executes.

        For every planned hop into this step, at least one permitting edge's
        guard must allow it against the pre-step state. The dedicated
        fallback is exempt (safe stop). Fail-closed: a raising guard denies.
        """
        for index in indices:
            candidate = candidates[index]
            if candidate.is_fallback:
                continue
            target = candidate.node_id
            allowed = any(
                self.graph.allows(source, target)
                and self.graph.guard_allows(source, target, state)
                for source in frontier
            )
            if not allowed:
                return AxiomCheck(
                    name="EdgeGuard",
                    stage="result",
                    passed=False,
                    node_id=target,
                    message=(
                        f"edge guard denied transition to {target!r} "
                        f"from {sorted(frontier)}"
                    ),
                )
        return AxiomCheck(name="EdgeGuard", stage="result", passed=True)

    def _transition_check(
        self,
        sources: set[str],
        target: str | None,
        state_manager: StateManager,
        preview_state: dict[str, Any],
    ) -> AxiomCheck:
        """Built-in transition-legality + guard axiom, evaluated at commit time.

        The plan validator already approved hops between planned nodes; this
        gate re-verifies the *actual* proposed target (handlers may deviate)
        and evaluates edge guards against the previewed state, fail-closed.
        """
        if target is None or target == state_manager.current_state:
            return AxiomCheck(name="TransitionLegality", stage="result", passed=True)
        if target in sources:
            # Moving onto a plan-validated step node's own state.
            return AxiomCheck(name="TransitionLegality", stage="result", passed=True)
        for source in sources | {state_manager.current_state}:
            if self.graph.allows(source, target) and self.graph.guard_allows(
                source, target, preview_state
            ):
                return AxiomCheck(name="TransitionLegality", stage="result", passed=True)
        return AxiomCheck(
            name="TransitionLegality",
            stage="result",
            passed=False,
            message=(
                f"no legal guard-allowed transition to {target!r} "
                f"from {sorted(sources)}"
            ),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _planned_node_ids(plan: RoutingPlan, candidates: list[Candidate]) -> list[str]:
        return [
            candidates[index].node_id
            for step in plan.execution_plan
            for index in step
        ]

    def _result(
        self,
        context: RunContext,
        plan: RoutingPlan | None,
        candidates: list[Candidate],
        steps: list[ExecutionStepResult],
        final_state: str,
        state: dict[str, Any],
        checks: list[AxiomCheck],
        *,
        completed: bool,
        rejection: str | None,
        committed: list[str],
    ) -> RunResult:
        rejected = rejection is not None
        audit = AuditRecord(
            request_id=context.request_id,
            intent=context.intent,
            initial_state=context.current_state,
            final_state=final_state,
            plan=plan,
            candidates=candidates,
            axiom_checks=checks,
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
