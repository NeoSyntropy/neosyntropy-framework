"""Axiom: a domain invariant that must hold before a controlled action commits.

One fail-closed engine covering the full axiom taxonomy — transition
legality, tool allow-lists, output schemas, and business-safety predicates —
wired into every control cycle at two points: before execution (plan stage)
and before commit (result stage).

A violated axiom rejects the step: no state commits, and the violation is
recorded in the audit trail ("broken axiom = no billable transition").
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

from .context import RunContext
from .models import AxiomCheck, NodeResult, RoutingPlan

Stage = Literal["plan", "result", "both"]


class AxiomViolation(Exception):
    """Raised when a business axiom is violated."""

    def __init__(self, message: str, *, axiom: str | None = None):
        super().__init__(message)
        self.axiom = axiom


class Proposal(BaseModel):
    """What an axiom judges: a proposed plan or a proposed node result.

    ``state`` is the state the workflow *would* have if the proposal were
    accepted (a preview at result stage, the current snapshot at plan stage).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["plan", "result"]
    state: dict[str, Any] = Field(default_factory=dict)
    current_state: str = "Start"
    next_state: str | None = None
    plan: RoutingPlan | None = None
    node_id: str | None = None
    result: NodeResult | None = None


class Axiom:
    """A named invariant. Predicate returns True when the proposal is legal.

    Scoping: empty ``states``/``nodes`` means global. ``states`` matches the
    workflow's current state; ``nodes`` matches the proposing node id.
    Predicate exceptions are violations — evaluation never fails open.
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[RunContext, Proposal], bool] | None = None,
        *,
        stage: Stage = "both",
        states: tuple[str, ...] | list[str] = (),
        nodes: tuple[str, ...] | list[str] = (),
        error_message: str = "",
    ):
        self.name = name
        self.predicate = predicate
        self.stage: Stage = stage
        self.states = tuple(states)
        self.nodes = tuple(nodes)
        self.error_message = error_message or f"Axiom '{name}' was violated."

    @property
    def scope(self) -> str:
        if self.nodes:
            return "node"
        if self.states:
            return "state"
        return "global"

    def applies(
        self, stage: Literal["plan", "result"], *, current_state: str, node_id: str | None
    ) -> bool:
        if self.stage != "both" and self.stage != stage:
            return False
        if self.states and current_state not in self.states:
            return False
        if self.nodes:
            if node_id is None:
                return False
            if node_id not in self.nodes:
                return False
        return True

    def evaluate(self, context: RunContext, proposal: Proposal) -> bool:
        if self.predicate is None:
            raise NotImplementedError(f"Axiom '{self.name}' has no predicate")
        return bool(self.predicate(context, proposal))

    def check(self, context: RunContext, proposal: Proposal) -> AxiomCheck:
        """Evaluate fail-closed and return an auditable check."""
        try:
            passed = self.evaluate(context, proposal)
            message = "" if passed else self.error_message
        except Exception as exc:  # fail closed: errors are violations
            passed = False
            message = f"{self.error_message} (evaluation error: {exc})"
        return AxiomCheck(
            name=self.name,
            stage=proposal.kind,
            passed=passed,
            message=message,
            node_id=proposal.node_id,
        )

    def enforce(self, context: RunContext, proposal: Proposal) -> None:
        result = self.check(context, proposal)
        if not result.passed:
            raise AxiomViolation(result.message, axiom=self.name)


class OutputAxiom(Axiom):
    """Control format: a node's output must match a JSON schema.

    Always result-stage.
    """

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        *,
        states: tuple[str, ...] | list[str] = (),
        nodes: tuple[str, ...] | list[str] = (),
        error_message: str = "",
    ):
        super().__init__(
            name,
            stage="result",
            states=states,
            nodes=nodes,
            error_message=error_message or "Output format does not match required schema.",
        )
        self.schema = schema

    def evaluate(self, context: RunContext, proposal: Proposal) -> bool:
        if proposal.result is None:
            return False
        try:
            jsonschema.validate(instance=proposal.result.output, schema=self.schema)
            return True
        except jsonschema.exceptions.ValidationError as exc:
            self.error_message = f"Schema validation failed: {exc.message}"
            return False


def axiom(
    name: str | None = None,
    *,
    stage: Stage = "both",
    states: tuple[str, ...] | list[str] = (),
    nodes: tuple[str, ...] | list[str] = (),
    error_message: str = "",
) -> Callable[[Callable[[RunContext, Proposal], bool]], Axiom]:
    """Declare an axiom from a predicate::

        @axiom(name="MarginFloor")
        def margin_floor(ctx, proposal) -> bool:
            return proposal.state.get("final_price", 0) >= ctx.state["cogs"] * 1.20

    The decorator returns the :class:`Axiom` itself, ready to attach to a
    graph or a :class:`~neosyntropy.control.manager.ControlManager`.
    """

    def decorator(fn: Callable[[RunContext, Proposal], bool]) -> Axiom:
        import inspect

        return Axiom(
            name=name or fn.__name__,
            predicate=fn,
            stage=stage,
            states=states,
            nodes=nodes,
            error_message=error_message or (inspect.getdoc(fn) or "").strip(),
        )

    return decorator


class AxiomEngine:
    """Evaluates every applicable axiom for a proposal; never fails open.

    The engine only *reports* checks; the control manager owns the decision
    to reject and guarantees no commit happens after a failed check.
    """

    def __init__(self, axioms: Iterable[Axiom] = ()):
        self.axioms: list[Axiom] = list(axioms)

    def add(self, axiom: Axiom) -> None:
        self.axioms.append(axiom)

    def evaluate(
        self,
        stage: Literal["plan", "result"],
        context: RunContext,
        proposal: Proposal,
        *,
        node_scoped_only: bool = False,
    ) -> list[AxiomCheck]:
        checks: list[AxiomCheck] = []
        for item in self.axioms:
            if node_scoped_only and not item.nodes:
                continue
            if not item.applies(
                stage, current_state=proposal.current_state, node_id=proposal.node_id
            ):
                continue
            checks.append(item.check(context, proposal))
        return checks
