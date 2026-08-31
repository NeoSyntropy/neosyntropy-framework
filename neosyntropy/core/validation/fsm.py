"""FSM-level path validation factories and helpers.

An FSM path validator is placed as the **last node before** ``End`` and can
inspect the full execution history of the run: every node that ran, its
output, and the accumulated workflow state.

:class:`FSMPathInfo` and :func:`extract_fsm_path` are the key helpers that
turn ``NodeContext.prior_executions`` into a structured, ordered view of the
path taken.

Typical authoring pattern::

    from neosyntropy.core.validation.fsm import (
        SemanticFSMPathValidator,
        functional_fsm_path_validator,
        extract_fsm_path,
    )

    # Semantic (LLM judge over the full path)
    final_check = SemanticFSMPathValidator(
        "final_path_check",
        input_schema=WorkflowState,
        prompt=(
            "Review the complete execution path and accumulated state. "
            "Return valid=false if any required step was skipped or the "
            "final output is inconsistent with the original request."
        ),
    )

    # Functional (Python handler with structured path info)
    @functional_fsm_path_validator(input_schema=WorkflowState)
    def final_path_check(ctx: NodeContext) -> ValidationResult:
        path = extract_fsm_path(ctx)
        required = {"VerifyIdentity", "ExtractClaim"}
        missing = required - set(path.nodes_executed)
        if missing:
            return ValidationResult(
                valid=False,
                reason=f"required steps not executed: {sorted(missing)}",
            )
        return ValidationResult(valid=True)

Both factories produce a standard node (``output_schema=ValidationResult``,
no group binding).  Wire a deterministic edge from the validator to ``End``
(``True``) or a fallback (``False``).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..node.schemas import ReasoningLevel, ValidationResult
from .node import SemanticValidationNode, functional_validation_node

if TYPE_CHECKING:
    from ..node.base import Node
    from ..node.combine import CombineNode
    from ..node.context import NodeContext


@dataclass
class FSMPathInfo:
    """Structured view of the execution path taken through the FSM so far.

    Built from ``NodeContext.prior_executions`` by :func:`extract_fsm_path`.
    Handlers decorated with :func:`functional_fsm_path_validator` typically
    call ``extract_fsm_path(ctx)`` as their first step.

    Attributes:
        nodes_executed: Ordered list of node ids that completed with status
                        ``"succeeded"`` before the current node ran.
        outputs:        Mapping of ``node_id → output`` for each succeeded node.
        state:          Snapshot of the accumulated workflow state at the time
                        the current node is invoked (read-only).
        input:          The original run input dict (read-only).
    """

    nodes_executed: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)


def extract_fsm_path(ctx: "NodeContext") -> FSMPathInfo:
    """Extract a structured :class:`FSMPathInfo` from a :class:`NodeContext`.

    Iterates ``ctx.prior_executions`` in order and collects every node that
    finished with status ``"succeeded"``.  Failed or fallback nodes are
    excluded from ``nodes_executed`` and ``outputs`` so handlers can detect
    gaps in the required path.

    Example::

        @functional_fsm_path_validator(input_schema=WorkflowState)
        def audit_path(ctx: NodeContext) -> ValidationResult:
            path = extract_fsm_path(ctx)
            if "VerifyIdentity" not in path.nodes_executed:
                return ValidationResult(valid=False, reason="identity not verified")
            return ValidationResult(valid=True)

    Args:
        ctx: The :class:`~neosyntropy.core.node.context.NodeContext` passed to
             the handler.

    Returns:
        A :class:`FSMPathInfo` with the ordered execution history.
    """
    nodes_executed: list[str] = []
    outputs: dict[str, Any] = {}
    for record in ctx.prior_executions:
        if record.status == "succeeded":
            nodes_executed.append(record.node_id)
            outputs[record.node_id] = record.output
    return FSMPathInfo(
        nodes_executed=nodes_executed,
        outputs=outputs,
        state=dict(ctx.state),
        input=dict(ctx.input),
    )


def SemanticFSMPathValidator(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    prompt: str,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
    reasoning: ReasoningLevel = "low",
    tools: Sequence[str] = (),
) -> "Node | CombineNode":
    """LLM-backed semantic gate on the entire FSM execution path.

    Intended to be placed as the **last node before** ``End`` in the FSM.
    The LLM receives the accumulated ``input_schema`` state (which carries all
    prior node outputs merged in) and evaluates whether the full path reached a
    valid conclusion.

    The node has no ``group`` binding and no ``after`` wiring — position it
    explicitly in the FSM's edge list.

    ``reasoning="low"`` (default) produces a single :class:`SchemaNode`.
    ``reasoning="high"`` (or non-empty *tools*) produces a
    :class:`~neosyntropy.core.node.combine.CombineNode` that reasons first,
    then extracts ``{"valid": bool, "reason": str}``.  The exit state for the
    high path is ``{id}.Schema``.

    Example::

        final_check = SemanticFSMPathValidator(
            "audit_path",
            input_schema=WorkflowState,
            prompt=(
                "Examine the accumulated state and confirm the workflow "
                "completed all required steps. Return valid=false with a "
                "reason if any required field is absent or inconsistent."
            ),
        )

        fsm = FSM(
            nodes=[...steps, final_check, fallback],
            entry=first_step,
            edges=[
                ...,
                edge_deterministic("LastStep", "audit_path"),
                edge_deterministic("audit_path", "End"),
                edge_fallback("audit_path", fallback.id),
            ],
        )

    Args:
        id:            Unique node id.
        input_schema:  Pydantic model class or JSON Schema dict the LLM
                       inspects.  Should cover the full accumulated state.
        prompt:        Validation criteria over the complete path / state.
        name:          Optional human-readable display name.
        description:   Optional longer description for observability.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:      Arbitrary key-value pairs attached to the node.
        provider:      Provider id used for inference.
        reasoning:     ``"low"`` (schema only) or ``"high"`` (reason then
                       extract).  Tools force ``"high"``.
        tools:         Tool names for the reasoning half.
    """
    return SemanticValidationNode(
        id=id,
        input_schema=input_schema,
        prompt=prompt,
        name=name,
        description=description,
        prerequisites=prerequisites,
        group=None,
        metadata=metadata,
        provider=provider,
        reasoning=reasoning,
        tools=tools,
    )


def functional_fsm_path_validator(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "valid",
    prerequisites: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "bool | ValidationResult"]], "Node"]:
    """Declare a developer-authored FSM path validation handler.

    Decorates a function that receives a
    :class:`~neosyntropy.core.node.context.NodeContext` and returns a ``bool``
    or :class:`ValidationResult`.  Use :func:`extract_fsm_path` inside the
    handler to get a structured view of the entire execution path.

    The node has no ``group`` binding — place it at the FSM's terminal
    position explicitly.

    State writes after the node runs:

    - ``state[output_key]``              — ``True`` / ``False``
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)

    Example::

        @functional_fsm_path_validator(input_schema=WorkflowState)
        def audit_path(ctx: NodeContext) -> ValidationResult:
            path = extract_fsm_path(ctx)
            required = {"VerifyIdentity", "ExtractClaim"}
            missing = required - set(path.nodes_executed)
            if missing:
                return ValidationResult(
                    valid=False,
                    reason=f"required steps not executed: {sorted(missing)}",
                )
            return ValidationResult(valid=True)

    Args:
        id:           Unique node id (defaults to the decorated function name).
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the docstring.
        input_schema: Pydantic model class or JSON Schema dict.
        output_key:   State key that receives the boolean result.  A companion
                      key ``<output_key>_reason`` receives the reason string.
                      Defaults to ``"valid"``.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:     Arbitrary key-value pairs attached to the node.
    """
    return functional_validation_node(
        id,
        name=name,
        description=description,
        input_schema=input_schema,
        output_key=output_key,
        prerequisites=prerequisites,
        group=None,
        metadata=metadata,
    )
