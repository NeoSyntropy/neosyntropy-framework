"""Validation node factories for the node level of the FSM hierarchy.

Both factories produce nodes whose ``output_schema`` is always
:class:`~neosyntropy.core.node.schemas.ValidationResult` so FSM edges can
branch on the single well-known ``state["valid"]`` key.

Typical workflow pattern::

    # 1. Run a schema / reasoning node that produces some artefact.
    # 2. Run a validation node — semantic or functional.
    # 3. Branch:
    #      valid == True  → continue to the next step
    #      valid == False → retry / raise to a fallback

See :func:`SemanticValidationNode` for LLM-backed judgement and
:func:`functional_validation_node` for developer-written Python logic.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from ..node._utils import _resolve_reasoning_level
from ..node.base import Node
from ..node.combine import CombineNode
from ..node.schema import SchemaNode
from ..node.schemas import ReasoningLevel, ValidationResult


def SemanticValidationNode(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    prompt: str,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    group: str | None = None,
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
    reasoning: ReasoningLevel = "low",
    tools: Sequence[str] = (),
) -> Node | CombineNode:
    """LLM-backed semantic validation: constrained JSON, optional tools.

    ``reasoning="low"`` (default) returns a single :func:`SchemaNode` whose
    ``output_schema`` is always :class:`ValidationResult`.

    ``reasoning="high"`` concatenates a :class:`~neosyntropy.core.node.combine.CombineNode`
    — a reasoning half that may call *tools*, then a schema half that extracts
    ``{"valid": bool, "reason": str}``.  Passing any *tools* automatically
    upgrades to ``high``.

    Wire a deterministic edge that branches on ``state["valid"]``
    (``True`` → proceed, ``False`` → retry / fallback).  For the high path
    leave from ``{id}.Schema``.

    Example::

        guard = SemanticValidationNode(
            "sql_safety_check",
            input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
            prompt=(
                "You are a SQL safety reviewer. "
                "Return valid=false and a reason if the SQL contains DROP, "
                "DELETE without a WHERE clause, or TRUNCATE."
            ),
        )

        # High reasoning with tools:
        deep_guard = SemanticValidationNode(
            "claim_check",
            input_schema=ClaimInput,
            prompt="Use fetch_order, then decide if the refund claim is valid.",
            tools=("fetch_order",),
        )

    Args:
        id:            Unique node id.
        input_schema:  Pydantic model class or JSON Schema dict describing the
                       workflow state the LLM may inspect.
        prompt:        Validation criteria fed to the LLM.  Be explicit about
                       what constitutes a pass (``valid=true``) vs. a fail.
        name:          Optional human-readable display name.
        description:   Optional longer description for observability.
        prerequisites: Node ids that must have succeeded before this node runs.
        group:         Optional group this node belongs to.
        metadata:      Arbitrary key-value pairs attached to the node.
        provider:      Provider id used for inference.
        reasoning:     ``"low"`` (schema only) or ``"high"`` (reason then
                       extract).  Tools force ``"high"``.
        tools:         Tool names for the reasoning half.  Non-empty implies
                       ``reasoning="high"``.
    """
    if not prompt:
        raise ValueError(
            f"SemanticValidationNode {id!r} requires a non-empty prompt "
            "describing what to validate"
        )
    level = _resolve_reasoning_level(
        reasoning, tools, owner=f"SemanticValidationNode {id!r}"
    )
    if level == "low":
        return SchemaNode(
            id=id,
            input_schema=input_schema,
            output_schema=ValidationResult,
            prompt=prompt,
            name=name,
            description=description,
            prerequisites=prerequisites,
            group=group,
            metadata=metadata,
            provider=provider,
        )
    return CombineNode(
        id=id,
        input_schema=input_schema,
        tools=tuple(tools),
        output_schema=ValidationResult,
        prompt=prompt,
        name=name,
        description=description,
        prerequisites=prerequisites,
        group=group,
        metadata=metadata,
        provider=provider,
        schema_prompt=(
            "Extract a ValidationResult JSON object "
            '{"valid": bool, "reason": str} from the prior reasoning notes.'
        ),
    )


def functional_validation_node(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "valid",
    prerequisites: tuple[str, ...] | list[str] = (),
    group: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "bool | ValidationResult"]], Node]:
    """Declare a developer-authored validation handler node.

    Decorate a function that receives a :class:`~neosyntropy.core.node.context.NodeContext`
    and returns either a plain ``bool`` or a full :class:`ValidationResult`.  The
    framework normalises ``bool`` returns and writes the result to the workflow
    state so downstream edges can branch on it immediately.

    State writes after the node runs:

    - ``state[output_key]``              — ``True`` / ``False``
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)

    The full :class:`ValidationResult` dict is also stored in
    ``NodeResult.output`` for observability / audit purposes.

    Example::

        @functional_validation_node(
            id="length_check",
            input_schema=MyState,
            output_key="output_valid",
        )
        def length_check(ctx: NodeContext) -> bool:
            return len(ctx.state.get("output", "")) > 10

        # Async handlers are supported too:
        @functional_validation_node(id="async_check", input_schema=MyState)
        async def async_check(ctx: NodeContext) -> ValidationResult:
            result = await some_external_check(ctx.state["value"])
            return ValidationResult(valid=result.ok, reason=result.message)

    Args:
        id:           Unique node id (defaults to the decorated function name).
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the
                      function's docstring.
        input_schema: Pydantic model class or JSON Schema dict.  Defaults to
                      ``{"type": "object"}`` when omitted.
        output_key:   State key that receives the boolean result.  A companion
                      key ``<output_key>_reason`` receives the reason string.
                      Defaults to ``"valid"``.
        prerequisites: Node ids that must have succeeded before this node runs.
        group:        Optional group this node belongs to.
        metadata:     Arbitrary key-value pairs attached to the node.
    """

    def decorator(fn: Callable[..., "bool | ValidationResult"]) -> Node:
        import inspect as _inspect

        node_id = id or fn.__name__
        resolved_input_schema: type[BaseModel] | dict[str, Any] = (
            input_schema if input_schema is not None else {"type": "object"}
        )

        async def _handler(ctx: Any) -> Any:
            raw = fn(ctx)
            if _inspect.isawaitable(raw):
                raw = await raw

            if isinstance(raw, bool):
                vr = ValidationResult(valid=raw)
            elif isinstance(raw, ValidationResult):
                vr = raw
            else:
                raise TypeError(
                    f"functional_validation_node {node_id!r} handler must return "
                    f"bool or ValidationResult, got {type(raw).__name__}"
                )

            return ctx.result(
                output=vr.model_dump(),
                state_updates={
                    output_key: vr.valid,
                    f"{output_key}_reason": vr.reason,
                },
            )

        return Node(
            id=node_id,
            name=name or node_id,
            description=(_inspect.getdoc(fn) or description or "").strip(),
            provider="neosyntropy/base",
            prompt="",
            prerequisites=tuple(prerequisites),
            tools=(),
            mode="schema_extraction",
            kind="handler",
            input_schema=resolved_input_schema,
            output_schema=ValidationResult,
            group=group,
            is_fallback=False,
            metadata=metadata or {},
            handler=_handler,
        )

    return decorator
