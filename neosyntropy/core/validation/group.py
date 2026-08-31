"""Group-level path validation factories.

A group path validator gates the **outcome of traversing a group's internal
nodes**.  Both factories produce a :class:`ValidationResult` node that is
automatically registered into the target group, and — when ``after=`` is
supplied — have the terminal deterministic edge wired for you.

Typical authoring pattern::

    billing_group = Group(name="billing", ...)

    # Semantic (LLM judge on the accumulated group state)
    billing_check = SemanticGroupPathValidator(
        "billing_path_check",
        group=billing_group,
        after="ProcessPayment",      # auto-wires edge ProcessPayment → validator
        input_schema=BillingState,
        prompt=(
            "Verify that the billing flow produced a confirmed payment. "
            "Return valid=false if payment_confirmed is missing or false."
        ),
    )

    # Functional (Python handler)
    @functional_group_path_validator(
        group=billing_group,
        after="ProcessPayment",
        input_schema=BillingState,
    )
    def billing_path_check(ctx: NodeContext) -> bool:
        return ctx.state.get("payment_confirmed", False)

Both factories:

1. Bind the validation node to the group (``node.group == group.name``).
2. Auto-register the node into the group via ``group.add_node()``.
3. When ``after=`` is provided, call ``group.add_edge(after, id)`` so the edge
   is compiled into the parent FSM automatically.

The resulting node's ``output_schema`` is always
:class:`~neosyntropy.core.node.schemas.ValidationResult` so edges can branch
on ``state["valid"]``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..node.schemas import ReasoningLevel, ValidationResult
from .node import SemanticValidationNode, functional_validation_node

if TYPE_CHECKING:
    from ..group import Group
    from ..node.base import Node
    from ..node.combine import CombineNode


def SemanticGroupPathValidator(
    id: str,
    *,
    group: "Group",
    input_schema: type[BaseModel] | dict[str, Any],
    prompt: str,
    after: str | None = None,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
    reasoning: ReasoningLevel = "low",
    tools: Sequence[str] = (),
) -> "Node | CombineNode":
    """LLM-backed semantic gate on the outcome of a group's execution path.

    Creates a :func:`~neosyntropy.core.validation.node.SemanticValidationNode`
    bound to *group*, registers it into the group, and (when *after* is given)
    wires the deterministic edge ``after → id`` inside the group.

    The node's ``output_schema`` is always :class:`ValidationResult`.  Wire a
    deterministic edge from this validator that branches on ``state["valid"]``
    (``True`` → proceed, ``False`` → retry / fallback).  For
    ``reasoning="high"`` the exit state is ``{id}.Schema``.

    Example::

        billing_check = SemanticGroupPathValidator(
            "billing_path_check",
            group=billing_group,
            after="ProcessPayment",
            input_schema=BillingState,
            prompt=(
                "Verify the billing flow produced a confirmed payment. "
                "Return valid=false if payment_confirmed is false."
            ),
        )

    Args:
        id:            Unique node id for the validator.
        group:         The :class:`~neosyntropy.core.group.Group` instance this
                       validator belongs to.  The node is registered into it.
        input_schema:  Pydantic model class or JSON Schema dict the LLM inspects.
        prompt:        Validation criteria.  Be explicit about pass vs. fail.
        after:         Id of the last node in the group path.  When provided,
                       a deterministic edge ``after → id`` is added to *group*.
        name:          Optional human-readable display name.
        description:   Optional longer description for observability.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:      Arbitrary key-value pairs attached to the node.
        provider:      Provider id used for inference.
        reasoning:     ``"low"`` (schema only) or ``"high"`` (reason then
                       extract).  Tools force ``"high"``.
        tools:         Tool names for the reasoning half.
    """
    node = SemanticValidationNode(
        id=id,
        input_schema=input_schema,
        prompt=prompt,
        name=name,
        description=description,
        prerequisites=prerequisites,
        group=group.name,
        metadata=metadata,
        provider=provider,
        reasoning=reasoning,
        tools=tools,
    )
    group.add_node(node)
    if after is not None:
        group.add_edge(after, id)
    return node


def functional_group_path_validator(
    id: str | None = None,
    *,
    group: "Group",
    after: str | None = None,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "valid",
    prerequisites: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "bool | ValidationResult"]], "Node"]:
    """Declare a developer-authored group path validation handler.

    Decorates a function that receives a
    :class:`~neosyntropy.core.node.context.NodeContext` and returns a ``bool``
    or :class:`ValidationResult`.  The node is automatically registered into
    *group* and the edge ``after → id`` is wired when *after* is provided.

    State writes after the node runs:

    - ``state[output_key]``              — ``True`` / ``False``
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)

    Example::

        @functional_group_path_validator(
            group=billing_group,
            after="ProcessPayment",
            input_schema=BillingState,
        )
        def billing_path_check(ctx: NodeContext) -> bool:
            return ctx.state.get("payment_confirmed", False)

    Args:
        id:           Unique node id (defaults to the decorated function name).
        group:        The :class:`~neosyntropy.core.group.Group` instance this
                      validator belongs to.  The node is registered into it.
        after:        Id of the last node in the group path.  When provided,
                      a deterministic edge ``after → id`` is added to *group*.
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the docstring.
        input_schema: Pydantic model class or JSON Schema dict.
        output_key:   State key that receives the boolean result.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:     Arbitrary key-value pairs attached to the node.
    """
    def decorator(fn: Callable[..., "bool | ValidationResult"]) -> "Node":
        node_id = id or fn.__name__
        inner = functional_validation_node(
            node_id,
            name=name,
            description=description,
            input_schema=input_schema,
            output_key=output_key,
            prerequisites=prerequisites,
            group=group.name,
            metadata=metadata,
        )(fn)
        group.add_node(inner)
        if after is not None:
            group.add_edge(after, node_id)
        return inner

    return decorator
