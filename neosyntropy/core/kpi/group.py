"""Group-level KPI factories.

A group path KPI node scores the **outcome of traversing a group's internal
nodes**.  Both factories produce a :class:`KpiResult` node that is
automatically registered into the target group, and — when ``after=`` is
supplied — have the terminal deterministic edge wired for you.

Typical authoring pattern::

    billing_group = Group(name="billing", ...)

    # Semantic (LLM scorer on the accumulated group state)
    billing_score = SemanticGroupPathKpi(
        "billing_quality",
        group=billing_group,
        after="ProcessPayment",
        input_schema=BillingState,
        prompt=(
            "Score the quality of the billing flow on a scale of 0–1. "
            "name='billing_quality'. "
            "Return score=0 if payment_confirmed is false or missing."
        ),
    )

    # Functional (Python scorer — writes state automatically)
    @functional_group_path_kpi(
        group=billing_group,
        after="ProcessPayment",
        input_schema=BillingState,
        output_key="billing_quality",
    )
    def billing_quality(ctx: NodeContext) -> KpiResult:
        confirmed = ctx.state.get("payment_confirmed", False)
        attempts = ctx.state.get("payment_attempts", 1)
        score = 1.0 if confirmed else 0.0
        penalty = max(0.0, (attempts - 1) * 0.1)
        return KpiResult(
            name="billing_quality",
            score=round(score - penalty, 3),
            reason=f"confirmed={confirmed} attempts={attempts}",
        )

Both factories:

1. Bind the KPI node to the group (``node.group == group.name``).
2. Auto-register the node into the group via ``group.add_node()``.
3. When ``after=`` is provided, call ``group.add_edge(after, id)`` so the edge
   is compiled into the parent FSM automatically.

The node is wired to proceed unconditionally after scoring — no fallback edge
is needed.  If a score threshold must gate the run, place a
:func:`~neosyntropy.core.validation.node.functional_validation_node` after
this node and branch on ``state["valid"]``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..node.schemas import KpiResult, ReasoningLevel
from .node import SemanticKpiNode, functional_kpi_node

if TYPE_CHECKING:
    from ..group import Group
    from ..node.base import Node
    from ..node.combine import CombineNode


def SemanticGroupPathKpi(
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
    """LLM-backed semantic KPI scorer for a group's execution path.

    Creates a :func:`~neosyntropy.core.kpi.node.SemanticKpiNode` bound to
    *group*, registers it into the group, and (when *after* is given) wires
    the deterministic edge ``after → id`` inside the group.

    The node's ``output_schema`` is always :class:`KpiResult`.  Wire a
    deterministic edge from this KPI node that proceeds unconditionally
    (no branching on the score is required — compose with a
    :func:`~neosyntropy.core.validation.node.functional_validation_node`
    if a threshold gate is needed).

    Example::

        billing_score = SemanticGroupPathKpi(
            "billing_quality",
            group=billing_group,
            after="ProcessPayment",
            input_schema=BillingState,
            prompt=(
                "Score the billing flow quality on a scale of 0–1. "
                "name='billing_quality'."
            ),
        )

    Args:
        id:            Unique node id for the KPI node.
        group:         The :class:`~neosyntropy.core.group.Group` instance.
                       The node is registered into it automatically.
        input_schema:  Pydantic model class or JSON Schema dict the LLM
                       inspects.
        prompt:        Scoring criteria.  Be explicit about what constitutes
                       a high vs. low score.
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
    node = SemanticKpiNode(
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


def functional_group_path_kpi(
    id: str | None = None,
    *,
    group: "Group",
    after: str | None = None,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "score",
    prerequisites: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "float | KpiResult"]], "Node"]:
    """Declare a developer-authored group path KPI handler.

    Decorates a function that receives a
    :class:`~neosyntropy.core.node.context.NodeContext` and returns a
    ``float`` or :class:`KpiResult`.  The node is automatically registered
    into *group* and the edge ``after → id`` is wired when *after* is given.

    State writes after the node runs:

    - ``state[output_key]``              — the numeric score
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)
    - ``state["kpis"]``                  — accumulated ``{name: score}`` dict

    Example::

        @functional_group_path_kpi(
            group=billing_group,
            after="ProcessPayment",
            input_schema=BillingState,
            output_key="billing_quality",
        )
        def billing_quality(ctx: NodeContext) -> KpiResult:
            confirmed = ctx.state.get("payment_confirmed", False)
            return KpiResult(
                name="billing_quality",
                score=1.0 if confirmed else 0.0,
                reason=f"confirmed={confirmed}",
            )

    Args:
        id:           Unique node id (defaults to the decorated function name).
        group:        The :class:`~neosyntropy.core.group.Group` instance.
        after:        Id of the last node in the group path.  When provided,
                      a deterministic edge ``after → id`` is added to *group*.
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the docstring.
        input_schema: Pydantic model class or JSON Schema dict.
        output_key:   State key for the numeric score.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:     Arbitrary key-value pairs attached to the node.
    """
    def decorator(fn: Callable[..., "float | KpiResult"]) -> "Node":
        node_id = id or fn.__name__
        inner = functional_kpi_node(
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
