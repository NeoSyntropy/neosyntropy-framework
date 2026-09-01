"""FSM-level KPI factories and helpers.

An FSM path KPI node is placed as the **last node before** ``End`` and scores
the full execution history of the run.  Unlike FSM path *validation*, the
node always proceeds to ``End`` — it does not route to a fallback.

:class:`~neosyntropy.core.validation.fsm.FSMPathInfo` and
:func:`~neosyntropy.core.validation.fsm.extract_fsm_path` are imported from
the validation package.  KPI shares the same path helper so there is no
duplication.

Typical authoring pattern::

    from neosyntropy.core.kpi import (
        SemanticFSMPathKpi,
        functional_fsm_path_kpi,
        extract_fsm_path,
    )

    # Semantic (LLM scorer over the full path)
    quality = SemanticFSMPathKpi(
        "path_quality",
        input_schema=WorkflowState,
        prompt=(
            "Score the overall quality of the FSM run on a scale of 0–1. "
            "name='path_quality'. "
            "Consider whether all required steps ran and the final output "
            "is coherent with the original request."
        ),
    )

    # Functional (Python scorer with structured path info)
    @functional_fsm_path_kpi(input_schema=WorkflowState)
    def path_quality(ctx: NodeContext) -> KpiResult:
        path = extract_fsm_path(ctx)
        required = {"VerifyIdentity", "ExtractClaim"}
        coverage = len(required & set(path.nodes_executed)) / len(required)
        return KpiResult(
            name="path_quality",
            score=coverage,
            reason=f"covered {len(required & set(path.nodes_executed))} of {len(required)} steps",
        )

Wire the KPI node straight to ``End`` — no guard, no fallback edge required::

    edges=[
        ...,
        edge_deterministic("LastStep", "path_quality"),
        edge_deterministic("path_quality", "End"),   # always continues
    ]

If a score threshold must gate the run, place a
:func:`~neosyntropy.core.validation.node.functional_validation_node` after
the KPI node and branch on ``state["valid"]``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..node.schemas import KpiResult, ReasoningLevel

# Re-import the shared path helpers from validation — no duplication.
from ..validation.fsm import FSMPathInfo, extract_fsm_path  # noqa: F401
from .node import SemanticKpiNode, functional_kpi_node

if TYPE_CHECKING:
    from ..node.base import Node
    from ..node.combine import CombineNode
    from ..node.context import NodeContext


def SemanticFSMPathKpi(
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
    """LLM-backed semantic KPI scorer over the entire FSM execution path.

    Intended to be placed as the **last node before** ``End`` in the FSM.
    The LLM receives the accumulated ``input_schema`` state (which carries all
    prior node outputs merged in) and produces a :class:`KpiResult` score.

    The node has no ``group`` binding and no ``after`` wiring — position it
    explicitly in the FSM's edge list.

    ``reasoning="low"`` (default) produces a single :class:`SchemaNode`.
    ``reasoning="high"`` (or non-empty *tools*) produces a
    :class:`~neosyntropy.core.node.combine.CombineNode` that reasons first,
    then extracts ``{"name": str, "score": number, "reason": str}``.  The
    exit state for the high path is ``{id}.Schema``.

    Example::

        quality = SemanticFSMPathKpi(
            "path_quality",
            input_schema=WorkflowState,
            prompt=(
                "Score the overall quality of this FSM run on a scale of 0–1. "
                "name='path_quality'. "
                "Return score=0 if any key output field is missing."
            ),
        )

        fsm = FSM(
            nodes=[...steps, quality],
            entry=first_step,
            edges=[
                ...,
                edge_deterministic("LastStep", "path_quality"),
                edge_deterministic("path_quality", "End"),
            ],
        )

    Args:
        id:            Unique node id.
        input_schema:  Pydantic model class or JSON Schema dict the LLM
                       inspects.  Should cover the full accumulated state.
        prompt:        Scoring criteria over the complete path / state.
        name:          Optional human-readable display name.
        description:   Optional longer description for observability.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:      Arbitrary key-value pairs attached to the node.
        provider:      Provider id used for inference.
        reasoning:     ``"low"`` (schema only) or ``"high"`` (reason then
                       extract).  Tools force ``"high"``.
        tools:         Tool names for the reasoning half.
    """
    return SemanticKpiNode(
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


def functional_fsm_path_kpi(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "score",
    prerequisites: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "float | KpiResult"]], "Node"]:
    """Declare a developer-authored FSM path KPI handler.

    Decorates a function that receives a
    :class:`~neosyntropy.core.node.context.NodeContext` and returns a
    ``float`` or :class:`KpiResult`.  Use :func:`extract_fsm_path` inside
    the handler to get a structured view of the entire execution path.

    The node has no ``group`` binding — place it at the FSM's terminal
    position explicitly.

    State writes after the node runs:

    - ``state[output_key]``              — the numeric score
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)
    - ``state["kpis"]``                  — accumulated ``{name: score}`` dict

    Example::

        @functional_fsm_path_kpi(input_schema=WorkflowState)
        def path_quality(ctx: NodeContext) -> KpiResult:
            path = extract_fsm_path(ctx)
            required = {"VerifyIdentity", "ExtractClaim"}
            coverage = len(required & set(path.nodes_executed)) / len(required)
            return KpiResult(
                name="path_quality",
                score=coverage,
                reason=f"executed: {path.nodes_executed}",
            )

    Args:
        id:           Unique node id (defaults to the decorated function name).
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the docstring.
        input_schema: Pydantic model class or JSON Schema dict.
        output_key:   State key for the numeric score.  A companion key
                      ``<output_key>_reason`` receives the reason string.
                      Defaults to ``"score"``.
        prerequisites: Node ids that must have succeeded before this node runs.
        metadata:     Arbitrary key-value pairs attached to the node.
    """
    return functional_kpi_node(
        id,
        name=name,
        description=description,
        input_schema=input_schema,
        output_key=output_key,
        prerequisites=prerequisites,
        group=None,
        metadata=metadata,
    )
