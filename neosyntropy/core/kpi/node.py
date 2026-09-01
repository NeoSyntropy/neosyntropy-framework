"""KPI node factories for the node level of the FSM hierarchy.

Both factories produce nodes whose ``output_schema`` is always
:class:`~neosyntropy.core.node.schemas.KpiResult`.  Unlike validation nodes
KPI nodes **never** fail the run — they score it.

Typical usage::

    # 1. Run one or more schema / reasoning nodes that produce artefacts.
    # 2. Run a KPI node — semantic or functional.
    # 3. Continue unconditionally to the next step or End.
    #    (branch on score only if needed, via a downstream validation node)

See :func:`SemanticKpiNode` for LLM-backed scoring and
:func:`functional_kpi_node` for developer-written Python logic.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from ..node._utils import _resolve_reasoning_level
from ..node.base import Node
from ..node.combine import CombineNode
from ..node.schema import SchemaNode
from ..node.schemas import KpiResult, ReasoningLevel

_LAYER1_SCAFFOLD = (
    "You are a KPI scorer for an FSM run path.\n"
    "Respond ONLY with a JSON object: "
    '{"name": string, "score": number, "reason": string}.\n'
    "score is typically between 0.0 and 1.0 but may exceed that range.\n"
    "Do not decide whether the run should continue — that is not your job."
)


def SemanticKpiNode(
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
    """LLM-backed semantic KPI scorer.

    ``reasoning="low"`` (default) returns a single :func:`SchemaNode` whose
    ``output_schema`` is always :class:`KpiResult`.

    ``reasoning="high"`` concatenates a
    :class:`~neosyntropy.core.node.combine.CombineNode` — a reasoning half
    that may call *tools*, then a schema half that extracts
    ``{"name": str, "score": number, "reason": str}``.  Passing any *tools*
    automatically upgrades to ``high``.

    Unlike :func:`SemanticValidationNode`, this node does not write
    ``state["score"]`` automatically — the LLM output is stored in
    ``NodeResult.output`` for observability.  Use
    :func:`functional_kpi_node` when you need automatic state writes.

    Example::

        quality = SemanticKpiNode(
            "answer_quality",
            input_schema={"type": "object"},
            prompt=(
                "Score the quality of the generated answer on a scale of 0–1. "
                "name='answer_quality'. Return score=0 if the answer is empty."
            ),
        )

    Args:
        id:            Unique node id.
        input_schema:  Pydantic model class or JSON Schema dict.
        prompt:        Scoring criteria fed to the LLM.  Be explicit about
                       what constitutes a high vs. low score.
        name:          Optional human-readable display name.
        description:   Optional longer description for observability.
        prerequisites: Node ids that must have succeeded before this node runs.
        group:         Optional group this node belongs to.
        metadata:      Arbitrary key-value pairs attached to the node.
        provider:      Provider id used for inference.
        reasoning:     ``"low"`` (schema only) or ``"high"`` (reason then
                       extract).  Tools force ``"high"``.
        tools:         Tool names for the reasoning half.
    """
    if not prompt:
        raise ValueError(
            f"SemanticKpiNode {id!r} requires a non-empty prompt "
            "describing what to score"
        )
    full_prompt = f"{_LAYER1_SCAFFOLD}\n\n{prompt}"
    level = _resolve_reasoning_level(
        reasoning, tools, owner=f"SemanticKpiNode {id!r}"
    )
    if level == "low":
        return SchemaNode(
            id=id,
            input_schema=input_schema,
            output_schema=KpiResult,
            prompt=full_prompt,
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
        output_schema=KpiResult,
        prompt=full_prompt,
        name=name,
        description=description,
        prerequisites=prerequisites,
        group=group,
        metadata=metadata,
        provider=provider,
        schema_prompt=(
            "Extract a KpiResult JSON object "
            '{"name": str, "score": number, "reason": str} from the prior reasoning notes.'
        ),
    )


def functional_kpi_node(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_key: str = "score",
    prerequisites: tuple[str, ...] | list[str] = (),
    group: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., "float | KpiResult"]], Node]:
    """Declare a developer-authored KPI handler node.

    Decorate a function that receives a
    :class:`~neosyntropy.core.node.context.NodeContext` and returns either a
    plain ``float`` (promoted to ``KpiResult(name=output_key, score=raw)``)
    or a full :class:`KpiResult`.

    State writes after the node runs:

    - ``state[output_key]``              — the numeric score
    - ``state[output_key + "_reason"]``  — explanation string (may be empty)
    - ``state["kpis"]``                  — dict accumulating
                                           ``{kpi_name: score}`` across all
                                           KPI nodes in this run

    Example::

        @functional_kpi_node(
            id="completeness",
            input_schema=WorkflowState,
            output_key="completeness",
        )
        def completeness(ctx: NodeContext) -> KpiResult:
            required = {"StepA", "StepB", "StepC"}
            ran = set(ctx.state.get("ran_steps", []))
            hit = required & ran
            return KpiResult(
                name="completeness",
                score=round(len(hit) / len(required), 3),
                reason=f"ran {sorted(hit)} of {sorted(required)}",
            )

        # Async handlers are supported too:
        @functional_kpi_node(id="latency_score")
        async def latency_score(ctx: NodeContext) -> float:
            ms = ctx.state.get("elapsed_ms", 0)
            return max(0.0, 1.0 - ms / 5000)

    Args:
        id:           Unique node id (defaults to the decorated function name).
        name:         Optional human-readable display name.
        description:  Optional longer description; falls back to the
                      function's docstring.
        input_schema: Pydantic model class or JSON Schema dict.  Defaults to
                      ``{"type": "object"}`` when omitted.
        output_key:   State key that receives the numeric score.  A companion
                      key ``<output_key>_reason`` receives the reason string.
                      Defaults to ``"score"``.
        prerequisites: Node ids that must have succeeded before this node runs.
        group:        Optional group this node belongs to.
        metadata:     Arbitrary key-value pairs attached to the node.
    """

    def decorator(fn: Callable[..., "float | KpiResult"]) -> Node:
        import inspect as _inspect

        node_id = id or fn.__name__
        resolved_input_schema: type[BaseModel] | dict[str, Any] = (
            input_schema if input_schema is not None else {"type": "object"}
        )

        async def _handler(ctx: Any) -> Any:
            raw = fn(ctx)
            if _inspect.isawaitable(raw):
                raw = await raw

            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                vr = KpiResult(name=output_key, score=float(raw))
            elif isinstance(raw, KpiResult):
                vr = raw
            else:
                raise TypeError(
                    f"functional_kpi_node {node_id!r} handler must return "
                    f"float or KpiResult, got {type(raw).__name__}"
                )

            kpis = dict(ctx.state.get("kpis") or {})
            kpis[vr.name] = vr.score

            return ctx.result(
                output=vr.model_dump(),
                state_updates={
                    output_key: vr.score,
                    f"{output_key}_reason": vr.reason,
                    "kpis": kpis,
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
            output_schema=KpiResult,
            group=group,
            is_fallback=False,
            metadata=metadata or {},
            handler=_handler,
        )

    return decorator
