"""ReasoningNode factory: provider-backed reasoning with optional tool calls.

A ``ReasoningNode`` runs the LLM in an open-ended reasoning mode where tool
calling is permitted.  Its structured output defaults to plain text unless
``output_schema`` is supplied.

For multi-step reasoning, pass a sequence of :class:`~schema.ReasoningStep`
objects; the factory then creates a sub-:class:`Workflow` instead of a single
:class:`Node`.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ._utils import _shared_kwargs
from .base import Node
from .schema import ReasoningStep, SchemaNode
from .schemas import REASONING_OUTPUT_SCHEMA


def ReasoningNode(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    tools: Sequence[str] = (),
    prompt: str = "",
    steps: Sequence[ReasoningStep] | None = None,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
) -> Any:
    """Provider-backed reasoning: tools allowed, optional structured output.

    When ``steps`` is provided the factory builds a multi-step sub-workflow
    (a :class:`~neosyntropy.core.graph.Workflow`) rather than a bare
    :class:`~base.Node`.  Each step may use up to ``ReasoningStep.max_tools``
    tools; the final step emits ``output_schema`` (or plain text when absent).

    Args:
        id:            Unique node id.  Sub-workflow step ids are
                       ``{id}_step_0``, ``{id}_step_1``, etc.
        input_schema:  Pydantic model class or JSON Schema dict.
        tools:         Tool names available (single-node path only).
        prompt:        Prompt text (single-node path; required unless ``steps``
                       is supplied).
        steps:         Ordered list of :class:`~schema.ReasoningStep` objects
                       for the multi-step path.
        name:          Optional human-readable display name.
        description:   Optional longer description.
        prerequisites: Node ids that must succeed first.
        group:         Optional group this node belongs to.
        is_fallback:   When ``True`` acts as the graph's safe stop.
        metadata:      Arbitrary key-value pairs for observability.
        provider:      Provider id used for inference.
        output_schema: Optional structured output schema for the final step.
    """
    if steps:
        from ..graph import Workflow

        nodes: list[Node] = []
        for i, step in enumerate(steps):
            node_id = f"{id}_step_{i}"
            nodes.append(
                Node(
                    **_shared_kwargs(
                        id=node_id,
                        name=name if i == 0 else node_id,
                        description=description if i == 0 else "",
                        prompt=step.instruction,
                        prerequisites=prerequisites if i == 0 else (),
                        group=group,
                        is_fallback=is_fallback,
                        metadata=metadata,
                        provider=provider,
                    ),
                    tools=tuple(step.tools),
                    mode="reasoning",
                    kind="reasoning",
                    input_schema=input_schema if i == 0 else {"type": "object"},
                    output_schema=(
                        {"type": "object"}
                        if i < len(steps) - 1
                        else (output_schema or REASONING_OUTPUT_SCHEMA)
                    ),
                    handler=None,
                )
            )

        fallback = SchemaNode(
            id=f"{id}_fallback",
            input_schema=input_schema,
            output_schema=output_schema or REASONING_OUTPUT_SCHEMA,
            prompt=f"Fallback logic for {id} reasoning steps",
            is_fallback=True,
        )
        return Workflow(nodes, fallback=fallback, entry=nodes[0])

    if not prompt:
        raise ValueError(f"ReasoningNode {id!r} requires prompt or steps")

    return Node(
        **_shared_kwargs(
            id=id,
            name=name,
            description=description,
            prompt=prompt,
            prerequisites=prerequisites,
            group=group,
            is_fallback=is_fallback,
            metadata=metadata,
            provider=provider,
        ),
        tools=tuple(tools),
        mode="reasoning",
        kind="reasoning",
        input_schema=input_schema,
        output_schema=output_schema or REASONING_OUTPUT_SCHEMA,
        handler=None,
    )
