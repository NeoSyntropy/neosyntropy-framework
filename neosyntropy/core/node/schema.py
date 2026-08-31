"""SchemaNode factory and step dataclasses for structured JSON extraction.

A ``SchemaNode`` uses constrained decoding to make the LLM emit a valid JSON
object matching ``output_schema`` in one shot — no tool calls, no reasoning
loop.  It is the cheapest and most deterministic node type.

For multi-step flows, pair :class:`ReasoningStep` instances with
:func:`~neosyntropy.core.node.reasoning.ReasoningNode`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ._utils import _is_model_type, _shared_kwargs
from .base import Node


@dataclass
class ReasoningStep:
    """One step inside a multi-step :func:`ReasoningNode`.

    Attributes:
        instruction: The prompt / instruction for this specific step.
        tools:       Tool names available in this step (max ``max_tools``).
        max_tools:   Hard cap on the number of tools (default 3).
    """

    instruction: str
    tools: Sequence[str] = ()
    max_tools: int = 3

    def __post_init__(self) -> None:
        if len(self.tools) > self.max_tools:
            raise ValueError(
                f"ReasoningStep cannot have more than {self.max_tools} tools."
            )


@dataclass
class SchemaStep:
    """The final parameter-extraction step of a multi-step workflow.

    Attributes:
        instruction: Optional prompt guiding the JSON extraction.
    """

    instruction: str | None = None


def SchemaNode(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    prompt: str,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    func: Callable[..., Any] | None = None,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
) -> Node:
    """Provider-backed schema extraction: constrained JSON, no tools.

    If ``func`` is provided the node uses the LLM to extract the function's
    parameters (inferred from its type hints) and then calls the function with
    the validated arguments.

    Args:
        id:            Unique node id.
        input_schema:  Pydantic model class or JSON Schema dict.
        prompt:        Instruction sent to the LLM for extraction.
        output_schema: Expected JSON shape.  Inferred from ``func`` when omitted.
        func:          Optional callable whose signature drives ``output_schema``
                       and is invoked after extraction.
        name:          Optional human-readable display name.
        description:   Optional longer description.
        prerequisites: Node ids that must succeed first.
        group:         Optional group this node belongs to.
        is_fallback:   When ``True`` this node acts as the graph's safe stop.
        metadata:      Arbitrary key-value pairs for observability.
        provider:      Provider id used for inference.
    """
    import inspect

    if not prompt:
        raise ValueError(f"SchemaNode {id!r} requires prompt")

    resolved_output_schema = output_schema

    if func is not None and resolved_output_schema is None:
        import typing

        from pydantic import create_model

        hints = typing.get_type_hints(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if not params:
            raise ValueError(
                f"Function {func.__name__} must take at least one parameter "
                "to derive output_schema."
            )

        first_param_type = hints.get(params[0])
        if first_param_type is not None and _is_model_type(first_param_type):
            resolved_output_schema = first_param_type
        else:
            fields: dict[str, Any] = {}
            for p_name, param in sig.parameters.items():
                p_type = hints.get(p_name, Any)
                default_val = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else ...
                )
                fields[p_name] = (p_type, default_val)
            resolved_output_schema = create_model(
                f"{func.__name__}_OutputModel", **fields
            )

    if resolved_output_schema is None:
        raise ValueError(
            f"SchemaNode {id!r} requires either output_schema or func with "
            "typed parameters"
        )

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
        tools=(),
        mode="schema_extraction",
        kind="schema",
        input_schema=input_schema,
        output_schema=resolved_output_schema,
        handler=func,
    )
