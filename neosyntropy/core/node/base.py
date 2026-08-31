"""The Node model and the ``@node`` decorator for Python handler nodes.

Provider-backed authoring (schema extraction, reasoning, combine) lives in
the sibling modules :mod:`schema`, :mod:`reasoning`, and :mod:`combine`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..context import RunContext  # noqa: F401 — re-exported for NodeContext
from ..edge import Edge
from ..models import ExecutionRecord, Message, NodeResult
from ._utils import _coerce_schema_field, _shared_kwargs
from .schemas import NodeKind, NodeMode

if TYPE_CHECKING:
    from ...tools.core.registry import BoundTools


class Node(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str
    name: str = ""
    description: str = ""
    # Provider assignment is a backend concern. This value remains in the
    # model only to read older graph definitions.
    provider: str = "neosyntropy/base"
    prompt: str = ""
    prerequisites: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    # Runtime execution mode on the wire / in the executor.
    mode: NodeMode | None = None
    # Authoring kind (schema / reasoning / handler / combine half).
    kind: NodeKind | None = None
    # Contract on the pre-step workflow state. Required. Validated by the
    # control manager before the node runs (fail-closed).
    input_schema: dict[str, Any]
    # JSON Schema for the node's structured output. Required. Pass a pydantic
    # model class at construction time (`output_schema=MyModel`); it is stored
    # as the constrained-decoding schema.
    output_schema: dict[str, Any]
    group: str | None = None
    is_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any] | None = Field(default=None, exclude=True, repr=False)
    input_model: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)
    output_model: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def coerce_schemas(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data = _coerce_schema_field(
            data,
            field="input_schema",
            model_field="input_model",
            strict=False,
        )
        data = _coerce_schema_field(
            data,
            field="output_schema",
            model_field="output_model",
            strict=True,
        )
        return data

    @model_validator(mode="after")
    def default_name_mode_kind(self) -> Node:
        if not self.name:
            object.__setattr__(self, "name", self.id)

        resolved_mode: NodeMode = self.mode or (
            "reasoning" if self.tools else "schema_extraction"
        )
        if resolved_mode == "schema_extraction" and self.tools:
            raise ValueError(
                f"node {self.id!r} mode='schema_extraction' cannot declare tools "
                "(it returns JSON directly); use mode='reasoning' for tool calling"
            )
        object.__setattr__(self, "mode", resolved_mode)

        resolved_kind: NodeKind = self.kind or (
            "handler"
            if self.handler is not None
            else ("reasoning" if resolved_mode == "reasoning" else "schema")
        )
        object.__setattr__(self, "kind", resolved_kind)

        if not self.input_schema:
            raise ValueError(f"node {self.id!r} requires input_schema")
        if not self.output_schema:
            raise ValueError(f"node {self.id!r} requires output_schema")
        return self

    def compile(self) -> tuple[list[Node], list[Edge]]:
        """Compile into base execution primitives. Returns ([self], [])."""
        return ([self], [])

    def input_error(self, payload: Mapping[str, Any]) -> str | None:
        """Return why ``payload`` fails this node's declared input_schema, or None.

        Used for documenting / introspecting the node contract. Runtime
        enforcement is the FSM entry gate on immutable run ``input`` only —
        workflow ``state`` is not schema-checked (any node may update it).
        """
        try:
            jsonschema.validate(instance=dict(payload), schema=self.input_schema)
        except jsonschema.exceptions.ValidationError as exc:
            return (
                f"node {self.id!r} input does not match input_schema: {exc.message}"
            )
        except jsonschema.exceptions.SchemaError as exc:
            return f"node {self.id!r} input_schema is invalid: {exc.message}"
        return None

    async def __call__(self, client: Any, **input_data: Any) -> Any:
        """Execute this node directly as a function.

        This handles LLM schema extraction and automatically calls the wrapped
        handler function (if one is attached) using the extracted parameters.
        """
        from ..graph import Workflow
        from .schema import SchemaNode

        fallback = SchemaNode(
            id="temp_fallback",
            input_schema=self.input_schema,
            output_schema=self.input_schema,
            prompt="Dummy fallback",
            is_fallback=True,
        )
        fsm = Workflow([self], fallback=fallback)

        if self.input_model:
            validated_input = self.input_model(**input_data)
        else:
            validated_input = input_data

        result = await fsm.arun(validated_input, client=client)
        if hasattr(result, "final_state"):
            return result.final_state
        return result


def node(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    provider: str = "neosyntropy/base",
    prompt: str = "",
    prerequisites: tuple[str, ...] | list[str] = (),
    tools: tuple[str, ...] | list[str] = (),
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Node]:
    """Declare a Python handler node.

    Provider-backed nodes use :func:`SchemaNode`, :func:`ReasoningNode`, or
    :func:`CombineNode` instead. ``input_schema`` and ``output_schema`` are
    required. Use :class:`~neosyntropy.OpenInput` when the node does not
    constrain state.
    """

    def decorator(fn: Callable[..., Any]) -> Node:
        import inspect

        tool_names = tuple(tools)
        return Node(
            id=id or fn.__name__,
            name=name or (id or fn.__name__),
            description=(description or inspect.getdoc(fn) or "").strip(),
            provider=provider,
            prompt=prompt,
            prerequisites=tuple(prerequisites),
            tools=tool_names,
            mode="reasoning" if tool_names else "schema_extraction",
            kind="handler",
            input_schema=input_schema,
            output_schema=output_schema,
            group=group,
            is_fallback=is_fallback,
            metadata=metadata or {},
            handler=fn,
        )

    return decorator
