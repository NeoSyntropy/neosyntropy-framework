"""Node: an executable capability, not a workflow position.

Multiple nodes may run in one cycle (parallel or sequential steps); there is
still exactly one current state. Tools are capabilities *on* a node
(``allowed tools``), never graph vertices.

Every provider-backed node runs in one of two modes:

- ``reasoning`` — the model may call allow-listed tools before producing output.
- ``schema_extraction`` — the model returns constrained JSON directly and
  therefore cannot declare tools.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context import RunContext
from .models import ExecutionRecord, Message, NodeResult
from .schemas import input_model_schema, strict_model_schema

if TYPE_CHECKING:
    from ..tools.registry import BoundTools

NodeMode = Literal["reasoning", "schema_extraction"]


def _is_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _coerce_schema_field(
    data: dict[str, Any],
    *,
    field: str,
    model_field: str,
    strict: bool,
) -> dict[str, Any]:
    raw = data.get(field)
    node_id = data.get("id")
    if raw is None:
        raise ValueError(
            f"node {node_id!r} requires {field} "
            "(pass a pydantic BaseModel or a JSON Schema object)"
        )
    if _is_model_type(raw):
        data[field] = (
            strict_model_schema(raw) if strict else input_model_schema(raw)
        )
        data[model_field] = raw
        return data
    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f"node {node_id!r} {field} must not be empty")
        data[field] = raw
        return data
    raise ValueError(
        f"node {node_id!r} {field} must be a pydantic "
        "BaseModel class or a JSON Schema object"
    )


class Node(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str
    name: str = ""
    description: str = ""
    # Provider assignment is a backend concern. This value remains in the
    # model only to read older graph definitions.
    provider: str = "backend"
    prompt: str = ""
    prerequisites: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    # Execution mode. Defaults from tools: any allow-list ⇒ reasoning;
    # otherwise schema_extraction. Schema extraction returns JSON directly
    # and rejects tools.
    mode: NodeMode | None = None
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
    def default_name_and_mode(self) -> Node:
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

        if not self.input_schema:
            raise ValueError(f"node {self.id!r} requires input_schema")
        if not self.output_schema:
            raise ValueError(f"node {self.id!r} requires output_schema")
        return self

    def input_error(self, state: Mapping[str, Any]) -> str | None:
        """Return why ``state`` fails this node's input contract, or None."""
        try:
            jsonschema.validate(instance=dict(state), schema=self.input_schema)
        except jsonschema.exceptions.ValidationError as exc:
            return (
                f"node {self.id!r} input does not match input_schema: {exc.message}"
            )
        except jsonschema.exceptions.SchemaError as exc:
            return f"node {self.id!r} input_schema is invalid: {exc.message}"
        return None


@dataclass
class NodeContext:
    """What a node handler receives: a snapshot plus its bound tools.

    ``run.state`` is a snapshot; mutating it commits nothing. Propose state
    changes through the returned :class:`NodeResult` (``ctx.result(...)``).
    """

    run: RunContext
    node: Node
    tools: BoundTools

    @property
    def intent(self) -> str:
        return self.run.intent

    @property
    def state(self) -> dict[str, Any]:
        return self.run.state

    @property
    def current_state(self) -> str:
        return self.run.current_state

    @property
    def history(self) -> list[Message]:
        return self.run.history

    @property
    def prior_executions(self) -> list[ExecutionRecord]:
        return self.run.prior_executions

    @property
    def metadata(self) -> dict[str, Any]:
        return self.run.metadata

    def result(
        self,
        output: Any = None,
        *,
        state_updates: dict[str, Any] | None = None,
        next_state: str | None = None,
        status: str = "succeeded",
        error: str | None = None,
    ) -> NodeResult:
        """Build a proposal for this node without repeating its id."""
        return NodeResult(
            node_id=self.node.id,
            status=status,  # type: ignore[arg-type]
            output=output,
            state_updates=state_updates or {},
            next_state=next_state,
            error=error,
        )


def node(
    id: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    provider: str = "backend",
    prompt: str = "",
    prerequisites: tuple[str, ...] | list[str] = (),
    tools: tuple[str, ...] | list[str] = (),
    mode: NodeMode | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Node]:
    """Declare a node from a Python handler.

    ``input_schema`` and ``output_schema`` are required. Use
    :class:`~neosyntropy.OpenInput` when the node does not constrain state.
    """

    def decorator(fn: Callable[..., Any]) -> Node:
        import inspect

        return Node(
            id=id or fn.__name__,
            name=name or (id or fn.__name__),
            description=(description or inspect.getdoc(fn) or "").strip(),
            provider=provider,
            prompt=prompt,
            prerequisites=tuple(prerequisites),
            tools=tuple(tools),
            mode=mode,
            input_schema=input_schema,
            output_schema=output_schema,
            group=group,
            is_fallback=is_fallback,
            metadata=metadata or {},
            handler=fn,
        )

    return decorator
