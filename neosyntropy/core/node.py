"""Node: an executable capability, not a workflow position.

Multiple nodes may run in one cycle (parallel or sequential steps); there is
still exactly one current state. Tools are capabilities *on* a node
(``allowed tools``), never graph vertices.

Authoring paths:

- :func:`SchemaNode` — provider-backed JSON extraction (input + output schema + prompt)
- :func:`ReasoningNode` — provider-backed reasoning with tools (plain-text notes)
- :func:`CombineNode` — expands to reasoning then schema FSM states
- :func:`node` — Python handler nodes only
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context import RunContext
from .edge import Edge, edge_deterministic
from .models import ExecutionRecord, Message, NodeResult
from .schemas import input_model_schema, strict_model_schema

if TYPE_CHECKING:
    from ..tools.registry import BoundTools

NodeMode = Literal["reasoning", "schema_extraction"]
NodeKind = Literal["schema", "reasoning", "handler", "combine_part"]

# Plain-text notes from a reasoning node (not a JSON object).
REASONING_OUTPUT_SCHEMA: dict[str, Any] = {"type": "string", "minLength": 1}

# State keys written by the reasoning half for the schema half to read.
REASONING_TEXT_KEY = "reasoning_text"
TOOL_EVIDENCE_KEY = "tool_evidence"

# CombineNode exit state id suffix.
COMBINE_SCHEMA_SUFFIX = ".Schema"


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


def _shared_kwargs(
    *,
    id: str,
    name: str | None,
    description: str,
    prompt: str,
    prerequisites: Sequence[str],
    group: str | None,
    is_fallback: bool,
    metadata: dict[str, Any] | None,
    provider: str,
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name or id,
        "description": description,
        "prompt": prompt,
        "prerequisites": tuple(prerequisites),
        "group": group,
        "is_fallback": is_fallback,
        "metadata": metadata or {},
        "provider": provider,
    }


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


def SchemaNode(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    output_schema: type[BaseModel] | dict[str, Any],
    prompt: str,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
) -> Node:
    """Provider-backed schema extraction: constrained JSON, no tools."""
    if not prompt:
        raise ValueError(f"SchemaNode {id!r} requires prompt")
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
        output_schema=output_schema,
        handler=None,
    )


def ReasoningNode(
    id: str,
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    tools: Sequence[str],
    prompt: str,
    name: str | None = None,
    description: str = "",
    prerequisites: Sequence[str] = (),
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    provider: str = "neosyntropy/base",
) -> Node:
    """Provider-backed reasoning: tools allowed, plain-text notes out."""
    if not prompt:
        raise ValueError(f"ReasoningNode {id!r} requires prompt")
    tool_names = tuple(tools)
    if not tool_names:
        raise ValueError(f"ReasoningNode {id!r} requires tools")
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
        tools=tool_names,
        mode="reasoning",
        kind="reasoning",
        input_schema=input_schema,
        output_schema=REASONING_OUTPUT_SCHEMA,
        handler=None,
    )


@dataclass
class CombineNode:
    """Authoring unit: expands to reasoning then schema FSM states.

    Entry state ``{id}`` (reasoning + tools) → exit ``{id}.Schema`` (JSON).
    External edges should target ``{id}`` and leave from ``{id}.Schema``.
    """

    id: str
    input_schema: type[BaseModel] | dict[str, Any]
    tools: Sequence[str]
    output_schema: type[BaseModel] | dict[str, Any]
    prompt: str
    name: str | None = None
    description: str = ""
    prerequisites: Sequence[str] = ()
    group: str | None = None
    is_fallback: bool = False
    metadata: dict[str, Any] | None = None
    provider: str = "neosyntropy/base"
    schema_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError(f"CombineNode {self.id!r} requires prompt")
        if not tuple(self.tools):
            raise ValueError(f"CombineNode {self.id!r} requires tools")
        if self.is_fallback:
            raise ValueError(
                f"CombineNode {self.id!r} cannot be fallback "
                "(use SchemaNode(..., is_fallback=True))"
            )

    @property
    def schema_id(self) -> str:
        return f"{self.id}{COMBINE_SCHEMA_SUFFIX}"

    def expand(self) -> tuple[list[Node], list[Edge]]:
        """Return the two nodes and the linking deterministic edge."""
        meta = dict(self.metadata or {})
        meta.setdefault("combine_id", self.id)
        reasoning = ReasoningNode(
            id=self.id,
            input_schema=self.input_schema,
            tools=self.tools,
            prompt=self.prompt,
            name=self.name,
            description=self.description,
            prerequisites=self.prerequisites,
            group=self.group,
            metadata={**meta, "combine_role": "reasoning"},
            provider=self.provider,
        )
        object.__setattr__(reasoning, "kind", "combine_part")

        extract_prompt = self.schema_prompt or (
            f"Extract structured JSON from the prior reasoning notes for {self.id}."
        )
        schema = SchemaNode(
            id=self.schema_id,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            prompt=extract_prompt,
            name=self.name or self.schema_id,
            description=self.description,
            prerequisites=(self.id,),
            group=self.group,
            metadata={**meta, "combine_role": "schema"},
            provider=self.provider,
        )
        object.__setattr__(schema, "kind", "combine_part")

        return (
            [reasoning, schema],
            [edge_deterministic(self.id, self.schema_id)],
        )


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
    def input(self) -> dict[str, Any]:
        return self.run.input

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
