"""CombineNode: a two-part authoring unit that expands to a reasoning + schema pair.

External graph edges should target the entry state ``{id}`` and leave from
the exit state ``{id}.Schema``.  The internal deterministic edge between the
two states is produced automatically by :meth:`CombineNode.expand`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..edge import Edge, edge_deterministic
from .base import Node
from .reasoning import ReasoningNode
from .schema import SchemaNode
from .schemas import COMBINE_SCHEMA_SUFFIX


@dataclass
class CombineNode:
    """Authoring unit: expands to reasoning then schema FSM states.

    Entry state ``{id}`` (reasoning + tools) → exit ``{id}.Schema`` (JSON).
    External edges should target ``{id}`` and leave from ``{id}.Schema``.

    Attributes:
        id:            Unique node id.  The schema half uses ``{id}.Schema``.
        input_schema:  Shared input schema for both the reasoning and schema
                       sub-nodes.
        tools:         Tool names available to the reasoning sub-node.
                       Empty when the pair is used as a high-reasoning
                       wrapper with no tool calls.
        output_schema: JSON shape produced by the schema sub-node.
        prompt:        Reasoning prompt (required).
        name:          Optional human-readable display name.
        description:   Optional longer description.
        prerequisites: Node ids that must succeed before this node runs.
        group:         Optional group this node belongs to.
        is_fallback:   Always ``False`` — use SchemaNode for fallbacks.
        metadata:      Arbitrary key-value pairs for observability.
        provider:      Provider id used for inference.
        schema_prompt: Override prompt for the extraction sub-node.
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
        if self.is_fallback:
            raise ValueError(
                f"CombineNode {self.id!r} cannot be fallback "
                "(use SchemaNode(..., is_fallback=True))"
            )

    @property
    def schema_id(self) -> str:
        """Id of the schema extraction sub-node (``{id}.Schema``)."""
        return f"{self.id}{COMBINE_SCHEMA_SUFFIX}"

    def expand(self) -> tuple[list[Node], list[Edge]]:
        """Return the two nodes and the linking deterministic edge.

        Call this when building a graph; the returned nodes and edge should be
        added alongside any external edges that reference this unit.
        """
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
