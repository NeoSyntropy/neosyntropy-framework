"""Node: an executable capability, not a workflow position.

Multiple nodes may run in one cycle (parallel or sequential steps); there is
still exactly one current state. Tools are capabilities *on* a node
(``allowed tools``), never graph vertices.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context import RunContext
from .models import ExecutionRecord, Message, NodeResult

if TYPE_CHECKING:
    from ..tools.registry import BoundTools


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
    group: str | None = None
    is_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any] | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def default_name(self) -> Node:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self


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
    group: str | None = None,
    is_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Node]:
    """Declare a node from a Python handler.

    The decorator returns the :class:`Node` itself, ready to be added to a
    :class:`~neosyntropy.core.graph.Graph`::

        @node(id="refund.calculate", group="refunds", prerequisites=("verify_identity",))
        def calculate_refund(ctx: NodeContext) -> NodeResult:
            ...

    Handlers may be sync or async and may return a :class:`NodeResult`
    (preferred, via ``ctx.result(...)``), ``None``, or any value (recorded as
    ``output``).
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
            group=group,
            is_fallback=is_fallback,
            metadata=metadata or {},
            handler=fn,
        )

    return decorator
