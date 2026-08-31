"""NodeContext: the runtime object passed to every Python handler node.

A ``NodeContext`` is a read-only snapshot of the workflow run at the moment
the handler is invoked.  Handlers must *propose* state changes via
:meth:`NodeContext.result` rather than mutating the state dict directly;
those proposals are committed by the control manager only after all gate
checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..context import RunContext
from ..models import ExecutionRecord, Message, NodeResult
from .base import Node

if TYPE_CHECKING:
    from ...tools.core.registry import BoundTools


@dataclass
class NodeContext:
    """What a node handler receives: a snapshot plus its bound tools.

    ``run.state`` is a snapshot; mutating it commits nothing. Propose state
    changes through the returned :class:`~neosyntropy.core.models.NodeResult`
    by calling :meth:`result`.

    Attributes:
        run:   The current :class:`~neosyntropy.core.context.RunContext`.
        node:  The :class:`~base.Node` definition being executed.
        tools: The :class:`~neosyntropy.tools.core.registry.BoundTools`
               instance scoped to this node's allowed tool set.
    """

    run: RunContext
    node: Node
    tools: BoundTools

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def input(self) -> dict[str, Any]:
        """Immutable run input (the original request payload)."""
        return self.run.input

    @property
    def state(self) -> dict[str, Any]:
        """Current workflow state snapshot (read-only — propose via result())."""
        return self.run.state

    @property
    def current_state(self) -> str:
        """Id of the FSM state the workflow is currently in."""
        return self.run.current_state

    @property
    def history(self) -> list[Message]:
        """Conversation history available to the handler."""
        return self.run.history

    @property
    def prior_executions(self) -> list[ExecutionRecord]:
        """Audit records for nodes that already ran in this cycle."""
        return self.run.prior_executions

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary request-scoped metadata dict."""
        return self.run.metadata

    # ------------------------------------------------------------------
    # Proposal builder
    # ------------------------------------------------------------------

    def result(
        self,
        output: Any = None,
        *,
        state_updates: dict[str, Any] | None = None,
        next_state: str | None = None,
        status: str = "succeeded",
        error: str | None = None,
    ) -> NodeResult:
        """Build a :class:`~neosyntropy.core.models.NodeResult` for this node.

        Omitting ``node_id`` is intentional — the context already knows the
        owning node.

        Args:
            output:        The value to store as ``NodeResult.output``.
            state_updates: Key-value pairs to merge into the workflow state
                           after gate checks pass.
            next_state:    Explicit FSM transition target (optional).
            status:        ``"succeeded"`` (default), ``"failed"``, or
                           ``"fallback"``.
            error:         Human-readable error description when ``status``
                           is not ``"succeeded"``.
        """
        return NodeResult(
            node_id=self.node.id,
            status=status,  # type: ignore[arg-type]
            output=output,
            state_updates=state_updates or {},
            next_state=next_state,
            error=error,
        )
