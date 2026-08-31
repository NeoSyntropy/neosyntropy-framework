"""Runtime context passed through one control cycle."""
from __future__ import annotations

from typing import Any

from pydantic import Field

from .models import ExecutionRecord, Message, RunRequest, RuntimeModel


class RunContext(RuntimeModel):
    """Normalized, trusted view of one request.

    Built once per cycle from a :class:`RunRequest`. Nodes receive snapshots
    of this context; mutating it never commits state.
    """

    request_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    current_state: str
    history: list[Message] = Field(default_factory=list)
    prior_executions: list[ExecutionRecord] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBuilder:
    """Maps external evidence (the request) into runtime context.

    Adapters that authenticate/normalize webhooks or events should feed a
    ``RunRequest``; they never choose nodes or mutate state.
    """

    def build(self, request: RunRequest) -> RunContext:
        return RunContext(
            request_id=request.request_id,
            input=dict(request.input),
            current_state=request.current_state,
            history=list(request.history),
            prior_executions=list(request.prior_executions),
            state=dict(request.state),
            metadata=dict(request.metadata),
        )
