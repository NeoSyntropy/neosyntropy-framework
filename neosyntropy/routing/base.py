"""Router protocol: routers propose, the graph decides."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.context import RunContext
from ..core.models import Candidate, RoutingPlan


class RouterError(RuntimeError):
    """The router failed to produce a plan (infrastructure error, not a rejection)."""


@runtime_checkable
class Router(Protocol):
    async def route(
        self, context: RunContext, candidates: list[Candidate]
    ) -> RoutingPlan:
        """Propose an execution plan over the candidate list."""
        ...
