"""Backend-owned semantic plan proposer (runtime adapter)."""
from __future__ import annotations

from typing import Protocol

from ..core.context import RunContext
from ..core.models import Candidate, RoutingPlan


class _RouteClient(Protocol):
    async def route(
        self,
        context: RunContext,
        candidates: list[Candidate],
        *,
        category: str = "general",
    ) -> RoutingPlan: ...


class BackendSemanticRouter:
    """Routes via the backend-owned semantic router service."""

    def __init__(self, client: _RouteClient, *, category: str = "general") -> None:
        self.client = client
        self.category = category

    async def route(
        self, context: RunContext, candidates: list[Candidate]
    ) -> RoutingPlan:
        return await self.client.route(
            context, candidates, category=self.category
        )
