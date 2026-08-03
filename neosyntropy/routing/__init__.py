"""Routers: developer declarations and runtime plan proposers.

Authoring (compile to FSM edges)::

    from neosyntropy.routing import DeterministicRouter, SemanticRouter

    intent = SemanticRouter(
        id="CustomerIntent",
        routes={"wants_to_pay": billing_group, "needs_support": support_group},
        fallback_node=general_chat,
    )
    auth = DeterministicRouter(
        id="CheckAuth",
        rules=[
            (lambda ctx: ctx.state.get("token_valid") is True, intent),
            (lambda ctx: ctx.state.get("token_valid") is False, login_node),
        ],
    )

Runtime adapters used inside ControlManager::

    PreferredPathRouter, BackendSemanticRouter
"""
from __future__ import annotations

from .backend_route import BackendSemanticRouter
from .base import Router, RouterError
from .declarations import DeterministicRouter, SemanticRouter
from .preferred import PreferredPathRouter

__all__ = [
    "BackendSemanticRouter",
    "DeterministicRouter",
    "PreferredPathRouter",
    "Router",
    "RouterError",
    "SemanticRouter",
]
