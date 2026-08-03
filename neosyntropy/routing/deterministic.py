"""Deterministic routing.

Developer API: :class:`DeterministicRouter` declarations (rules → nodes/routers).
Runtime offline adapter: :class:`PreferredPathRouter`.
"""
from __future__ import annotations

from .declarations import DeterministicRouter
from .preferred import PreferredPathRouter

# Backward-compatible name for the runtime preferred-path adapter.
# Prefer PreferredPathRouter in new framework code.
GraphDeterministicRouter = PreferredPathRouter

__all__ = [
    "DeterministicRouter",
    "PreferredPathRouter",
    "GraphDeterministicRouter",
]
