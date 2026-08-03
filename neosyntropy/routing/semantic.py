"""Semantic routing.

Developer API: :class:`SemanticRouter` declarations (labeled routes → nodes/groups).
Runtime backend adapter: :class:`BackendSemanticRouter`.
"""
from __future__ import annotations

from .backend_route import BackendSemanticRouter
from .declarations import SemanticRouter

__all__ = [
    "SemanticRouter",
    "BackendSemanticRouter",
]
