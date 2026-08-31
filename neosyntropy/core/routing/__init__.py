"""Routing module for NeoSyntropy: deterministic, semantic, and preferred routing."""
from .base import Router
from .declarations import (
    RouteTarget,
    _normalize_input_schema,
    _target_id,
    collect_nested_routers,
    collect_router_ids,
    compile_routers,
)
from .deterministic import DeterministicRouter
from .preferred import PreferredPathRouter
from .semantic import SemanticRouter

__all__ = [
    "Router",
    "DeterministicRouter",
    "SemanticRouter",
    "PreferredPathRouter",
    "PreferredPathAdapter",
    "RouteTarget",
    "collect_router_ids",
    "compile_routers",
    "collect_nested_routers",
]
