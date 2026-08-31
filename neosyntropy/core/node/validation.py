"""Backward-compatible re-exports from ``neosyntropy.core.validation.node``.

The validation node factories now live at
:mod:`neosyntropy.core.validation.node`.  This module re-exports them so
existing imports continue to work without modification::

    # Still works:
    from neosyntropy.core.node.validation import SemanticValidationNode
    from neosyntropy.core.node.validation import functional_validation_node

Prefer importing from :mod:`neosyntropy.core.validation` in new code.
"""
from __future__ import annotations

from ..validation.node import (  # noqa: F401
    SemanticValidationNode,
    functional_validation_node,
)

__all__ = ["SemanticValidationNode", "functional_validation_node"]
