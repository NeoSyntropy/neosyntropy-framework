"""Node authoring primitives — package.

This package replaces the monolithic ``core/node.py`` module.  All public
names are re-exported here so existing import paths such as::

    from neosyntropy.core.node import Node, SchemaNode, NodeContext

continue to work without any changes at the call site.

Sub-module layout
-----------------
schemas.py    — :data:`NodeMode`, :data:`NodeKind`, constants,
                :class:`ValidationResult`, :class:`KpiResult`
_utils.py     — private helpers shared by the factory modules
base.py       — :class:`Node` model and the :func:`node` decorator
schema.py     — :func:`SchemaNode`, :class:`ReasoningStep`,
                :class:`SchemaStep`
reasoning.py  — :func:`ReasoningNode`
combine.py    — :class:`CombineNode`
validation.py — :func:`SemanticValidationNode`,
                :func:`functional_validation_node`
kpi.py        — :func:`SemanticKpiNode`,
                :func:`functional_kpi_node`
context.py    — :class:`NodeContext`
retrieval.py  — :func:`retrieval_node`
"""
from __future__ import annotations

# -- shared type aliases & constants ---------------------------------------
from .schemas import (
    COMBINE_SCHEMA_SUFFIX,
    REASONING_OUTPUT_SCHEMA,
    REASONING_TEXT_KEY,
    TOOL_EVIDENCE_KEY,
    KpiResult,
    NodeKind,
    NodeMode,
    ReasoningLevel,
    ValidationResult,
)

# -- private helpers (kept for import compat — private by convention) ------
from ._utils import _coerce_schema_field, _is_model_type, _shared_kwargs

# -- core model & handler decorator ----------------------------------------
from .base import Node, node

# -- node factory functions & dataclasses ----------------------------------
from .schema import ReasoningStep, SchemaNode, SchemaStep
from .reasoning import ReasoningNode
from .combine import CombineNode
from .validation import SemanticValidationNode, functional_validation_node
from .kpi import SemanticKpiNode, functional_kpi_node
from .context import NodeContext
from .retrieval import retrieval_node

__all__ = [
    # schemas
    "NodeMode",
    "NodeKind",
    "ReasoningLevel",
    "REASONING_OUTPUT_SCHEMA",
    "REASONING_TEXT_KEY",
    "TOOL_EVIDENCE_KEY",
    "COMBINE_SCHEMA_SUFFIX",
    "ValidationResult",
    "KpiResult",
    # private utils (kept for compat)
    "_is_model_type",
    "_coerce_schema_field",
    "_shared_kwargs",
    # base
    "Node",
    "node",
    # schema extraction
    "ReasoningStep",
    "SchemaStep",
    "SchemaNode",
    # reasoning
    "ReasoningNode",
    # combine
    "CombineNode",
    # validation
    "SemanticValidationNode",
    "functional_validation_node",
    # kpi
    "SemanticKpiNode",
    "functional_kpi_node",
    # context
    "NodeContext",
    # retrieval
    "retrieval_node",
]
