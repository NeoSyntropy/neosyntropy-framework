"""Validation primitives for every level of the FSM authoring hierarchy.

Three levels of validation, each with a semantic (LLM-backed) and functional
(Python handler) factory:

- **Node level** — :func:`SemanticValidationNode` and
  :func:`functional_validation_node` validate the output of a single node.
  Use these as inline gates mid-path.

- **Group level** — :func:`SemanticGroupPathValidator` and
  :func:`functional_group_path_validator` validate the outcome of traversing
  a group's internal nodes.  Both factories auto-register the validator into
  the group and optionally wire the terminal edge.

- **FSM level** — :func:`SemanticFSMPathValidator` and
  :func:`functional_fsm_path_validator` validate the **entire run path** and
  are designed to sit as the last node before ``End``.  Use
  :func:`extract_fsm_path` inside functional handlers to get a structured view
  of the execution history.

All factories produce nodes whose ``output_schema`` is always
:class:`~neosyntropy.core.node.schemas.ValidationResult` so FSM edges can
branch on the single well-known ``state["valid"]`` key.
"""
from .fsm import (
    FSMPathInfo,
    SemanticFSMPathValidator,
    extract_fsm_path,
    functional_fsm_path_validator,
)
from .group import SemanticGroupPathValidator, functional_group_path_validator
from .node import SemanticValidationNode, functional_validation_node

__all__ = [
    # Node level
    "SemanticValidationNode",
    "functional_validation_node",
    # Group level
    "SemanticGroupPathValidator",
    "functional_group_path_validator",
    # FSM level
    "SemanticFSMPathValidator",
    "functional_fsm_path_validator",
    "FSMPathInfo",
    "extract_fsm_path",
]
