"""KPI scoring primitives for every level of the FSM authoring hierarchy.

Three levels of KPI, each with a semantic (LLM-backed) and functional
(Python handler) factory — mirroring the validation package:

- **Node level** — :func:`SemanticKpiNode` and :func:`functional_kpi_node`
  score a single node's output mid-path.

- **Group level** — :func:`SemanticGroupPathKpi` and
  :func:`functional_group_path_kpi` score the outcome of traversing a
  group's internal nodes.  Both factories auto-register the scorer into the
  group and optionally wire the terminal edge.

- **FSM level** — :func:`SemanticFSMPathKpi` and
  :func:`functional_fsm_path_kpi` score the **entire run path** and are
  designed to sit as the last node before ``End``.  Use
  :func:`extract_fsm_path` inside functional handlers to get a structured
  view of the execution history.

All factories produce nodes whose ``output_schema`` is always
:class:`~neosyntropy.core.node.schemas.KpiResult`.

Unlike validation nodes, KPI nodes **never** fail the run.  If a score
threshold must gate execution, place a
:func:`~neosyntropy.core.validation.node.functional_validation_node` after
the KPI node and branch on ``state["valid"]``.

``extract_fsm_path`` and :class:`FSMPathInfo` are re-exported from the
validation package so you can import everything from one place:

    from neosyntropy.core.kpi import (
        functional_fsm_path_kpi,
        extract_fsm_path,
    )
"""
from .fsm import (
    FSMPathInfo,
    SemanticFSMPathKpi,
    extract_fsm_path,
    functional_fsm_path_kpi,
)
from .group import SemanticGroupPathKpi, functional_group_path_kpi
from .node import SemanticKpiNode, functional_kpi_node

__all__ = [
    # Node level
    "SemanticKpiNode",
    "functional_kpi_node",
    # Group level
    "SemanticGroupPathKpi",
    "functional_group_path_kpi",
    # FSM level
    "SemanticFSMPathKpi",
    "functional_fsm_path_kpi",
    # Shared path helpers (re-exported from validation for convenience)
    "FSMPathInfo",
    "extract_fsm_path",
]
