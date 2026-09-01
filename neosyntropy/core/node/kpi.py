"""Backward-compatible re-exports from ``neosyntropy.core.kpi.node``.

The KPI node factories now live at
:mod:`neosyntropy.core.kpi.node`.  This module re-exports them so
existing import paths such as::

    from neosyntropy.core.node.kpi import SemanticKpiNode
    from neosyntropy.core.node.kpi import functional_kpi_node

continue to work.  Prefer importing from :mod:`neosyntropy.core.kpi` in
new code.
"""
from ..kpi.node import (  # noqa: F401
    SemanticKpiNode,
    functional_kpi_node,
)

__all__ = ["SemanticKpiNode", "functional_kpi_node"]
