"""Type aliases, module-level constants, and the shared ValidationResult model.

Every other module in this package imports from here rather than defining
its own type aliases or duplicating constants.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

NodeMode = Literal["reasoning", "schema_extraction"]
NodeKind = Literal["schema", "reasoning", "handler", "combine_part"]
ReasoningLevel = Literal["low", "high"]

# Plain-text notes from a reasoning node (not a JSON object).
REASONING_OUTPUT_SCHEMA: dict[str, Any] = {"type": "string", "minLength": 1}

# State keys written by the reasoning half for the schema half to read.
REASONING_TEXT_KEY = "reasoning_text"
TOOL_EVIDENCE_KEY = "tool_evidence"

# CombineNode exit state id suffix.
COMBINE_SCHEMA_SUFFIX = ".Schema"


class ValidationResult(BaseModel):
    """Standard output contract for all validation nodes.

    Both :func:`SemanticValidationNode` and :func:`functional_validation_node`
    emit this shape so FSM edges can branch on a single well-known key.

    Attributes:
        valid:  ``True`` means the check passed; ``False`` means it failed.
        reason: Human-readable explanation produced by the LLM or the
                developer's handler.  Empty string when omitted.
    """

    valid: bool
    reason: str = ""


class KpiResult(BaseModel):
    """Standard output contract for all KPI nodes.

    KPI nodes **never** fail the run — they score it.  If a threshold gate is
    needed, place a :func:`~neosyntropy.core.validation.node.functional_validation_node`
    after the KPI node and branch on ``state["valid"]``.

    The ``valid`` field intentionally does not exist on this model: that key
    belongs to validation edges and must not bleed into KPI semantics.

    Attributes:
        name:   Developer-given metric name (e.g. ``"completeness"``).
                Written as the key in the accumulated ``state["kpis"]`` dict.
        score:  Numeric measurement.  Typically ``0.0``–``1.0`` but the range
                is not enforced by the schema.
        reason: Human-readable explanation.  Empty string when omitted.
    """

    name: str
    score: float
    reason: str = ""
