"""Shared private helpers used by multiple node factory modules.

These are implementation details — callers should import from the
public package ``__init__`` rather than this module directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..schemas import input_model_schema, strict_model_schema
from .schemas import ReasoningLevel


def _resolve_reasoning_level(
    reasoning: str,
    tools: Sequence[str] = (),
    *,
    owner: str,
) -> ReasoningLevel:
    """Return ``high`` when tools are set; otherwise validate ``low`` / ``high``."""
    if reasoning not in ("low", "high"):
        raise ValueError(
            f"{owner} reasoning must be 'low' or 'high', got {reasoning!r}"
        )
    if tuple(tools):
        return "high"
    return reasoning  # type: ignore[return-value]


def _is_model_type(value: Any) -> bool:
    """Return True when *value* is a pydantic BaseModel subclass (not an instance)."""
    return isinstance(value, type) and issubclass(value, BaseModel)


def _coerce_schema_field(
    data: dict[str, Any],
    *,
    field: str,
    model_field: str,
    strict: bool,
) -> dict[str, Any]:
    """Convert a pydantic model class or plain dict to a JSON Schema dict in-place.

    Mutates and returns *data*.  Raises ``ValueError`` on any invalid input.

    Args:
        data:        The raw constructor kwargs dict being built.
        field:       The key holding the raw schema value (``"input_schema"`` or
                     ``"output_schema"``).
        model_field: The key that will store the original model class so the
                     executor can validate instances at runtime.
        strict:      When ``True`` use the strict (constrained-decoding) schema;
                     when ``False`` use the permissive input schema.
    """
    raw = data.get(field)
    node_id = data.get("id")
    if raw is None:
        raise ValueError(
            f"node {node_id!r} requires {field} "
            "(pass a pydantic BaseModel or a JSON Schema object)"
        )
    if _is_model_type(raw):
        data[field] = strict_model_schema(raw) if strict else input_model_schema(raw)
        data[model_field] = raw
        return data
    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f"node {node_id!r} {field} must not be empty")
        data[field] = raw
        return data
    raise ValueError(
        f"node {node_id!r} {field} must be a pydantic "
        "BaseModel class or a JSON Schema object"
    )


def _shared_kwargs(
    *,
    id: str,
    name: str | None,
    description: str,
    prompt: str,
    prerequisites: Sequence[str],
    group: str | None,
    is_fallback: bool,
    metadata: dict[str, Any] | None,
    provider: str,
) -> dict[str, Any]:
    """Build the common Node constructor kwargs shared by every factory function."""
    return {
        "id": id,
        "name": name or id,
        "description": description,
        "prompt": prompt,
        "prerequisites": tuple(prerequisites),
        "group": group,
        "is_fallback": is_fallback,
        "metadata": metadata or {},
        "provider": provider,
    }
