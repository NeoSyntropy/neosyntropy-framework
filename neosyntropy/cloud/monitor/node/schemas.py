from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NodeManifest(BaseModel):
    """Wire shape stored as NodeManifest on the backend."""

    schema_version: int = 3
    id: str
    name: str | None = None
    description: str | None = None
    kind: str | None = None
    mode: str | None = None
    prompt: str | None = None
    provider: str = "neosyntropy/base"
    prerequisites: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    group: str | None = None
    is_fallback: bool = False
    metadata: dict[str, Any] | None = None
    structure_hash: str | None = None


class NodeEvent(BaseModel):
    """Lifecycle event for a node execution."""

    node_id: str
    event_type: str
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class NodeExecutionEvent(BaseModel):
    """Emitted when a node completes execution."""

    node_id: str
    status: str
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: datetime
    metadata: dict[str, Any] | None = None
