from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class NodeManifest(BaseModel):
    """Wire shape stored as NodeManifest on the backend."""

    schema_version: int = 1
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    kind: Optional[str] = None
    mode: Optional[str] = None
    prompt: Optional[str] = None
    tools: List[str] = []
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    group: Optional[str] = None
    is_fallback: bool = False
    metadata: Optional[Dict[str, Any]] = None


class NodeEvent(BaseModel):
    """Lifecycle event for a node execution."""

    node_id: str
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class NodeExecutionEvent(BaseModel):
    """Emitted when a node completes execution."""

    node_id: str
    status: str
    duration_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
