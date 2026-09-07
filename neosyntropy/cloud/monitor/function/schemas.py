from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class FunctionManifest(BaseModel):
    """Wire shape stored as FunctionManifest on the backend."""

    schema_version: int = 1
    name: str | None = None
    function_name: str
    function_module: str | None = None
    docstring: str | None = None
    description: str | None = None
    is_async: bool = False
    node_count: int | None = None
    entry: str | None = None
    input_schema: dict[str, Any] | None = None
    decorator: str | None = None


class FunctionEvent(BaseModel):
    """Lifecycle event for a decorated function invocation."""

    function_name: str
    event_type: str
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class FunctionCallEvent(BaseModel):
    """Emitted when a decorated function is called."""

    function_name: str
    duration_ms: float | None = None
    status: str
    timestamp: datetime
    metadata: dict[str, Any] | None = None
