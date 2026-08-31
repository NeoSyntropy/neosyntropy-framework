from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class FunctionManifest(BaseModel):
    """Wire shape stored as FunctionManifest on the backend."""

    schema_version: int = 1
    name: Optional[str] = None
    function_name: str
    function_module: Optional[str] = None
    docstring: Optional[str] = None
    description: Optional[str] = None
    is_async: bool = False
    source_code: Optional[str] = None
    node_count: Optional[int] = None
    entry: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    decorator: Optional[str] = None


class FunctionEvent(BaseModel):
    """Lifecycle event for a decorated function invocation."""

    function_name: str
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class FunctionCallEvent(BaseModel):
    """Emitted when a decorated function is called."""

    function_name: str
    duration_ms: Optional[float] = None
    status: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
