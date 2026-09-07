from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ProjectManifest(BaseModel):
    """Wire shape for a project registration event."""

    schema_version: int = 1
    project_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProjectEvent(BaseModel):
    """Lifecycle event for a project."""

    project_id: str
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
