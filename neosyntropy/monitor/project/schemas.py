from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ProjectEvent(BaseModel):
    project_id: str
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
