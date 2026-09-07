from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class VectorDbInfo(BaseModel):
    type: str
    name: Optional[str] = None
    uri: Optional[str] = None


class DatabaseInfo(BaseModel):
    type: str
    name: Optional[str] = None


class ContentSourceInfo(BaseModel):
    type: str
    path: Optional[str] = None


class KnowledgeManifest(BaseModel):
    """Wire shape stored as KnowledgeBase.manifest on the backend."""

    schema_version: int = 1
    name: str
    description: Optional[str] = None
    topics: List[str] = []
    vector_dbs: List[VectorDbInfo] = []
    databases: List[DatabaseInfo] = []
    embedder: Optional[str] = None
    reranker: Optional[str] = None
    transform_pipeline: Optional[str] = None
    content_count: int = 0
    content_sources: List[ContentSourceInfo] = []


class KnowledgeRow(BaseModel):
    """Compatibility export expected by ``monitor.knowledge`` package imports."""

    knowledge_id: str
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeEvent(BaseModel):
    """Lifecycle event for a knowledge base."""

    knowledge_id: str
    event_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeQueryEvent(BaseModel):
    """Emitted when a knowledge base is queried."""

    knowledge_id: str
    query: str
    result_count: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
