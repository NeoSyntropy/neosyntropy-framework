"""Knowledge manifest generator for UI visualisation and telemetry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neosyntropy.knowledge.knowledge import Knowledge


def _vector_db_info(vdb: Any) -> dict[str, Any]:
    """Extract serialisable metadata from a vector-db instance."""
    return {
        "type": type(vdb).__name__,
        "name": getattr(vdb, "name", None)
        or getattr(vdb, "table_name", None)
        or getattr(vdb, "collection_name", None),
        "uri": getattr(vdb, "uri", None) or getattr(vdb, "url", None),
    }


def _database_info(db: Any) -> dict[str, Any]:
    """Extract serialisable metadata from a relational/NoSQL db instance."""
    return {
        "type": type(db).__name__,
        "name": getattr(db, "name", None),
    }


def knowledge_manifest(kb: "Knowledge") -> dict[str, Any]:
    """Return the visualisation shape for a Knowledge instance.

    This is the canonical definition of what gets sent to the server when a
    Knowledge object is registered.  The structure is stored as
    ``KnowledgeBase.manifest`` on the backend, parallel to how
    ``graph_manifest`` is stored as ``ProjectGraph.manifest``.
    """
    embedder = getattr(kb, "embedder", None)
    reranker = getattr(kb, "reranker", None)
    transform = getattr(kb, "transform_pipeline", None)

    return {
        "schema_version": 1,
        "name": kb.name,
        "description": kb.description,
        "topics": kb.topics,
        "vector_dbs": [_vector_db_info(v) for v in kb.vector_dbs],
        "databases": [_database_info(d) for d in kb.databases],
        "embedder": type(embedder).__name__ if embedder is not None else None,
        "reranker": type(reranker).__name__ if reranker is not None else None,
        "transform_pipeline": type(transform).__name__ if transform is not None else None,
        "content_count": len(kb.contents) if kb.contents else 0,
        "content_sources": [
            {
                "type": type(s).__name__,
                "path": getattr(s, "path", None) or getattr(s, "bucket", None),
            }
            for s in (kb.content_sources or [])
        ],
    }
