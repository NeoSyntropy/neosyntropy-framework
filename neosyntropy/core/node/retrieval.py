"""retrieval_node: FSM node that queries a vector store or knowledge base.

The node reads a query string from the workflow state, performs a semantic
search against a :class:`~neosyntropy.vectordb.base.VectorDb` or
:class:`~neosyntropy.knowledge.protocol.KnowledgeRetrievalProtocol`, and
writes the retrieved documents back into state under ``output_key``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import NodeResult
from .base import Node

if TYPE_CHECKING:
    from ...knowledge.protocol import KnowledgeRetrievalProtocol
    from ...vectordb.base import VectorDb


def retrieval_node(
    id: str,
    vector_db: "VectorDb | KnowledgeRetrievalProtocol | None" = None,
    query_key: str = "query",
    output_key: str = "context",
    limit: int = 5,
    description: str = "Retrieves semantic context from the knowledge base or vector database.",
    name: str | None = None,
    group: str | None = None,
    format_as_string: bool = False,
    knowledge: "KnowledgeRetrievalProtocol | VectorDb | None" = None,
    **kwargs: Any,
) -> Node:
    """Create an FSM :class:`~base.Node` that retrieves documents and injects them into state.

    Args:
        id:              Unique node id.
        vector_db:       The :class:`~neosyntropy.vectordb.base.VectorDb` or
                         :class:`~neosyntropy.knowledge.protocol.KnowledgeRetrievalProtocol`
                         to query.
        query_key:       State key that contains the search query string.
        output_key:      State key where retrieved results are written.
        limit:           Maximum number of documents to retrieve.
        description:     Node description shown in observability tooling.
        name:            Optional human-readable display name.
        group:           Optional group this node belongs to.
        format_as_string: When ``True`` joins document content into a single
                         newline-separated string.  When ``False`` (default)
                         returns a list of ``{"content": ..., "meta_data": ...}``
                         dicts.
        knowledge:       Alias for ``vector_db`` — use whichever reads more
                         naturally at the call site.
        **kwargs:        Extra keyword arguments forwarded to :class:`~base.Node`.
                         ``input_schema`` and ``output_schema`` may be overridden
                         this way.
    """
    target = knowledge or vector_db
    if target is None:
        raise ValueError(
            "Either 'knowledge' or 'vector_db' must be provided to retrieval_node."
        )

    def handler(state: dict[str, Any]) -> NodeResult:
        query = state.get(query_key)
        if not query:
            return NodeResult(
                node_id=id,
                status="failed",
                error=f"Missing query key {query_key!r} in state.",
                state_updates={},
            )

        try:
            docs = target.search(query=query, limit=limit)
        except Exception as exc:
            return NodeResult(
                node_id=id,
                status="failed",
                error=f"Knowledge retrieval search failed: {exc}",
                state_updates={},
            )

        if format_as_string:
            formatted: Any = "\n\n".join(doc.content for doc in docs)
        else:
            formatted = [
                {
                    "content": doc.content,
                    "meta_data": getattr(
                        doc, "meta_data", getattr(doc, "metadata", {})
                    ),
                }
                for doc in docs
            ]

        return NodeResult(
            node_id=id,
            status="succeeded",
            state_updates={output_key: formatted},
        )

    node_input_schema = kwargs.pop(
        "input_schema", {"type": "object", "required": [query_key]}
    )
    node_output_schema = kwargs.pop(
        "output_schema", {"type": "object", "required": [output_key]}
    )

    return Node(
        id=id,
        name=name or id,
        description=description,
        handler=handler,
        input_schema=node_input_schema,
        output_schema=node_output_schema,
        group=group,
        **kwargs,
    )
