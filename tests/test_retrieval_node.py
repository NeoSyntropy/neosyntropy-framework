import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from neosyntropy.core.node import retrieval_node
from neosyntropy.vectordb.base import VectorDb
from neosyntropy.knowledge.document.base import Document

def test_retrieval_node_success():
    """Test that the retrieval node correctly extracts query and injects docs."""
    docs = [
        Document(id="1", content="Alpha is the first.", meta_data={"source": "doc1"}),
        Document(id="2", content="Beta is the second.", meta_data={"source": "doc2"})
    ]
    
    db = MagicMock(spec=VectorDb)
    db.search.return_value = docs

    node = retrieval_node(
        id="FetchContext",
        vector_db=db,
        query_key="search_query",
        output_key="context",
        limit=2
    )

    
    # Simulate execution with the state containing the query
    initial_state = {"search_query": "What is alpha and beta?"}
    result = node.handler(initial_state)
    
    assert result.status == "succeeded"
    assert "context" in result.state_updates
    
    injected_context = result.state_updates["context"]
    assert len(injected_context) == 2
    assert injected_context[0]["content"] == "Alpha is the first."
    assert injected_context[0]["meta_data"] == {"source": "doc1"}
    
    # Assert VectorDb was called correctly
    db.search.assert_called_once_with(query="What is alpha and beta?", limit=2)


def test_retrieval_node_missing_query():
    """Test that the node fails if the query key is missing in the state."""
    db = MagicMock(spec=VectorDb)
    
    node = retrieval_node(
        id="FetchContext",
        vector_db=db,
        query_key="search_query",
        output_key="context"
    )
    
    initial_state = {"wrong_key": "query here"}
    result = node.handler(initial_state)
    
    assert result.status == "failed"
    assert "Missing query key" in result.error


def test_retrieval_node_format_as_string():
    """Test that the node can format output as a single string block."""
    docs = [
        Document(id="1", content="Paragraph 1", meta_data={}),
        Document(id="2", content="Paragraph 2", meta_data={})
    ]
    db = MagicMock(spec=VectorDb)
    db.search.return_value = docs
    
    node = retrieval_node(
        id="FetchContext",
        vector_db=db,
        query_key="search_query",
        output_key="context",
        format_as_string=True
    )
    
    result = node.handler({"search_query": "test"})
    assert result.status == "succeeded"
    
    injected_context = result.state_updates["context"]
    assert isinstance(injected_context, str)
    assert injected_context == "Paragraph 1\n\nParagraph 2"


def test_retrieval_node_with_knowledge_protocol():
    """Test retrieval_node using KnowledgeRetrievalProtocol."""
    from neosyntropy.knowledge.protocol import KnowledgeRetrievalProtocol
    
    class MockKnowledge:
        def build_retrieval_fsm(self, knowledge=None, **kwargs):
            return "retrieval_fsm"
            
        def search(self, query: str, limit: int = 5, **kwargs):
            wf = self.build_retrieval_fsm(**kwargs)
            return [Document(id="k1", content=f"Retrieved '{query}' via {wf}", meta_data={"source": "knowledge"})]
            
        async def asearch(self, query: str, **kwargs):
            return self.search(query, **kwargs)

    knowledge_inst = MockKnowledge()
    assert isinstance(knowledge_inst, KnowledgeRetrievalProtocol)
    
    node = retrieval_node(
        id="RetrievalFetch",
        knowledge=knowledge_inst,
        query_key="user_query",
        output_key="retrieved_context",
        limit=3
    )

    result = node.handler({"user_query": "syntropy dynamics"})
    assert result.status == "succeeded"
    assert "retrieved_context" in result.state_updates
    
    res_docs = result.state_updates["retrieved_context"]
    assert len(res_docs) == 1
    assert res_docs[0]["content"] == "Retrieved 'syntropy dynamics' via retrieval_fsm"
    assert res_docs[0]["meta_data"] == {"source": "knowledge"}

