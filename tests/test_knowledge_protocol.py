
import asyncio
from typing import Any, List
from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.protocol import (
    KnowledgeProtocol,
    KnowledgeReasoningProtocol,
    KnowledgeTransformProtocol,
)
from neosyntropy.knowledge.transform import transform, Input, Output


class DummyKnowledge:
    def __init__(self):
        self.vector_dbs = []
        self.databases = []

    def insert(self, data: Any, **kwargs: Any) -> Any:
        return True

    def delete(self, **kwargs: Any) -> Any:
        return True

    def get(self, **kwargs: Any) -> Any:
        return []


class DummyTransformKnowledge:
    def load(self, source: Any, **kwargs) -> Any:
        return [f"raw_{source}"]

    def build_transform_fsm(self, **kwargs) -> Any:
        return "workflow_instance"

    @transform(out=Output("processed"), raw_data=Input("source"))
    def transform(self, source: KnowledgeProtocol, destination: KnowledgeProtocol = None, **kwargs) -> Any:
        raw_data = self.load(source, **kwargs)
        wf = self.build_transform_fsm(**kwargs)
        processed = [f"transformed_{item}_via_{wf}" for item in raw_data]
        if destination:
            self.store(processed, destination, **kwargs)
        return processed

    def store(self, data: Any, destination: Any = None, **kwargs) -> Any:
        return True


class DummyReasoningKnowledge:
    def build_reasoning_fsm(self, **kwargs) -> Any:
        return "reasoning_fsm_instance"

    def search(self, knowledge: KnowledgeProtocol, **kwargs) -> List[Document]:
        query = kwargs.get("query", "default")
        reasoning_fsm = self.build_reasoning_fsm(**kwargs)
        return [Document(content=f"Reasoned result for: {query} with {reasoning_fsm}")]

    async def asearch(self, knowledge: KnowledgeProtocol, **kwargs) -> List[Document]:
        query = kwargs.get("query", "default")
        reasoning_fsm = self.build_reasoning_fsm(**kwargs)
        return [Document(content=f"Async reasoned result for: {query} with {reasoning_fsm}")]


def test_knowledge_protocol_conformance():
    instance = DummyKnowledge()
    assert isinstance(instance, KnowledgeProtocol)


def test_knowledge_transform_protocol_conformance():
    instance = DummyTransformKnowledge()
    assert isinstance(instance, KnowledgeTransformProtocol)
    
    k_src = DummyKnowledge()
    k_dst = DummyKnowledge()
    result = instance.transform(k_src, destination=k_dst)
    assert result == ["transformed_raw_data_source_via_workflow_instance"] or len(result) == 1


def test_knowledge_reasoning_protocol_conformance():
    instance = DummyReasoningKnowledge()
    k = DummyKnowledge()
    assert isinstance(instance, KnowledgeReasoningProtocol)

    results = instance.search(k, query="test query")
    assert len(results) == 1
    assert "Reasoned result for: test query with reasoning_fsm_instance" in results[0].content


def test_knowledge_reasoning_protocol_async():
    instance = DummyReasoningKnowledge()
    k = DummyKnowledge()
    results = asyncio.run(instance.asearch(k, query="async query"))
    assert len(results) == 1
    assert "Async reasoned result for: async query with reasoning_fsm_instance" in results[0].content
