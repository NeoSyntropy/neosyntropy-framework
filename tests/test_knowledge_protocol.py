import asyncio
from typing import Any, List
from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.protocol import (
    KnowledgeProtocol,
    KnowledgeReasoningProtocol,
    KnowledgeTransformProtocol,
)
from neosyntropy.knowledge.transform import transform, Input, Output


class DummyTransformKnowledge:
    def load(self, source: Any, **kwargs) -> Any:
        return [f"raw_{source}"]

    def build_transform_fsm(self, **kwargs) -> Any:
        return "workflow_instance"

    # Alias for backwards compatibility testing
    build_workflow = build_transform_fsm

    @transform(out=Output("processed"), raw_data=Input("source"))
    def transform(self, source: Any, destination: Any = None, **kwargs) -> Any:
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

    # Alias for backwards compatibility testing
    build_reasoning_workflow = build_reasoning_fsm

    def search(self, query: str, **kwargs) -> List[Document]:
        reasoning_fsm = self.build_reasoning_fsm(**kwargs)
        return [Document(content=f"Reasoned result for: {query} with {reasoning_fsm}")]

    async def asearch(self, query: str, **kwargs) -> List[Document]:
        reasoning_fsm = self.build_reasoning_fsm(**kwargs)
        return [Document(content=f"Async reasoned result for: {query} with {reasoning_fsm}")]


def test_knowledge_transform_protocol_conformance():
    instance = DummyTransformKnowledge()
    assert isinstance(instance, KnowledgeTransformProtocol)
    
    result = instance.transform("data_source", destination="db_dest")
    assert result == ["transformed_raw_data_source_via_workflow_instance"]


def test_knowledge_reasoning_protocol_conformance():
    instance = DummyReasoningKnowledge()
    assert isinstance(instance, KnowledgeReasoningProtocol)
    assert isinstance(instance, KnowledgeProtocol)

    results = instance.search("test query")
    assert len(results) == 1
    assert "Reasoned result for: test query with reasoning_fsm_instance" in results[0].content


def test_knowledge_reasoning_protocol_async():
    instance = DummyReasoningKnowledge()
    results = asyncio.run(instance.asearch("async query"))
    assert len(results) == 1
    assert "Async reasoned result for: async query with reasoning_fsm_instance" in results[0].content
