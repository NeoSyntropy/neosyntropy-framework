import pytest
from pathlib import Path
from neosyntropy.knowledge.filesystem import FileSystemKnowledge
from neosyntropy.knowledge.protocol import KnowledgeRetrievalProtocol, KnowledgeTransformProtocol
from neosyntropy.core.graph import Workflow, FSM
from neosyntropy.core.node import ReasoningStep

def test_filesystem_knowledge_protocols(tmp_path: Path):
    """Test that FileSystemKnowledge conforms to the necessary protocols."""
    fs_knowledge = FileSystemKnowledge(base_dir=str(tmp_path))
    
    assert isinstance(fs_knowledge, KnowledgeRetrievalProtocol)
    assert isinstance(fs_knowledge, KnowledgeTransformProtocol)
    
def test_filesystem_knowledge_search(tmp_path: Path):
    """Test backwards-compatibility of retrieve mapping to search."""
    (tmp_path / "test.txt").write_text("apple banana cherry")
    fs_knowledge = FileSystemKnowledge(base_dir=str(tmp_path))
    
    docs = fs_knowledge.search("banana")
    assert len(docs) == 1
    assert "banana" in docs[0].content
    
    # Check backwards-compatibility alias
    docs_old = fs_knowledge.retrieve("banana")
    assert len(docs_old) == 1
    assert "banana" in docs_old[0].content

def test_filesystem_knowledge_build_retrieval_fsm(tmp_path: Path):
    """Test building a multi-step retrieval FSM with ReasoningStep."""
    fs_knowledge = FileSystemKnowledge(base_dir=str(tmp_path))
    
    # Use default steps
    workflow = fs_knowledge.build_retrieval_fsm()
    assert isinstance(workflow, FSM)
    
    # Custom steps
    custom_steps = [
        ReasoningStep("Custom test instruction 1", tools=["list_files"]),
        ReasoningStep("Custom test instruction 2", tools=["get_file"])
    ]
    custom_workflow = fs_knowledge.build_retrieval_fsm(steps=custom_steps)
    assert isinstance(custom_workflow, FSM)
    assert len(custom_workflow.nodes) == 3
    
def test_filesystem_knowledge_transform_protocol(tmp_path: Path):
    """Test basic structure for transform protocol methods."""
    (tmp_path / "data.txt").write_text("some content")
    fs_knowledge = FileSystemKnowledge(base_dir=str(tmp_path))
    
    data = fs_knowledge.load()
    assert len(data) == 1
    
    wf = fs_knowledge.build_transform_fsm()
    assert isinstance(wf, FSM)
    
    processed = fs_knowledge.transform()
    assert len(processed) == 1
    assert "content" in processed[0]
