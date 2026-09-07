from importlib import import_module
from typing import Any

__all__ = [
    "Knowledge",
    "FileSystemKnowledge",
    "KnowledgeProtocol",
    "KnowledgeTransformProtocol",
    "KnowledgeRetrievalProtocol",
    "transform",
    "Input",
    "Output",
]


def __getattr__(name: str) -> Any:
    if name == "Knowledge":
        from neosyntropy.knowledge.knowledge import Knowledge

        return Knowledge
    if name == "FileSystemKnowledge":
        from neosyntropy.knowledge.filesystem import FileSystemKnowledge

        return FileSystemKnowledge
    if name in {
        "KnowledgeProtocol",
        "KnowledgeTransformProtocol",
        "KnowledgeRetrievalProtocol",
    }:
        return getattr(import_module("neosyntropy.knowledge.protocol"), name)
    if name in {"transform", "Input", "Output"}:
        return getattr(import_module("neosyntropy.knowledge.transform"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

