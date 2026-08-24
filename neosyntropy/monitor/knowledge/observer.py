from typing import Any, Dict
from neosyntropy.monitor.base import MonitorObserver, AsyncMonitorObserver

class KnowledgeObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="knowledge_observer", backend_url=backend_url)

    def log_knowledge_query(self, knowledge_id: str, query: str):
        self.log_event("knowledge_query", {"knowledge_id": knowledge_id, "query": query})

    def log_knowledge_retrieval(self, knowledge_id: str, document_ids: list[str]):
        self.log_event("knowledge_retrieval", {"knowledge_id": knowledge_id, "document_ids": document_ids})

class AsyncKnowledgeObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="knowledge_observer", backend_url=backend_url)

    async def log_knowledge_query(self, knowledge_id: str, query: str):
        await self.log_event("knowledge_query", {"knowledge_id": knowledge_id, "query": query})

    async def log_knowledge_retrieval(self, knowledge_id: str, document_ids: list[str]):
        await self.log_event("knowledge_retrieval", {"knowledge_id": knowledge_id, "document_ids": document_ids})
