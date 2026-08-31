from typing import Any, Dict
from neosyntropy.monitor.base import MonitorObserver, AsyncMonitorObserver

class GraphObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="graph_observer", backend_url=backend_url)

    def log_graph_execution(self, graph_id: str, status: str):
        self.log_event("graph_execution", {"graph_id": graph_id, "status": status})

class AsyncGraphObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="graph_observer", backend_url=backend_url)

    async def log_graph_execution(self, graph_id: str, status: str):
        await self.log_event("graph_execution", {"graph_id": graph_id, "status": status})
