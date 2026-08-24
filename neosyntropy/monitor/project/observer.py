from typing import Any, Dict
from neosyntropy.monitor.base import MonitorObserver, AsyncMonitorObserver

class ProjectObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="project_observer", backend_url=backend_url)

    def log_project_created(self, project_id: str, metadata: Dict[str, Any]):
        self.log_event("project_created", {"project_id": project_id, **metadata})

class AsyncProjectObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="project_observer", backend_url=backend_url)

    async def log_project_created(self, project_id: str, metadata: Dict[str, Any]):
        await self.log_event("project_created", {"project_id": project_id, **metadata})
