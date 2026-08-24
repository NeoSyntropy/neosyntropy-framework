from typing import Any, Dict
from neosyntropy.monitor.base import MonitorObserver, AsyncMonitorObserver

class NodeObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="node_observer", backend_url=backend_url)

    def log_node_execution(self, node_id: str, status: str):
        self.log_event("node_execution", {"node_id": node_id, "status": status})

class AsyncNodeObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="node_observer", backend_url=backend_url)

    async def log_node_execution(self, node_id: str, status: str):
        await self.log_event("node_execution", {"node_id": node_id, "status": status})

class ReasoningStepObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="reasoning_step_observer", backend_url=backend_url)

    def log_reasoning_step(self, step_id: str, instruction: str, tools: list[str]):
        self.log_event("reasoning_step", {"step_id": step_id, "instruction": instruction, "tools": tools})

class AsyncReasoningStepObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="reasoning_step_observer", backend_url=backend_url)

    async def log_reasoning_step(self, step_id: str, instruction: str, tools: list[str]):
        await self.log_event("reasoning_step", {"step_id": step_id, "instruction": instruction, "tools": tools})
