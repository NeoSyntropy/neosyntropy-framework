from typing import Any, Dict
from neosyntropy.monitor.base import MonitorObserver, AsyncMonitorObserver

class FunctionObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="function_observer", backend_url=backend_url)

    def log_function_call(self, function_name: str, args: Dict[str, Any]):
        self.log_event("function_call", {"function_name": function_name, "args": args})

    def log_function_return(self, function_name: str, result: Any):
        self.log_event("function_return", {"function_name": function_name, "result": result})

class AsyncFunctionObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="function_observer", backend_url=backend_url)

    async def log_function_call(self, function_name: str, args: Dict[str, Any]):
        await self.log_event("function_call", {"function_name": function_name, "args": args})

    async def log_function_return(self, function_name: str, result: Any):
        await self.log_event("function_return", {"function_name": function_name, "result": result})
