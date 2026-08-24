from collections.abc import Mapping
from typing import Any, Dict

from neosyntropy.backend import BackendClient
from neosyntropy.monitor.base import (
    MonitorObserver,
    AsyncMonitorObserver,
    RunObserver as TelemetryRunObserver,
    MAX_EVENT_PAYLOAD_BYTES,
    bounded_event_payload,
)

class RunObserver(MonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="run_observer", backend_url=backend_url)

    def log_run_start(self, run_id: str, metadata: Dict[str, Any]):
        self.log_event("run_start", {"run_id": run_id, **metadata})

    def log_run_end(self, run_id: str, status: str):
        self.log_event("run_end", {"run_id": run_id, "status": status})

class AsyncRunObserver(AsyncMonitorObserver):
    def __init__(self, backend_url: str = None):
        super().__init__(name="run_observer", backend_url=backend_url)

    async def log_run_start(self, run_id: str, metadata: Dict[str, Any]):
        await self.log_event("run_start", {"run_id": run_id, **metadata})

    async def log_run_end(self, run_id: str, status: str):
        await self.log_event("run_end", {"run_id": run_id, "status": status})

class BackendTelemetryReporter:
    """Run observer backed by NeoSyntropy's telemetry API."""

    def __init__(
        self,
        client: BackendClient,
        *,
        max_event_payload_bytes: int = MAX_EVENT_PAYLOAD_BYTES,
    ) -> None:
        self.client = client
        self.max_event_payload_bytes = max_event_payload_bytes
        self._sequences: dict[str, int] = {}

    async def run_started(
        self,
        *,
        request_id: str,
        initial_state: str,
        manifest: Mapping[str, Any],
        input: Mapping[str, Any] | None = None,
    ) -> str | None:
        run_id = await self.client.telemetry_run_started(
            request_id=request_id,
            initial_state=initial_state,
            manifest=dict(manifest),
            input=dict(input) if input is not None else None,
        )
        if run_id is not None:
            self._sequences[run_id] = 0
        return run_id

    async def event(
        self, run_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        sequence = self._sequences.get(run_id, 0) + 1
        self._sequences[run_id] = sequence
        await self.client.telemetry_event(
            run_id,
            event_type,
            bounded_event_payload(dict(payload), self.max_event_payload_bytes),
            external_id=f"{run_id}:{sequence}",
            sequence=sequence,
        )

    async def run_finished(
        self,
        run_id: str,
        *,
        status: str,
        final_state: str,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            await self.client.telemetry_run_finished(
                run_id,
                status=status,
                final_state=final_state,
                output=dict(output) if output is not None else None,
            )
        finally:
            self._sequences.pop(run_id, None)
