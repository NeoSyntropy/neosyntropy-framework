import asyncio
import inspect
import json
from collections.abc import Mapping
from typing import Any, Dict, Optional, Protocol, runtime_checkable

class MonitorObserver:
    """
    Base class for monitoring components in NeoSyntropy.
    Observers can send events to the NeoSyntropy backend or a local database.
    """
    def __init__(self, name: str, backend_url: Optional[str] = None):
        self.name = name
        self.backend_url = backend_url

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Logs an event to the default monitoring backend."""
        pass

class AsyncMonitorObserver:
    """
    Async base class for monitoring components in NeoSyntropy.
    """
    def __init__(self, name: str, backend_url: Optional[str] = None):
        self.name = name
        self.backend_url = backend_url

    async def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Logs an event to the default monitoring backend asynchronously."""
        pass


@runtime_checkable
class RunObserver(Protocol):
    """Pluggable sink for control-lifecycle telemetry.

    ``input`` and ``output`` carry the run/step debug payloads (run input, state
    snapshots, node results) when the manager captures payloads.
    """

    async def run_started(
        self,
        *,
        request_id: str,
        initial_state: str,
        manifest: Mapping[str, Any],
        input: Mapping[str, Any] | None = None,
    ) -> str | None: ...

    async def event(
        self, run_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None: ...

    async def run_finished(
        self,
        run_id: str,
        *,
        status: str,
        final_state: str,
        output: Mapping[str, Any] | None = None,
    ) -> None: ...


# Kept below the backend's default 64 KiB per-event cap so debug-heavy events
# are truncated client-side instead of rejected (and lost) server-side.
MAX_EVENT_PAYLOAD_BYTES = 49_152


def bounded_event_payload(
    payload: dict[str, Any], limit: int = MAX_EVENT_PAYLOAD_BYTES
) -> dict[str, Any]:
    """Trim oversized debug fields so the event is stored, not dropped."""

    def encoded_size(data: dict[str, Any]) -> int:
        return len(json.dumps(data, separators=(",", ":"), default=str).encode())

    if encoded_size(payload) <= limit:
        return payload
    trimmed = dict(payload)
    for key in ("output", "input"):
        if key in trimmed:
            trimmed[key] = {
                "truncated": True,
                "reason": "payload exceeded telemetry size limit",
            }
            if encoded_size(trimmed) <= limit:
                break
    return trimmed


async def best_effort_call(
    operation: Any, *, timeout: float
) -> Any:
    """Await one observer operation without allowing it to affect execution."""
    try:
        result = operation
        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=timeout)
        return result
    except Exception:
        return None
