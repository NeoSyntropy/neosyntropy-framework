import asyncio
import inspect
import json
from collections.abc import Mapping
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from neosyntropy.backend import BackendClient


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


# ---------------------------------------------------------------------------
# Concept manifest contract
# ---------------------------------------------------------------------------

class ConceptManifestProvider(Protocol):
    """Protocol that every observable concept object should satisfy.

    Implementing this protocol is the contract that tells the monitoring layer
    what structured data to send to the server when the concept is registered.
    The ``concept_type`` string must match the backend API path segment
    (e.g. ``"knowledge"``, ``"vector_db"``, ``"worker"``).
    """

    concept_type: str

    def manifest(self) -> dict[str, Any]:
        """Return the serialisable snapshot to store on the backend."""
        ...


class BackendConceptReporter:
    """Generic reporter for non-run concepts (knowledge, vector_db, workers, …).

    Mirrors :class:`~neosyntropy.monitor.run.observer.BackendTelemetryReporter`
    for run telemetry, but handles the registration of concept manifests that
    exist independently of any individual run (e.g. a knowledge base defined
    once and reused across many runs).

    Usage::

        reporter = BackendConceptReporter(client, project_id="proj_abc")
        await reporter.concept_registered("knowledge", knowledge_manifest(kb))
    """

    def __init__(self, client: "BackendClient", project_id: str) -> None:
        self._client = client
        self._project_id = project_id

    async def concept_registered(
        self,
        concept_type: str,
        manifest: dict[str, Any],
    ) -> str | None:
        """Upsert a concept manifest on the backend.

        Calls ``BackendClient.register_concept`` which POSTs to
        ``/api/v1/observability/projects/{project_id}/{concept_type}``.
        Returns the server-assigned concept id, or ``None`` on failure.
        """
        register_fn = getattr(self._client, "register_concept", None)
        if register_fn is None:
            return None
        try:
            result = register_fn(
                project_id=self._project_id,
                concept_type=concept_type,
                manifest=manifest,
            )
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:
            return None

    async def event(
        self,
        concept_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Record a lifecycle event for a previously registered concept."""
        event_fn = getattr(self._client, "concept_event", None)
        if event_fn is None:
            return
        try:
            result = event_fn(
                concept_id=concept_id,
                event_type=event_type,
                payload=bounded_event_payload(payload),
            )
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
