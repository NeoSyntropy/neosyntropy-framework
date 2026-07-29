"""Safe, best-effort observability contracts for framework runs."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .backend import BackendClient
from .core.graph import Graph


def graph_manifest(graph: Graph) -> dict[str, Any]:
    """Return the visualization-only graph shape safe to send off-process.

    Executable and potentially sensitive fields are deliberately excluded:
    prompts, handlers, guards, tools, providers, metadata, axiom code, and
    transition policy are never represented in this manifest.
    """
    return {
        "schema_version": 1,
        "nodes": [
            {
                "id": item.id,
                "name": item.name,
                "group": item.group,
                "is_fallback": item.is_fallback,
            }
            for item in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            }
            for edge in graph.edges
        ],
        "groups": [{"name": group.name} for group in graph.groups.values()],
    }


def control_graph_manifest(graph: Graph) -> dict[str, Any]:
    """Minimum graph definition for backend-owned control runs.

    Includes structure the backend needs to validate and commit transitions.
    Handlers, prompts, tools, guards, providers, and axiom code are excluded.
    """
    return {
        "schema_version": 1,
        "nodes": [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "prerequisites": list(item.prerequisites),
                "is_fallback": item.is_fallback,
                "group": item.group,
            }
            for item in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            }
            for edge in graph.edges
        ],
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
    }


@runtime_checkable
class RunObserver(Protocol):
    """Pluggable sink for sanitized control-lifecycle telemetry."""

    async def run_started(
        self, *, request_id: str, initial_state: str, manifest: Mapping[str, Any]
    ) -> str | None: ...

    async def event(
        self, run_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None: ...

    async def run_finished(
        self, run_id: str, *, status: str, final_state: str
    ) -> None: ...


class BackendTelemetryReporter:
    """Run observer backed by NeoSyntropy's telemetry API."""

    def __init__(self, client: BackendClient) -> None:
        self.client = client
        self._sequences: dict[str, int] = {}

    async def run_started(
        self, *, request_id: str, initial_state: str, manifest: Mapping[str, Any]
    ) -> str | None:
        run_id = await self.client.telemetry_run_started(
            request_id=request_id,
            initial_state=initial_state,
            manifest=dict(manifest),
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
            dict(payload),
            external_id=f"{run_id}:{sequence}",
            sequence=sequence,
        )

    async def run_finished(
        self, run_id: str, *, status: str, final_state: str
    ) -> None:
        try:
            await self.client.telemetry_run_finished(
                run_id, status=status, final_state=final_state
            )
        finally:
            self._sequences.pop(run_id, None)


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
