"""Safe, best-effort observability contracts for framework runs."""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .backend import BackendClient
from .core.graph import FSM

if TYPE_CHECKING:
    from .tools.registry import ToolRegistry


def tool_catalog(tools: ToolRegistry | Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Serialize registered tools for console inspection (no handlers)."""
    if tools is None:
        return []
    registered = getattr(tools, "tools", tools)
    if not isinstance(registered, Mapping):
        return []
    catalog: list[dict[str, Any]] = []
    for key, item in registered.items():
        name = getattr(item, "name", None) or (key if isinstance(key, str) else None)
        if not name:
            continue
        catalog.append(
            {
                "name": name,
                "description": getattr(item, "description", "") or "",
                "input_schema": getattr(item, "json_schema", None)
                or getattr(item, "input_schema", None)
                or {},
                "output_schema": getattr(item, "return_schema", None)
                or getattr(item, "output_schema", None),
            }
        )
    return catalog


def graph_manifest(
    graph: FSM,
    tools: ToolRegistry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the visualization graph shape safe to send off-process.

    Includes the fields the console needs to render node cards (description,
    prompt, tool allow-list names, node output schema), the graph's entry
    contract, plus a tool catalog with input/output schemas. Executable /
    sensitive fields stay excluded: handlers, guards, providers, metadata,
    and transition policy are never represented in this manifest.
    """
    return {
        "schema_version": 1,
        # What the workflow itself takes in at Start.
        "input_schema": graph.input_schema,
        "nodes": [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "prompt": item.prompt,
                "mode": item.mode,
                "tools": list(item.tools),
                "input_schema": item.input_schema,
                "output_schema": item.output_schema,
                "group": item.group,
                "is_fallback": item.is_fallback,
            }
            for item in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "target_kind": edge.target_kind,
            }
            for edge in graph.edges
        ],
        "groups": [{"name": group.name} for group in graph.groups.values()],
        "tools": tool_catalog(tools),
    }


def control_graph_manifest(graph: FSM) -> dict[str, Any]:
    """Minimum graph definition for backend-owned control runs.

    Includes structure the backend needs to validate and commit transitions.
    Handlers, prompts, tools, guards, and providers are excluded.
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
                "input_schema": item.input_schema,
                "output_schema": item.output_schema,
            }
            for item in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "target_kind": edge.target_kind,
            }
            for edge in graph.edges
        ],
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
    }


@runtime_checkable
class RunObserver(Protocol):
    """Pluggable sink for control-lifecycle telemetry.

    ``input`` and ``output`` carry the run/step debug payloads (intent, state
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
