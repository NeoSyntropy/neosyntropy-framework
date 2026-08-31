"""Unit tests for ReasoningNode ``tools_executed`` telemetry."""

from __future__ import annotations

import asyncio
from typing import Any

from neosyntropy.control.manager import (
    ControlManager,
    _executed_tool_payload,
    _MAX_TOOL_RESULT_CHARS,
)
from neosyntropy.core.models import NodeResult, ToolCallRecord


class _RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        self.events.append((run_id, event_type, dict(payload)))


def test_executed_tool_payload_truncates_large_result() -> None:
    huge = "x" * (_MAX_TOOL_RESULT_CHARS + 50)
    payload = _executed_tool_payload(
        ToolCallRecord(
            tool="lookup",
            arguments={"id": "1"},
            ok=True,
            result=huge,
            latency_ms=1.5,
        )
    )
    assert payload["tool"] == "lookup"
    assert payload["ok"] is True
    assert payload["result"]["truncated"] is True
    assert payload["result"]["original_chars"] == len(huge)
    assert len(payload["result"]["preview"]) == _MAX_TOOL_RESULT_CHARS


def test_observe_tools_executed_skips_denied_and_emits_executed() -> None:
    observer = _RecordingObserver()
    manager = ControlManager.__new__(ControlManager)
    manager.observer = observer
    manager.telemetry_timeout = 1.0

    results = [
        NodeResult(
            node_id="ReasoningNode",
            status="succeeded",
            output="notes",
            tool_calls=[
                ToolCallRecord(
                    tool="lookup_customer_account",
                    arguments={"customer_id": "cust_12345"},
                    ok=True,
                    result={"vip_tier": "Gold"},
                    latency_ms=0.2,
                ),
                ToolCallRecord(
                    tool="not_allowlisted",
                    denied=True,
                    error="not allowed",
                ),
            ],
        ),
        NodeResult(
            node_id="Summarize",
            status="succeeded",
            output={"decision": "approve"},
            tool_calls=[],
        ),
    ]

    asyncio.run(
        manager._observe_tools_executed("run-1", step=0, results=results)
    )

    tools_events = [event for event in observer.events if event[1] == "tools_executed"]
    assert len(tools_events) == 1
    run_id, event_type, payload = tools_events[0]
    assert run_id == "run-1"
    assert event_type == "tools_executed"
    assert payload["step"] == 0
    assert payload["node_id"] == "ReasoningNode"
    assert [item["tool"] for item in payload["tools"]] == ["lookup_customer_account"]
    assert payload["tools"][0]["ok"] is True
    assert payload["tools"][0]["arguments"] == {"customer_id": "cust_12345"}


def test_observe_tools_executed_noop_without_observer() -> None:
    manager = ControlManager.__new__(ControlManager)
    manager.observer = None
    manager.telemetry_timeout = 1.0
    asyncio.run(
        manager._observe_tools_executed(
            "run-1",
            step=0,
            results=[
                NodeResult(
                    node_id="ReasoningNode",
                    tool_calls=[
                        ToolCallRecord(tool="lookup", ok=True, result={"a": 1}),
                    ],
                )
            ],
        )
    )
