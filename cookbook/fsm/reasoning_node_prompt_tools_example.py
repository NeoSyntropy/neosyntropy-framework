"""FSM cookbook: prompt-driven reasoning with allow-listed tools.

Run::

    python cookbook/fsm/reasoning_node_prompt_tools_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    FSM,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    ToolRegistry,
    tool,
)

TESTS_ENV_PATH = Path(__file__).resolve().parents[2] / "tests" / ".env"
DEFAULT_API_URL = "http://127.0.0.1:8000"


def _load_tests_env() -> None:
    if not TESTS_ENV_PATH.is_file():
        return
    for raw in TESTS_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Copy tests/.env.example to tests/.env and fill values."
        )
    return value


def _provider() -> str:
    return os.environ.get("NEOSYNTROPY_PROVIDER", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _client_for_example() -> Client:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", DEFAULT_API_URL).strip()
        or DEFAULT_API_URL,
    )
    stamp = int(time.time())
    project = client.create_project(
        "FSM reasoning node cookbook",
        f"fsm-reasoning-node-prompt-{stamp}",
        description="Live cookbook run for a reasoning node with tools",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


class SupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str


class RoutingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lane: str
    confidence: float
    summary: str


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    routing_tools = ToolRegistry()

    @tool(registry=routing_tools)
    def billing_tool(args: SupportRequest) -> dict:
        """Return a billing-focused signal for the request."""
        text = args.intent.lower()
        confidence = 0.86 if any(token in text for token in ["bill", "charge", "refund"]) else 0.42
        print(f"[billing_tool] {args.intent!r} -> billing")
        return {"lane": "billing", "confidence": confidence, "evidence": "billing keywords"}

    @tool(registry=routing_tools)
    def shipping_tool(args: SupportRequest) -> dict:
        """Return a shipping-focused signal for the request."""
        text = args.intent.lower()
        confidence = 0.86 if any(token in text for token in ["ship", "delivery", "package"]) else 0.42
        print(f"[shipping_tool] {args.intent!r} -> shipping")
        return {"lane": "shipping", "confidence": confidence, "evidence": "shipping keywords"}

    @tool(registry=routing_tools)
    def returns_tool(args: SupportRequest) -> dict:
        """Return a returns-focused signal for the request."""
        text = args.intent.lower()
        confidence = 0.86 if any(token in text for token in ["return", "refund", "exchange"]) else 0.42
        print(f"[returns_tool] {args.intent!r} -> returns")
        return {"lane": "returns", "confidence": confidence, "evidence": "returns keywords"}

    route = ReasoningNode(
        id="RouteIntent",
        input_schema=SupportRequest,
        tools=["billing_tool", "shipping_tool", "returns_tool"],
        prompt=(
            "Use the billing_tool, shipping_tool, and returns_tool signals to decide whether "
            "the user's intent is about billing, shipping, or returns. Then produce a "
            "RoutingSummary with lane, confidence, and summary."
        ),
        provider=_provider(),
        output_schema=RoutingSummary,
    )

    out_of_scope = SchemaNode(
        id="OutOfScope",
        input_schema=SupportRequest,
        output_schema=TextOutput,
        provider=_provider(),
        prompt="Politely refuse out-of-scope requests in one short sentence.",
        is_fallback=True,
    )

    fsm = FSM(
        entry=route,
        nodes=[route, out_of_scope],
        edges=[
            edge_deterministic("RouteIntent", "End"),
            edge_fallback("RouteIntent", "OutOfScope"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )

    result = fsm.run(
        SupportRequest(intent="My package is late and I need help with the shipment."),
        state={},
        client=client,
        tools=routing_tools,
    )
    print(f"final_state: {result.final_state}")
    print(f"rejected: {result.rejected} rejection: {result.rejection}")
    for step in result.steps:
        for item in step.results:
            print(
                f"Node: {item.node_id}, status={item.status}, "
                f"error={item.error}, Output: {item.output}"
            )
            for record in item.tool_calls:
                print(
                    f"  tool={record.tool} ok={record.ok} denied={record.denied} "
                    f"error={record.error} result={record.result}"
                )
    return result


if __name__ == "__main__":
    main()
