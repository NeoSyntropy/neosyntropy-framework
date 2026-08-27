"""FSM cookbook: prompt-driven reasoning with allow-listed tools.

Run::

    python cookbook/fsm/reasoning_node_prompt_tools_example.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    ReasoningStep,
    SchemaStep,
    ToolRegistry,
    tool,
    workflow,
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


class StochasticArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    stochastic_tools = ToolRegistry()

    @tool(registry=stochastic_tools)
    def stochastic(args: StochasticArgs) -> dict:
        """Return a noisy but repeatable hint about the request."""
        options = ["billing", "shipping", "returns"]
        chosen = options[len(args.intent) % len(options)]
        confidence = round(0.55 + (len(args.intent) % 5) * 0.07, 2)
        print(f"[stochastic] {args.intent!r} -> {chosen}")
        return {"hint": chosen, "confidence": confidence, "sample": random.randint(1, 3)}

    @workflow(
        input_schema=SupportRequest,
        steps=[
            ReasoningStep(
                instruction=(
                    "Use the `stochastic` tool to decide whether the user's intent is about "
                    "billing, shipping, or returns."
                ),
                tools=["stochastic"],
            ),
            SchemaStep(
                instruction=(
                    "Summarize the chosen lane, confidence, and reasoning into lane, "
                    "confidence, and summary fields."
                )
            ),
        ],
        client=client,
        tools=stochastic_tools,
        provider=_provider(),
    )
    def route(params: RoutingSummary) -> str:
        return f"{params.lane} ({params.confidence}): {params.summary}"

    result = route(intent="My package is late and I need help with the shipment.")
    print(result)


if __name__ == "__main__":
    main()
