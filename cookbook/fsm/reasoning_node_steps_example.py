"""FSM cookbook: step-based reasoning with `ReasoningStep`.

Run::

    python cookbook/fsm/reasoning_node_steps_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import Client, ReasoningNode, ReasoningStep, ToolRegistry, tool

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
        "FSM reasoning steps cookbook",
        f"fsm-reasoning-node-steps-{stamp}",
        description="Live cookbook run for a step-based reasoning flow",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class DecisionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str
    summary: str
    next_action: str


class IntentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class PolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    registry = ToolRegistry()

    @tool(registry=registry)
    def find_intent(args: IntentArgs) -> dict:
        """Classify the request into a simple support lane."""
        text = args.text.lower()
        if "refund" in text or "charge" in text:
            lane = "billing"
        elif "ship" in text or "delivery" in text:
            lane = "shipping"
        else:
            lane = "general"
        print(f"[find_intent] {args.text!r} -> {lane}")
        return {"lane": lane}

    @tool(registry=registry)
    def find_policy(args: PolicyArgs) -> dict:
        """Return a tiny policy hint for the identified lane."""
        text = args.text.lower()
        if "address" in text:
            policy = "update_address_before_dispatch"
            next_action = "contact_ops"
        elif "refund" in text:
            policy = "refund_window_check"
            next_action = "review_order"
        else:
            policy = "general_support"
            next_action = "respond_normally"
        print(f"[find_policy] {args.text!r} -> {policy}")
        return {"policy": policy, "next_action": next_action}

    fsm = ReasoningNode(
        id="SupportDecision",
        input_schema=CaseInput,
        steps=[
            ReasoningStep(
                instruction="Use `find_intent` to classify the request into a support lane.",
                tools=["find_intent"],
            ),
            ReasoningStep(
                instruction="Use `find_policy` to pull the matching support rule and next action.",
                tools=["find_policy"],
            ),
            ReasoningStep(
                instruction=(
                    "Using the lane and policy evidence, extract a deterministic decision summary "
                    "with decision, summary, and next_action."
                )
            ),
        ],
        provider=_provider(),
        output_schema=DecisionSummary,
    )

    result = fsm.run(
        CaseInput(text="I need to change my delivery address before the package is shipped."),
        state={},
        client=client,
        tools=registry,
    )
    print(f"final_state: {result.final_state}")
    for step in result.steps:
        for item in step.results:
            print(f"Node: {item.node_id}, Output: {item.output}")


if __name__ == "__main__":
    main()
