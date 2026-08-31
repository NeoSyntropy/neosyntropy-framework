"""Validation cookbook — group level: semantic gate on a group's execution path.

A ``SemanticGroupPathValidator`` adds an LLM-backed ``ValidationResult`` node
at the end of a group's internal path.  It auto-registers into the group and
— when ``after=`` is supplied — wires the final internal edge for you.

Group structure::

    ┌── triage ─────────────────────────────────────────────────────────┐
    │  ExtractTicket → ClassifyUrgency → ValidateTriage (LLM gate)      │
    └───────────────────────────────────────────────────────────────────┘

FSM::

    entry: ExtractTicket
    ValidateTriage → End
    OutOfScope (fallback)

The validator's output (``{"valid": bool, "reason": str}``) is available in
``NodeResult.output`` for each run step so you can inspect it after the run.

Run::

    python cookbook/validation/group_path_validation_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    FSM,
    Group,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.core.validation import SemanticGroupPathValidator

TESTS_ENV_PATH = Path(__file__).resolve().parents[3] / "tests" / ".env"
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
    return (
        os.environ.get("NEOSYNTROPY_PROVIDER", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash"
    )


def _client_for_example() -> Client:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", DEFAULT_API_URL).strip()
        or DEFAULT_API_URL,
    )
    stamp = int(time.time())
    project = client.create_project(
        "Validation cookbook — group level",
        f"validation-group-{stamp}",
        description="Semantic path validator on a triage group",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class TicketDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_name: str
    email: str | None = None
    topic: str


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    urgency: Literal["low", "normal", "high"]
    suggested_team: str


# ---------------------------------------------------------------------------
# FSM builder
# ---------------------------------------------------------------------------

def build_fsm(provider: str) -> FSM:
    triage_group = Group(name="triage")

    # Step 1 — extract structured ticket details
    extract = SchemaNode(
        id="ExtractTicket",
        input_schema=TicketInput,
        output_schema=TicketDetails,
        provider=provider,
        prompt=(
            "Extract the customer's name, email address if present, "
            "and the main support topic from the message."
        ),
    )
    triage_group.add_node(extract)

    # Step 2 — classify urgency and route to a team
    classify = SchemaNode(
        id="ClassifyUrgency",
        input_schema=TicketDetails,
        output_schema=TriageResult,
        provider=provider,
        prompt=(
            "Classify the ticket urgency (low / normal / high) based on the "
            "topic and any implied severity.  Suggest the most appropriate "
            "support team: billing, technical, or general."
        ),
    )
    triage_group.add_node(classify)

    # Wire the internal group edge ExtractTicket → ClassifyUrgency
    triage_group.add_edge("ExtractTicket", "ClassifyUrgency")

    # Step 3 — SemanticGroupPathValidator: LLM judge over the group path.
    #   • auto-registers into triage_group
    #   • after="ClassifyUrgency" wires the edge ClassifyUrgency → ValidateTriage
    #   • output_schema is always ValidationResult {"valid": bool, "reason": str}
    SemanticGroupPathValidator(
        "ValidateTriage",
        group=triage_group,
        after="ClassifyUrgency",
        input_schema=TriageResult,
        provider=provider,
        prompt=(
            "Review the triage result. "
            "Return valid=false if urgency is missing, the suggested_team is "
            "not one of billing/technical/general, or if the topic is clearly "
            "unrelated to any support category."
        ),
    )

    # Fallback outside the group
    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=TicketInput,
        output_schema=TextOutput,
        provider=provider,
        prompt="Politely decline messages that are not support tickets.",
    )

    return FSM(
        entry="ExtractTicket",
        nodes=[out_of_scope],
        groups=[triage_group],
        edges=[
            # After the group validator, continue to End
            edge_deterministic("ValidateTriage", "End"),
            edge_fallback("ExtractTicket", "OutOfScope"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    fsm = build_fsm(_provider())

    tickets = [
        "Hi, I'm James. My account is locked and I can't reset my password — "
        "I need this fixed urgently, it's blocking my whole team.",
        # Off-topic input that should fail validation
        "What is the capital of France?",
    ]

    for text in tickets:
        print(f"\n--- ticket: {text[:70]!r} ---")
        result = fsm.run(TicketInput(text=text), client=client)
        print(f"final_state: {result.final_state}")
        for step in result.steps:
            for item in step.results:
                label = (
                    f"  node={item.node_id}  status={item.status}"
                )
                if item.node_id == "ValidateTriage":
                    label += f"  output={item.output}"
                print(label)


if __name__ == "__main__":
    main()
