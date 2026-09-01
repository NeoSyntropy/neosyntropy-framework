"""KPI cookbook — group level: score a group's execution path.

A ``functional_group_path_kpi`` scores the **outcome of traversing a group's
internal nodes**.  It auto-registers into the group and — when ``after=`` is
supplied — wires the final internal edge for you.

The run **always continues** after the KPI node — no branching is needed.

Group structure::

    ┌── triage ──────────────────────────────────────────────────────────┐
    │  ExtractTicket → ClassifyUrgency → ScoreTriage (Python scorer)     │
    └────────────────────────────────────────────────────────────────────┘

FSM::

    entry: ExtractTicket
    ScoreTriage ──▶ End
    OutOfScope (fallback)

State after the run::

    state["triage_quality"]        — numeric score (0–1)
    state["triage_quality_reason"] — explanation string
    state["kpis"]                  — {"triage_quality": <score>}

Run::

    python cookbook/kpi/group_kpi_example.py
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
    NodeContext,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.core.kpi import functional_group_path_kpi
from neosyntropy.core.node.schemas import KpiResult

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
        "KPI cookbook — group level",
        f"kpi-group-{stamp}",
        description="Functional KPI scorer on a triage group path",
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
# Scoring rubric
# ---------------------------------------------------------------------------
#
# triage_quality score is a weighted average of three signals:
#   50%  urgency is one of the expected values (structural correctness)
#   30%  suggested_team is one of the expected teams (routing correctness)
#   20%  email was extracted (extraction completeness)

_VALID_URGENCIES = {"low", "normal", "high"}
_VALID_TEAMS = {"billing", "technical", "general"}


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

    # Step 3 — functional_group_path_kpi: Python scorer, no LLM call.
    #   • auto-registers into triage_group
    #   • after="ClassifyUrgency" wires the edge ClassifyUrgency → ScoreTriage
    #   • writes state["triage_quality"], state["triage_quality_reason"],
    #     and state["kpis"] = {"triage_quality": <score>}
    @functional_group_path_kpi(
        group=triage_group,
        after="ClassifyUrgency",
        input_schema=TriageResult,
        output_key="triage_quality",
        description="Score the quality of the triage classification.",
    )
    def score_triage(ctx: NodeContext) -> KpiResult:
        urgency: str = ctx.state.get("urgency", "")
        team: str = ctx.state.get("suggested_team", "")
        email: str | None = ctx.state.get("email")

        urgency_score = 1.0 if urgency in _VALID_URGENCIES else 0.0
        team_score = 1.0 if team.lower() in _VALID_TEAMS else 0.0
        email_score = 1.0 if email else 0.0

        score = round(
            0.5 * urgency_score + 0.3 * team_score + 0.2 * email_score,
            3,
        )
        return KpiResult(
            name="triage_quality",
            score=score,
            reason=(
                f"urgency={urgency!r}({urgency_score:.1f}) "
                f"team={team!r}({team_score:.1f}) "
                f"email={'present' if email else 'missing'}({email_score:.1f})"
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
            # After the group KPI scorer, continue to End unconditionally
            edge_deterministic("ScoreTriage", "End"),
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
        "I need this fixed urgently. My email is james@example.com.",
        # Off-topic input — expect urgency/team to be unpredictable → lower score
        "What is the capital of France?",
    ]

    for text in tickets:
        print(f"\n--- ticket: {text[:70]!r} ---")
        result = fsm.run(TicketInput(text=text), client=client)
        print(f"final_state                  : {result.final_state}")
        for step in result.steps:
            for item in step.results:
                print(f"  node={item.node_id}  status={item.status}  output={item.output}")
        print(f"state[triage_quality]        : {result.state.get('triage_quality')}")
        print(f"state[triage_quality_reason] : {result.state.get('triage_quality_reason')}")
        print(f"state[kpis]                  : {result.state.get('kpis')}")


if __name__ == "__main__":
    main()
