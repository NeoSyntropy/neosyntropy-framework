"""KPI cookbook — node level: score a single node's output mid-path.

A ``functional_kpi_node`` wraps a Python function that inspects the
accumulated workflow state and returns a ``float`` or a full ``KpiResult``.
The framework writes ``state[output_key]``, ``state[output_key + "_reason"]``,
and accumulates the score into the shared ``state["kpis"]`` dict.

Unlike a validation node the run **always continues** — no branching is
needed unless you also want a threshold gate (see README for that pattern).

Flow::

    SummarizeText → ScoreSummary ──▶ End
                         │
                         └─ writes state["summary_quality"] = 0.75
                            state["kpis"] = {"summary_quality": 0.75}

Run::

    python cookbook/kpi/node_kpi_example.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neosyntropy import (
    Client,
    FSM,
    NodeContext,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.core.kpi import functional_kpi_node
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
        "KPI cookbook — node level",
        f"kpi-node-{stamp}",
        description="Functional KPI node scoring a single node output",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ArticleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: str
    bullets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring rubric
# ---------------------------------------------------------------------------
#
# summary_quality score is a weighted average of three signals:
#   40%  headline present and non-trivial (> 5 words)
#   40%  bullets count (capped at 3, normalised to 0–1)
#   20%  average bullet length (quality proxy; capped at 15 words)

_IDEAL_BULLETS = 3
_IDEAL_BULLET_WORDS = 15


def _score_summary(headline: str, bullets: list[str]) -> KpiResult:
    headline_score = 1.0 if len(headline.split()) > 5 else 0.5 if headline else 0.0
    bullet_score = min(len(bullets), _IDEAL_BULLETS) / _IDEAL_BULLETS
    avg_words = (
        sum(len(b.split()) for b in bullets) / len(bullets)
        if bullets else 0
    )
    length_score = min(avg_words, _IDEAL_BULLET_WORDS) / _IDEAL_BULLET_WORDS
    score = round(0.4 * headline_score + 0.4 * bullet_score + 0.2 * length_score, 3)
    return KpiResult(
        name="summary_quality",
        score=score,
        reason=(
            f"headline_score={headline_score:.2f} "
            f"bullet_score={bullet_score:.2f} "
            f"length_score={length_score:.2f}"
        ),
    )


# ---------------------------------------------------------------------------
# FSM builder
# ---------------------------------------------------------------------------

def build_fsm(provider: str) -> FSM:
    summarize = SchemaNode(
        id="SummarizeText",
        input_schema=ArticleInput,
        output_schema=Summary,
        provider=provider,
        prompt=(
            "Summarise the article in one short headline and at least two "
            "bullet points. Each bullet must be a complete sentence."
        ),
    )

    # functional_kpi_node: pure Python scorer, no LLM call.
    # Writes state["summary_quality"], state["summary_quality_reason"],
    # and state["kpis"] = {"summary_quality": <score>}.
    @functional_kpi_node(
        id="ScoreSummary",
        input_schema=Summary,
        output_key="summary_quality",
        description="Score the summary quality on a 0–1 scale.",
    )
    def score_summary(ctx: NodeContext) -> KpiResult:
        headline: str = ctx.state.get("headline", "")
        bullets: list[str] = ctx.state.get("bullets", [])
        return _score_summary(headline, bullets)

    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=ArticleInput,
        output_schema=TextOutput,
        provider=provider,
        prompt="Politely explain that you could not produce a useful summary.",
    )

    return FSM(
        entry=summarize,
        nodes=[summarize, score_summary, out_of_scope],
        edges=[
            # Chain extraction → KPI scorer
            edge_deterministic("SummarizeText", "ScoreSummary"),
            # KPI nodes always continue unconditionally
            edge_deterministic("ScoreSummary", "End"),
            edge_fallback("SummarizeText", "OutOfScope"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    fsm = build_fsm(_provider())

    inputs = [
        "Researchers at MIT have developed a new battery technology that charges "
        "in under five minutes and lasts three times longer than current lithium-ion "
        "cells. The breakthrough uses a solid-state electrolyte and could reach "
        "commercial production within two years.",
        # Short input — expect a lower quality score
        "OK.",
    ]

    for text in inputs:
        print(f"\n--- input: {text[:60]!r} ---")
        result = fsm.run(ArticleInput(text=text), client=client)
        print(f"final_state                  : {result.final_state}")
        for step in result.steps:
            for item in step.results:
                print(f"  node={item.node_id}  status={item.status}  output={item.output}")
        print(f"state[summary_quality]       : {result.state.get('summary_quality')}")
        print(f"state[summary_quality_reason]: {result.state.get('summary_quality_reason')}")
        print(f"state[kpis]                  : {result.state.get('kpis')}")


if __name__ == "__main__":
    main()
