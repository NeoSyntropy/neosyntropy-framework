"""Validation cookbook — node level: functional gate on a single node's output.

A ``functional_validation_node`` wraps a Python function that inspects the
accumulated workflow state and returns ``True`` / ``False`` (or a full
``ValidationResult``).  The framework writes ``state["valid"]`` and
``state["valid_reason"]`` so downstream deterministic edges can branch without
extra code.

Flow::

    SummarizeText → ValidateSummary ──(valid)──▶ End
                           │
                      (not valid)
                           ▼
                       OutOfScope   ← fallback

Run::

    python cookbook/validation/node_validation_example.py
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
from neosyntropy.core.validation import functional_validation_node
from neosyntropy.core.node.schemas import ValidationResult

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
        "Validation cookbook — node level",
        f"validation-node-{stamp}",
        description="Functional validation gate on a single node output",
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
# Nodes
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

    # functional_validation_node: pure Python, no LLM call needed.
    # Writes state["valid"] and state["valid_reason"] automatically.
    @functional_validation_node(
        id="ValidateSummary",
        input_schema=Summary,
        description="Ensure the summary has a headline and at least two bullets.",
    )
    def validate_summary(ctx: NodeContext) -> ValidationResult:
        headline = ctx.state.get("headline", "")
        bullets: list[str] = ctx.state.get("bullets", [])
        if not headline:
            return ValidationResult(valid=False, reason="headline is empty")
        if len(bullets) < 2:
            return ValidationResult(
                valid=False,
                reason=f"expected ≥2 bullets, got {len(bullets)}",
            )
        return ValidationResult(valid=True, reason="summary looks good")

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
        nodes=[summarize, validate_summary, out_of_scope],
        edges=[
            # Chain extraction → validation
            edge_deterministic("SummarizeText", "ValidateSummary"),
            # Branch on state["valid"] written by the functional validator
            edge_deterministic(
                "ValidateSummary",
                "End",
                guard=lambda s: bool(s.get("valid", False)),
            ),
            edge_deterministic(
                "ValidateSummary",
                "OutOfScope",
                guard=lambda s: not bool(s.get("valid", True)),
            ),
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
        # Edge-case: very short input that should fail validation
        "OK.",
    ]

    for text in inputs:
        print(f"\n--- input: {text[:60]!r} ---")
        result = fsm.run(ArticleInput(text=text), client=client)
        print(f"final_state : {result.final_state}")
        for step in result.steps:
            for item in step.results:
                print(f"  node={item.node_id}  status={item.status}  output={item.output}")
        print(f"state[valid]  : {result.state.get('valid')}")
        print(f"state[reason] : {result.state.get('valid_reason')}")


if __name__ == "__main__":
    main()
