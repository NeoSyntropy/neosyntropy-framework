"""Validation cookbook — FSM level: validate the entire run path before End.

A ``functional_fsm_path_validator`` sits as the last node before ``End`` and
can inspect the full execution history through ``extract_fsm_path(ctx)``.
``FSMPathInfo`` gives you the ordered list of succeeded node ids, their
outputs, and the accumulated state — so you can enforce invariants over the
whole run, not just a single node.

Flow::

    ParseQuery → GenerateAnswer → AuditPath ──(valid)──▶ End
                                       │
                                  (not valid)
                                       ▼
                                  OutOfScope   ← fallback

The ``AuditPath`` node checks that both ``ParseQuery`` and ``GenerateAnswer``
ran successfully before allowing the run to complete.

Run::

    python cookbook/validation/fsm_path_validation_example.py
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
    NodeContext,
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.core.validation import (
    functional_fsm_path_validator,
    extract_fsm_path,
)
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
        "Validation cookbook — FSM level",
        f"validation-fsm-{stamp}",
        description="Functional path validator over the entire FSM run",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


class ParsedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str
    keywords: list[str]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# Required steps — both must be present in the path for the run to be valid
# ---------------------------------------------------------------------------

REQUIRED_STEPS = {"ParseQuery", "GenerateAnswer"}


# ---------------------------------------------------------------------------
# FSM builder
# ---------------------------------------------------------------------------

def build_fsm(provider: str) -> FSM:
    parse = SchemaNode(
        id="ParseQuery",
        input_schema=QueryInput,
        output_schema=ParsedQuery,
        provider=provider,
        prompt=(
            "Extract the user's intent (one short phrase) and a list of "
            "relevant keywords from the question."
        ),
    )

    answer = SchemaNode(
        id="GenerateAnswer",
        input_schema=ParsedQuery,
        output_schema=Answer,
        provider=provider,
        prompt=(
            "Answer the question based on the parsed intent and keywords. "
            "Provide a confidence score between 0.0 and 1.0."
        ),
    )

    # functional_fsm_path_validator: pure Python handler, no LLM call.
    # Receives a NodeContext; extract_fsm_path() turns prior_executions
    # into a structured FSMPathInfo for easy inspection.
    @functional_fsm_path_validator(
        id="AuditPath",
        input_schema=Answer,
        description=(
            "Verify that both ParseQuery and GenerateAnswer ran successfully "
            "before the run is allowed to complete."
        ),
    )
    def audit_path(ctx: NodeContext) -> ValidationResult:
        path = extract_fsm_path(ctx)

        missing = REQUIRED_STEPS - set(path.nodes_executed)
        if missing:
            return ValidationResult(
                valid=False,
                reason=f"required steps did not succeed: {sorted(missing)}",
            )

        # Optional: also check the answer confidence is acceptable
        answer_output = path.outputs.get("GenerateAnswer", {})
        confidence = answer_output.get("confidence", 1.0) if isinstance(answer_output, dict) else 1.0
        if confidence < 0.3:
            return ValidationResult(
                valid=False,
                reason=f"answer confidence too low: {confidence:.2f}",
            )

        path_str = " → ".join(path.nodes_executed)
        return ValidationResult(
            valid=True,
            reason=f"path OK: {path_str}",
        )

    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=QueryInput,
        output_schema=TextOutput,
        provider=provider,
        prompt="Politely explain that the query could not be processed.",
    )

    return FSM(
        entry=parse,
        nodes=[parse, answer, audit_path, out_of_scope],
        edges=[
            edge_deterministic("ParseQuery", "GenerateAnswer"),
            edge_deterministic("GenerateAnswer", "AuditPath"),
            # Branch on state["valid"] written by the FSM path validator
            edge_deterministic(
                "AuditPath",
                "End",
                guard=lambda s: bool(s.get("valid", False)),
            ),
            edge_deterministic(
                "AuditPath",
                "OutOfScope",
                guard=lambda s: not bool(s.get("valid", True)),
            ),
            edge_fallback("ParseQuery", "OutOfScope"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()
    fsm = build_fsm(_provider())

    queries = [
        "What are the main causes of climate change?",
        "How does photosynthesis work in C4 plants?",
    ]

    for question in queries:
        print(f"\n--- question: {question!r} ---")
        result = fsm.run(QueryInput(question=question), client=client)
        print(f"final_state     : {result.final_state}")
        for step in result.steps:
            for item in step.results:
                print(f"  node={item.node_id}  status={item.status}")
        print(f"state[valid]    : {result.state.get('valid')}")
        print(f"state[reason]   : {result.state.get('valid_reason')}")


if __name__ == "__main__":
    main()
