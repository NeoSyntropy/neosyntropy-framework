"""KPI cookbook — FSM level: score the entire run path before End.

A ``functional_fsm_path_kpi`` sits as the last node before ``End`` and scores
the full execution history through ``extract_fsm_path(ctx)``.
``FSMPathInfo`` gives you the ordered list of succeeded node ids, their
outputs, and the accumulated state — so you can compute composite quality
metrics over the whole run, not just a single node.

The KPI node always proceeds to ``End`` — no fallback edge is needed.

Flow::

    ParseQuery → GenerateAnswer → PathScore ──▶ End
                                       │
                                       └─ writes state["path_quality"] = 0.82
                                          state["kpis"] = {"path_quality": 0.82}

Run::

    python cookbook/kpi/fsm_path_kpi_example.py
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
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.core.kpi import (
    extract_fsm_path,
    functional_fsm_path_kpi,
)
from neosyntropy.core.node.schemas import KpiResult
from neosyntropy import TextOutput

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
        "KPI cookbook — FSM level",
        f"kpi-fsm-{stamp}",
        description="Functional path KPI scorer over the entire FSM run",
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
# Required steps — both must have run for the path to be fully covered
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

    # functional_fsm_path_kpi: pure Python scorer, no LLM call.
    # Receives a NodeContext; extract_fsm_path() turns prior_executions
    # into a structured FSMPathInfo for easy inspection.
    @functional_fsm_path_kpi(
        id="PathScore",
        input_schema=Answer,
        output_key="path_quality",
        description=(
            "Compute a composite quality score (coverage × confidence) "
            "for the full FSM run."
        ),
    )
    def path_score(ctx: NodeContext) -> KpiResult:
        path = extract_fsm_path(ctx)

        # Component 1: step coverage (0–1)
        hit = REQUIRED_STEPS & set(path.nodes_executed)
        coverage = len(hit) / len(REQUIRED_STEPS)

        # Component 2: answer confidence (0–1)
        answer_output = path.outputs.get("GenerateAnswer", {})
        confidence = (
            float(answer_output.get("confidence", 0.0))
            if isinstance(answer_output, dict)
            else 0.0
        )

        score = round(0.6 * coverage + 0.4 * confidence, 3)

        return KpiResult(
            name="path_quality",
            score=score,
            reason=(
                f"coverage={coverage:.2f} "
                f"confidence={confidence:.2f} "
                f"path={path.nodes_executed}"
            ),
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
        nodes=[parse, answer, path_score, out_of_scope],
        edges=[
            edge_deterministic("ParseQuery", "GenerateAnswer"),
            edge_deterministic("GenerateAnswer", "PathScore"),
            # KPI nodes always continue — no guard, no fallback from this node
            edge_deterministic("PathScore", "End"),
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
        print(f"state[path_quality]        : {result.state.get('path_quality')}")
        print(f"state[path_quality_reason] : {result.state.get('path_quality_reason')}")
        print(f"state[kpis]                : {result.state.get('kpis')}")


if __name__ == "__main__":
    main()
