"""FSM cookbook: semantic routing with a chained follow-up step.

Run::

    python cookbook/fsm/semantic_router_sequential_example.py
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
    OpenInput,
    SemanticRouter,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
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
        "FSM semantic router cookbook",
        f"fsm-semantic-router-sequential-{stamp}",
        description="Live cookbook run for semantic routing chains",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


class SupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


@node(id="CaptureRequest", input_schema=SupportRequest, output_schema=TextOutput)
def capture_request(ctx):
    return ctx.result(
        output={"message": "Captured support request."},
        state_updates={"request_text": ctx.input["text"]},
        next_state="SupportIntent",
    )


@node(id="InvestigateBilling", input_schema=OpenInput, output_schema=TextOutput)
def investigate_billing(ctx):
    return ctx.result(
        output={"message": "Investigating billing details."},
        state_updates={"lane": "billing", "investigated": True},
        next_state="ResolveRequest",
    )


@node(id="InvestigateShipping", input_schema=OpenInput, output_schema=TextOutput)
def investigate_shipping(ctx):
    return ctx.result(
        output={"message": "Investigating shipping details."},
        state_updates={"lane": "shipping", "investigated": True},
        next_state="ResolveRequest",
    )


@node(id="ResolveRequest", input_schema=OpenInput, output_schema=TextOutput)
def resolve_request(ctx):
    lane = ctx.state.get("lane", "general")
    return ctx.result(
        output={"message": f"Resolved the {lane} request."},
        state_updates={"resolved": True},
        next_state="End",
    )


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can only route billing or shipping requests."})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()

    router = SemanticRouter(
        id="SupportIntent",
        input_schema=SupportRequest,
        routes={
            "billing": investigate_billing,
            "shipping": investigate_shipping,
        },
        fallback_node=out_of_scope,
        provider=_provider(),
    )

    fsm = FSM(
        entry=capture_request,
        nodes=[
            capture_request,
            investigate_billing,
            investigate_shipping,
            resolve_request,
            out_of_scope,
        ],
        routers=[router],
        edges=[
            edge_deterministic("CaptureRequest", "SupportIntent"),
            edge_deterministic("InvestigateBilling", "ResolveRequest"),
            edge_deterministic("InvestigateShipping", "ResolveRequest"),
            edge_deterministic("ResolveRequest", "End"),
            edge_fallback("SupportIntent", "OutOfScope"),
        ],
    )

    result = fsm.run(
        SupportRequest(text="I need to change my shipping address before the order goes out."),
        state={},
        client=client,
    )
    print(f"final_state: {result.final_state}")
    print(result.steps[-1].results[0].output)


if __name__ == "__main__":
    main()
