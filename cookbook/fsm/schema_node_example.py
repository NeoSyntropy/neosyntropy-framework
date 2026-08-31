"""FSM cookbook: simple schema extraction with `SchemaNode`.

Run::

    python cookbook/fsm/schema_node_example.py
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
    SchemaNode,
    TextOutput,
    edge_deterministic,
    edge_fallback,
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
        "FSM schema node cookbook",
        f"fsm-schema-node-{stamp}",
        description="Live cookbook run for schema extraction",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


class TicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class TicketSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_name: str
    email: str | None = None
    topic: str
    urgency: str = "normal"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()

    extract_ticket = SchemaNode(
        id="ExtractTicket",
        input_schema=TicketInput,
        output_schema=TicketSummary,
        provider=_provider(),
        prompt=(
            "Extract the customer's name, email address if present, the main topic, "
            "and the urgency level from the message. Use urgency values low, normal, or high."
        ),
    )

    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=TicketInput,
        output_schema=TextOutput,
        provider=_provider(),
        prompt="Politely refuse messages that are not support tickets.",
    )

    fsm = FSM(
        entry=extract_ticket,
        nodes=[extract_ticket, out_of_scope],
        edges=[
            edge_deterministic("ExtractTicket", "End"),
            edge_fallback("ExtractTicket", "OutOfScope"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )

    result = fsm.run(
        TicketInput(
            text="Hi, I'm María Garcia. Please update my billing email to maria@example.com."
        ),
        state={},
        client=client,
    )
    print(f"final_state: {result.final_state}")
    for step in result.steps:
        for item in step.results:
            print(f"Node: {item.node_id}, Output: {item.output}")


if __name__ == "__main__":
    main()
