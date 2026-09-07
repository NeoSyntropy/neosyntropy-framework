"""FSM cookbook: pure Python handler nodes with the ``@node`` decorator.

This example builds a two-node order-validation pipeline entirely in Python —
no LLM calls, no tools. ``ValidateOrder`` validates the payload and enriches
workflow state; ``FormatResponse`` reads that state and returns an appropriate
success or rejection message.

Key pattern — backend sequential execution
------------------------------------------
In a backend-controlled FSM, nodes are executed through *deterministic edges*.
``_enqueue_deterministic_next_step`` auto-chains the next node when exactly
one deterministic edge leaves the current node. With two edges the backend
can't resolve which to take (guards are client-side only), so the pipeline
must be *linear*: both valid and invalid orders pass through the same nodes
with the outcome stored in workflow state.

Run::

    python cookbook/fsm/python_node_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    Client,
    NodeContext,
    OpenInput,
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
        raise SystemExit(f"Missing {name}. Copy tests/.env.example to tests/.env and fill values.")
    return value


def _client_for_example() -> Client:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL,
    )
    # Fixed slug so ensure_project finds and reuses the project on every run
    # (backend enforces a unique case-insensitive name per user, so a
    # timestamp slug would cause a 409 on second invocation).
    project = client.create_project(
        "FSM python node cookbook",
        "fsm-python-node-cookbook",
        description="Live cookbook run for pure Python handler nodes",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    customer: str
    amount: float
    currency: str = "USD"


class ValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str


class ResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


# ---------------------------------------------------------------------------
# Python handler nodes
# ---------------------------------------------------------------------------


@node(
    id="ValidateOrder",
    input_schema=OrderInput,
    output_schema=ValidationOutput,
)
def validate_order(ctx: NodeContext) -> object:
    """Validate the order payload and enrich workflow state.

    The outcome (valid or not) is stored in ``state_updates`` so the next
    node in the pipeline can read it from ``ctx.state``.

    No explicit ``next_state`` is set here — the backend auto-chains to
    ``FormatResponse`` via the single deterministic edge declared in the FSM.
    """
    data = ctx.input
    order_id: str = data.get("order_id", "").strip()
    amount: float = float(data.get("amount", 0))
    customer: str = data.get("customer", "").strip()

    if not order_id:
        print("[ValidateOrder] rejected — missing order_id")
        return ctx.result(
            output={"valid": False, "reason": "order_id is required"},
            state_updates={"valid": False, "reason": "order_id is required"},
        )

    if amount <= 0:
        print(f"[ValidateOrder] rejected — amount {amount} is not positive")
        return ctx.result(
            output={"valid": False, "reason": f"amount must be positive, got {amount}"},
            state_updates={
                "valid": False,
                "reason": f"amount must be positive, got {amount}",
            },
        )

    print(f"[ValidateOrder] order {order_id!r} for {customer!r} — amount {amount} OK")
    return ctx.result(
        output={"valid": True, "reason": "all checks passed"},
        state_updates={
            "valid": True,
            "order_id": order_id,
            "customer": customer,
            "amount": amount,
        },
    )


@node(
    id="ErrorFallback",
    input_schema=OpenInput,
    output_schema=TextOutput,
    is_fallback=True,
)
def error_fallback(ctx: NodeContext) -> object:
    """Catch-all for unexpected FSM failures (required by every FSM graph)."""
    reason = ctx.state.get("reason", "an unexpected error occurred")
    msg = f"FSM error: {reason}"
    print(f"[ErrorFallback] {msg}")
    return ctx.result(output={"message": msg})


@node(
    id="FormatResponse",
    input_schema=OpenInput,
    output_schema=ResponseOutput,
)
def format_response(ctx: NodeContext) -> object:
    """Build the final user-facing message from the enriched workflow state.

    Reads the ``valid`` flag set by ``ValidateOrder`` and produces either a
    confirmation or a rejection string.  Both outcomes pass through this same
    node — conditional branching at the backend level requires LLM semantic
    routing; for pure Python handlers the outcome lives in state.
    """
    state = ctx.state
    if state.get("valid"):
        customer = state.get("customer", "Customer")
        order_id = state.get("order_id", "unknown")
        amount = state.get("amount", 0)
        currency = ctx.input.get("currency", "USD")
        msg = (
            f"Hi {customer}! Your order #{order_id} for {currency} {amount:.2f} has been confirmed."
        )
    else:
        reason = state.get("reason", "validation failed")
        msg = f"Order could not be processed: {reason}"
    print(f"[FormatResponse] {msg}")
    return ctx.result(output={"message": msg})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_fsm() -> FSM:
    """Build the order-validation graph without performing I/O."""
    return FSM(
        entry=validate_order,
        nodes=[validate_order, format_response, error_fallback],
        edges=[
            edge_deterministic("ValidateOrder", "FormatResponse"),
            edge_deterministic("FormatResponse", "End"),
            edge_fallback("ValidateOrder", "ErrorFallback"),
            edge_fallback("FormatResponse", "ErrorFallback"),
        ],
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = _client_for_example()

    # Single deterministic chain: ValidateOrder → FormatResponse → End.
    # After ValidateOrder finishes, the backend auto-enqueues FormatResponse
    # (unique deterministic edge, non-End target).  After FormatResponse
    # finishes, the backend auto-hops to End (unique deterministic edge,
    # End target) and marks the run complete.
    fsm = build_fsm()

    for label, payload in [
        ("valid order", OrderInput(order_id="ORD-001", customer="Alice", amount=149.99)),
        ("zero amount", OrderInput(order_id="ORD-002", customer="Bob", amount=0.0)),
        ("missing id", OrderInput(order_id="", customer="Carol", amount=50.0)),
    ]:
        print(f"\n--- {label} ---")
        result = fsm.run(payload, state={}, client=client)
        print(f"final_state: {result.final_state}")
        for step in result.steps:
            for item in step.results:
                print(f"  Node: {item.node_id}, status={item.status}, output={item.output}")


if __name__ == "__main__":
    main()
