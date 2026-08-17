"""Validate tools_executed telemetry against real ReasoningNode tool runs.

Puts 5 tools on a ReasoningNode, captures telemetry, then critic-compares
``tools_executed`` events to in-process ``NodeResult.tool_calls``.

Run (backend + Cloud SQL proxy up, tests/.env filled)::

    python tests/validate_tools_executed_telemetry.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
    tool,
)
from neosyntropy.core.models import GenerateResult, ToolCall
from neosyntropy.observability import BackendTelemetryReporter
from neosyntropy.tools.calling import ToolCallingLoop

VERTEX_MODEL = "gemini-2.5-flash"
TESTS_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_tests_env() -> None:
    if not TESTS_ENV_PATH.is_file():
        raise SystemExit(
            f"Missing {TESTS_ENV_PATH}. Copy tests/.env.example to tests/.env and fill values."
        )
    for raw in TESTS_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required value {name} in {TESTS_ENV_PATH}.")
    return value


class TeeObserver:
    """Forwards telemetry to the backend and keeps a local copy for critique."""

    def __init__(self, inner: BackendTelemetryReporter) -> None:
        self.inner = inner
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self.run_id: str | None = None

    async def run_started(self, **kwargs: Any) -> str | None:
        run_id = await self.inner.run_started(**kwargs)
        self.run_id = str(run_id) if run_id else None
        return run_id

    async def event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        self.events.append((run_id, event_type, dict(payload)))
        await self.inner.event(run_id, event_type, payload)

    async def run_finished(self, run_id: str, **kwargs: Any) -> None:
        await self.inner.run_finished(run_id, **kwargs)


registry = ToolRegistry()


class InvestigationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str


class CustomerArgs(BaseModel):
    customer_id: str


@tool(registry=registry)
def lookup_customer_account(args: CustomerArgs) -> dict:
    """Look up customer account details including loyalty tier and status."""
    return {
        "customer_id": args.customer_id,
        "vip_tier": "Gold",
        "account_status": "active",
        "extended_return_window_days": 60,
    }


class OrderArgs(BaseModel):
    order_id: str


@tool(registry=registry)
def fetch_transaction_history(args: OrderArgs) -> dict:
    """Fetch order transaction details: purchase date, product, amount."""
    return {
        "order_id": args.order_id,
        "product_id": "prod_laptop_pro",
        "purchase_days_ago": 35,
        "amount": 1299.99,
        "item_condition_reported": "defective",
    }


class ProductArgs(BaseModel):
    product_id: str


@tool(registry=registry)
def check_product_return_policy(args: ProductArgs) -> dict:
    """Check return policy rules and warranty for a product."""
    return {
        "product_id": args.product_id,
        "category": "electronics",
        "standard_return_days": 30,
        "allows_defective_return": True,
        "restocking_fee_percent": 0.0,
    }


class WarrantyArgs(BaseModel):
    product_id: str
    purchase_days_ago: int


@tool(registry=registry)
def check_warranty_coverage(args: WarrantyArgs) -> dict:
    """Check manufacturer warranty coverage for a product and purchase age."""
    return {
        "product_id": args.product_id,
        "purchase_days_ago": args.purchase_days_ago,
        "warranty_months": 24,
        "covered": args.purchase_days_ago <= 730,
        "coverage_type": "manufacturer_defect",
    }


class ShippingArgs(BaseModel):
    order_id: str


@tool(registry=registry)
def lookup_shipping_status(args: ShippingArgs) -> dict:
    """Look up delivery / shipping status for an order."""
    return {
        "order_id": args.order_id,
        "delivered": True,
        "delivery_days_ago": 30,
        "carrier": "UPS",
    }


class RefundClaimSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str
    order_id: str
    claim_valid: bool
    decision: str
    rationale: str
    tools_used: list[str]


FIVE_TOOLS = (
    "lookup_customer_account",
    "fetch_transaction_history",
    "check_product_return_policy",
    "check_warranty_coverage",
    "lookup_shipping_status",
)

reason_node = ReasoningNode(
    id="ReasoningNode",
    input_schema=InvestigationInput,
    provider=VERTEX_MODEL,
    prompt=(
        "Investigate the refund request. You MUST call ALL FIVE of these tools "
        "before finishing: lookup_customer_account, fetch_transaction_history, "
        "check_product_return_policy, check_warranty_coverage, and "
        "lookup_shipping_status. Do not skip any tool. Use customer_id "
        "cust_12345, order_id ord_98765, and product_id prod_laptop_pro. "
        "After all five tools return, write brief notes on claim validity."
    ),
    tools=FIVE_TOOLS,
)

summarize = SchemaNode(
    id="Summarize",
    input_schema=OpenInput,
    output_schema=RefundClaimSummary,
    provider=VERTEX_MODEL,
    prompt=(
        "Summarize the investigation as JSON. List every tool that was actually "
        "used in tools_used. Set claim_valid and decision "
        "(approve / deny / escalate) with a short rationale."
    ),
)

out_of_scope = SchemaNode(
    id="OutOfScope",
    is_fallback=True,
    input_schema=OpenInput,
    output_schema=TextOutput,
    provider=VERTEX_MODEL,
    prompt="Politely refuse out-of-scope requests.",
)

fsm = FSM(
    entry=reason_node,
    nodes=[reason_node, summarize, out_of_scope],
    edges=[
        edge_deterministic("ReasoningNode", "Summarize"),
        edge_deterministic("Summarize", "End"),
        edge_fallback("ReasoningNode", "OutOfScope"),
    ],
)


def _real_executed_tools(result: Any) -> list[dict[str, Any]]:
    """Non-denied tool calls from ReasoningNode steps (ground truth)."""
    out: list[dict[str, Any]] = []
    for step in result.steps:
        for item in step.results:
            if item.node_id != "ReasoningNode":
                continue
            for record in item.tool_calls:
                if record.denied:
                    continue
                out.append(
                    {
                        "tool": record.tool,
                        "arguments": dict(record.arguments),
                        "ok": record.ok,
                        "error": record.error,
                    }
                )
    return out


def _telemetry_executed_tools(observer: TeeObserver) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _run_id, event_type, payload in observer.events:
        if event_type != "tools_executed":
            continue
        if payload.get("node_id") != "ReasoningNode":
            continue
        for item in payload.get("tools") or []:
            out.append(
                {
                    "tool": item.get("tool"),
                    "arguments": dict(item.get("arguments") or {}),
                    "ok": item.get("ok"),
                    "error": item.get("error"),
                }
            )
    return out


def _critic(real: list[dict[str, Any]], telem: list[dict[str, Any]]) -> dict[str, Any]:
    real_names = [row["tool"] for row in real]
    telem_names = [row["tool"] for row in telem]
    real_set = set(real_names)
    telem_set = set(telem_names)
    missing_in_telemetry = sorted(real_set - telem_set)
    extra_in_telemetry = sorted(telem_set - real_set)
    # Order-sensitive multiset compare (same tool can appear once each).
    order_match = real_names == telem_names
    arg_mismatches: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(real, telem)):
        if left["tool"] != right["tool"]:
            continue
        if left["arguments"] != right["arguments"] or left["ok"] != right["ok"]:
            arg_mismatches.append({"index": index, "real": left, "telemetry": right})
    passed = (
        not missing_in_telemetry
        and not extra_in_telemetry
        and order_match
        and not arg_mismatches
        and len(real) > 0
    )
    return {
        "passed": passed,
        "real_count": len(real),
        "telemetry_count": len(telem),
        "real_tools": real_names,
        "telemetry_tools": telem_names,
        "missing_in_telemetry": missing_in_telemetry,
        "extra_in_telemetry": extra_in_telemetry,
        "order_match": order_match,
        "arg_mismatches": arg_mismatches,
        "five_tools_declared": list(FIVE_TOOLS),
        "all_five_executed": set(FIVE_TOOLS).issubset(real_set),
    }


class ScriptedFiveToolProvider:
    """Deterministic provider: emit all five tool calls, then summary JSON."""

    def __init__(self) -> None:
        self._turn = 0
        self._tool_plan: list[list[ToolCall]] = [
            [ToolCall(tool="lookup_customer_account", arguments={"customer_id": "cust_12345"})],
            [ToolCall(tool="fetch_transaction_history", arguments={"order_id": "ord_98765"})],
            [ToolCall(tool="check_product_return_policy", arguments={"product_id": "prod_laptop_pro"})],
            [
                ToolCall(
                    tool="check_warranty_coverage",
                    arguments={"product_id": "prod_laptop_pro", "purchase_days_ago": 35},
                )
            ],
            [ToolCall(tool="lookup_shipping_status", arguments={"order_id": "ord_98765"})],
            [],
        ]
        self._texts = [
            "Calling customer lookup.",
            "Calling transaction history.",
            "Calling return policy.",
            "Calling warranty coverage.",
            "Calling shipping status.",
            (
                "All five tools returned. Gold VIP, defective laptop at day 35, "
                "policy allows defective return, warranty covers, delivered."
            ),
        ]

    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        tools: Any = None,
        **_: Any,
    ) -> GenerateResult | str:
        if schema and schema.get("type") == "object":
            return json.dumps(
                {
                    "customer_id": "cust_12345",
                    "order_id": "ord_98765",
                    "claim_valid": True,
                    "decision": "approve",
                    "rationale": "VIP within extended window; defective return allowed.",
                    "tools_used": list(FIVE_TOOLS),
                }
            )
        index = min(self._turn, len(self._tool_plan) - 1)
        self._turn += 1
        return GenerateResult(
            text=self._texts[index],
            tool_calls=list(self._tool_plan[index]),
        )


if __name__ == "__main__":
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        project_id=_require_env("NEOSYNTROPY_PROJECT_ID"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com").strip()
        or "https://api.neosyntropy.com",
        telemetry_timeout=20.0,
    )
    backend = client._as_backend()
    tee = TeeObserver(BackendTelemetryReporter(backend))
    # Default tool-loop cap is 4; allow all five tools in one reasoning step.
    if ToolCallingLoop.__init__.__kwdefaults__ is not None:
        ToolCallingLoop.__init__.__kwdefaults__["max_tool_calls"] = 8

    # Override Vertex with a scripted provider so all five tools actually run.
    # Control + telemetry still go through the live backend.
    scripted = ScriptedFiveToolProvider()

    result = fsm.run(
        InvestigationInput(
            intent=(
                "Investigate refund for order ord_98765 / customer cust_12345 "
                "(laptop defect). Call all five investigation tools."
            ),
        ),
        state={
            "customer_id": "cust_12345",
            "order_id": "ord_98765",
            "issue": "laptop defect",
            "product_id": "prod_laptop_pro",
        },
        client=client,
        tools=registry,
        observer=tee,
        providers={VERTEX_MODEL: scripted},
        telemetry_timeout=20.0,
    )

    print("=== Run ===")
    print(f"final_state={result.final_state} rejected={result.rejected} run_id={tee.run_id}")
    if result.rejection:
        print(f"rejection={result.rejection}")

    print("\n=== ReasoningNode outputs + real tool_calls ===")
    for step in result.steps:
        for item in step.results:
            if item.node_id != "ReasoningNode":
                continue
            print(f"\n--- step {step.step} output ---\n{item.output}")
            for record in item.tool_calls:
                verdict = "ok" if record.ok else ("denied" if record.denied else "failed")
                print(
                    f"  [{verdict}] {record.tool} args={record.arguments} "
                    f"err={record.error or ''}"
                )

    real = _real_executed_tools(result)
    telem = _telemetry_executed_tools(tee)
    critique = _critic(real, telem)

    print("\n=== tools_executed telemetry events ===")
    for _run_id, event_type, payload in tee.events:
        if event_type == "tools_executed":
            print(json.dumps(payload, indent=2, default=str))

    print("\n=== CRITIC: telemetry vs real ReasoningNode executions ===")
    print(json.dumps(critique, indent=2, default=str))

    if not critique["passed"] or not critique["all_five_executed"]:
        raise SystemExit(1)
    print(
        "\nCRITIC PASS: all 5 tools executed; tools_executed telemetry "
        "matches ReasoningNode real tool_calls."
    )
