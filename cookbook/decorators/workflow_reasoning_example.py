"""@workflow: reason with tools, then extract order parameters.

Two ReasoningSteps gather catalog and stock evidence. The trailing SchemaNode
predicts typed parameters and calls place_order().

Creates a dedicated project, then runs against the local API (port 8000).

Run::

    python cookbook/decorators/workflow_reasoning_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    ReasoningStep,
    SchemaStep,
    ToolRegistry,
    tool,
    workflow,
)

TESTS_ENV_PATH = Path(__file__).resolve().parents[2] / "tests" / ".env"
DEFAULT_API_URL = "http://127.0.0.1:8000"

CATALOG = {
    "laptop": {"sku": "sku_laptop_pro", "name": "Laptop Pro"},
    "monitor": {"sku": "sku_monitor_27", "name": "27-inch Monitor"},
    "keyboard": {"sku": "sku_keyboard_mech", "name": "Mechanical Keyboard"},
}

STOCK = {
    "sku_laptop_pro": {"warehouse": "wh_east", "available": 12},
    "sku_monitor_27": {"warehouse": "wh_west", "available": 4},
    "sku_keyboard_mech": {"warehouse": "wh_east", "available": 30},
}


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


def _client_for_example() -> Client:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", DEFAULT_API_URL).strip()
        or DEFAULT_API_URL,
    )
    stamp = int(time.time())
    project = client.create_project(
        "Cookbook workflow reasoning",
        f"cookbook-workflow-reasoning-{stamp}",
        description="Live cookbook run for @workflow",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


def _provider() -> str:
    return os.environ.get("NEOSYNTROPY_PROVIDER", "neosyntropy/base").strip() or "neosyntropy/base"


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class LookupSkuArgs(BaseModel):
    query: str


class CheckStockArgs(BaseModel):
    sku: str


class OrderParams(BaseModel):
    """Parameters predicted after the reasoning tools return evidence."""

    model_config = ConfigDict(extra="forbid")
    sku: str
    quantity: int
    warehouse: str


def main() -> None:
    client = _client_for_example()
    registry = ToolRegistry()

    @tool(registry=registry)
    def lookup_sku(args: LookupSkuArgs) -> dict:
        """Resolve a product name or description to a catalog SKU."""
        needle = args.query.lower()
        for key, item in CATALOG.items():
            if key in needle or item["name"].lower() in needle:
                print(f"[lookup_sku] {args.query!r} -> {item['sku']}")
                return item
        print(f"[lookup_sku] no match for {args.query!r}")
        return {"sku": None, "name": None}

    @tool(registry=registry)
    def check_stock(args: CheckStockArgs) -> dict:
        """Return warehouse location and available units for a SKU."""
        row = STOCK.get(args.sku, {"warehouse": None, "available": 0})
        print(f"[check_stock] {args.sku} -> {row}")
        return {"sku": args.sku, **row}

    @workflow(
        input_schema=UserRequest,
        steps=[
            ReasoningStep(
                instruction="Identify the product in the request and look up its SKU.",
                tools=["lookup_sku"],
            ),
            ReasoningStep(
                instruction="Check warehouse stock for that SKU before placing the order.",
                tools=["check_stock"],
            ),
            SchemaStep(
                instruction=(
                    "Using the catalog and stock tool results, extract sku, quantity, "
                    "and warehouse for the customer's order. Quantity comes from the request."
                )
            ),
        ],
        client=client,
        tools=registry,
        provider=_provider(),
    )
    def place_order(params: OrderParams) -> str:
        available = STOCK.get(params.sku, {}).get("available", 0)
        if params.quantity > available:
            return (
                f"Cannot order {params.quantity} x {params.sku}: "
                f"only {available} in {params.warehouse}."
            )
        return (
            f"Ordered {params.quantity} x {params.sku} from {params.warehouse}."
        )

    result = place_order(text="We need 3 laptops for the sales team")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(result)


if __name__ == "__main__":
    main()
