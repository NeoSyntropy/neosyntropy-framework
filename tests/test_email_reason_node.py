"""Local runnable demo: ReasoningNode looks up a profile and sends email via EmailTools.

Loads credentials from ``tests/.env`` next to this file (see ``tests/.env.example``).

Run::

    python tests/test_email_reason_node.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    ReasoningNode,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
    node,
    tool,
)
from neosyntropy.tools.email.email import EmailTools

RECIPIENT_EMAIL = "avraham@upsailor.ai"
VERTEX_MODEL = "gemini-2.5-flash"
TESTS_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_tests_env() -> None:
    """Load KEY=VALUE pairs from tests/.env into os.environ (file wins)."""
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
        raise SystemExit(
            f"Missing required value {name} in {TESTS_ENV_PATH}."
        )
    return value


def _client_from_env() -> Client:
    return Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        project_id=_require_env("NEOSYNTROPY_PROJECT_ID"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com").strip()
        or "https://api.neosyntropy.com",
        telemetry_timeout=20.0,
    )


def _email_tools_from_env() -> EmailTools:
    return EmailTools(
        receiver_email=RECIPIENT_EMAIL,
        sender_name="NeoSyntropy",
        sender_email=os.environ.get("SENDER_EMAIL", "test@example.com"),
        sender_passkey=os.environ.get("SENDER_PASSKEY", "testpasskey"),
    )


class PersonalizedMailInput(BaseModel):
    """Typed run input for the personalized email workflow."""

    model_config = ConfigDict(extra="forbid")
    intent: str


class PersonProfileArgs(BaseModel):
    email: str


class MailSendSummary(BaseModel):
    """Structured JSON summary after the mail attempt."""

    model_config = ConfigDict(extra="forbid")
    sent: bool
    to_email: str
    subject: str
    notes: str | None = None


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(registry=registry)
    def get_person_profile(args: PersonProfileArgs) -> dict:
        """Return a simple personalized profile for the given email address."""
        profiles = {
            RECIPIENT_EMAIL.lower(): {
                "email": RECIPIENT_EMAIL,
                "display_name": "Avraham",
                "interest": "deterministic AI workflows",
                "tone": "friendly and concise",
                "product_hook": "NeoSyntropy Email tools",
            },
        }
        profile = profiles.get(args.email.lower())
        if profile is None:
            return {
                "email": args.email,
                "display_name": "there",
                "interest": "building with NeoSyntropy",
                "tone": "friendly",
                "product_hook": "NeoSyntropy",
            }
        return profile

    _email_tools_from_env().register(registry)
    return registry


def build_fsm() -> FSM:
    reason_node = ReasoningNode(
        id="ReasoningNode",
        input_schema=PersonalizedMailInput,
        provider=VERTEX_MODEL,
        prompt=(
            "You help send a short personalized email.\n"
            "Emit tool calls ONLY in this exact wire format (no other syntax):\n"
            "  <TOOL:get_person_profile>\n"
            "  <TOOL:email_user>\n"
            "Steps:\n"
            f"1) Emit <TOOL:get_person_profile> for email {RECIPIENT_EMAIL}.\n"
            "2) After the tool result, write a one-paragraph greeting using the profile.\n"
            f"3) Emit <TOOL:email_user> so the mail goes to {RECIPIENT_EMAIL} "
            "with a clear subject and body.\n"
            "Do not invent other recipients. Do not use function-call or <TOOL_CODE> syntax."
        ),
        tools=("get_person_profile", "email_user"),
    )

    @node(
        id="Summarize",
        input_schema=OpenInput,
        output_schema=MailSendSummary,
    )
    def summarize(ctx):
        """Record whether email_user succeeded."""
        send_calls = [
            item
            for item in ctx.tools.registry.invocations
            if item.tool == "email_user"
        ]
        last = send_calls[-1] if send_calls else None
        subject = ""
        if last and isinstance(last.arguments, dict):
            subject = str(last.arguments.get("subject") or "")
        sent = bool(last and last.ok)
        return ctx.result(
            output=MailSendSummary(
                sent=sent,
                to_email=RECIPIENT_EMAIL,
                subject=subject or "(none)",
                notes=(
                    "email_user succeeded"
                    if sent
                    else "email_user was not invoked successfully"
                ),
            )
        )

    @node(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
    )
    def out_of_scope(ctx):
        return ctx.result(output={"message": "Out of scope for personalized email demo."})

    return FSM(
        entry=reason_node,
        nodes=[reason_node, summarize, out_of_scope],
        edges=[
            edge_deterministic("ReasoningNode", "Summarize"),
            edge_deterministic("Summarize", "End"),
            edge_fallback("ReasoningNode", "OutOfScope"),
        ],
    )


def main() -> None:
    _load_tests_env()
    client = _client_from_env()
    registry = build_registry()
    fsm = build_fsm()
    result = fsm.run(
        PersonalizedMailInput(
            intent=(
                f"Send a short personalized welcome email to {RECIPIENT_EMAIL} "
                "using their profile."
            ),
        ),
        state={"recipient_email": RECIPIENT_EMAIL},
        client=client,
        tools=registry,
    )
    print("Execution Finished!")
    print(f"Final State: {result.final_state} (Rejected: {result.rejected})")
    send_ok = any(
        inv.tool == "email_user" and inv.ok for inv in registry.invocations
    )
    print(f"\nEmailTools email_user ok: {send_ok}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
