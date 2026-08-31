"""Two @function_calling decorators on the same project.

Creates one shared project, registers two functions under it, and runs both.
Both FSMs appear in the project_graphs table and show up as separate cards
in the console's StateGraph list.

Run::

    python cookbook/decorators/function_calling_example.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import Client, function_calling

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


def _client_for_example() -> Client:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", DEFAULT_API_URL).strip()
        or DEFAULT_API_URL,
    )
    stamp = int(time.time())
    # One shared project — both functions register their graphs here.
    project = client.create_project(
        "Cookbook two functions",
        f"cookbook-two-functions-{stamp}",
        description="Two @function_calling decorators on the same project",
    )
    print(f"project: {project.get('name')} ({project.get('id')})")
    return client


def _provider() -> str:
    return os.environ.get("NEOSYNTROPY_PROVIDER", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


# ── Shared input schema ────────────────────────────────────────────────────

class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


# ── Function 1: greet ──────────────────────────────────────────────────────

class GreetParams(BaseModel):
    """Parameters the model must extract before greet() runs."""

    model_config = ConfigDict(extra="forbid")
    name: str
    language: str = "en"


# ── Function 2: summarize ──────────────────────────────────────────────────

class SummarizeParams(BaseModel):
    """Parameters the model must extract before summarize() runs."""

    model_config = ConfigDict(extra="forbid")
    topic: str
    max_words: int = 30


def main() -> None:
    client = _client_for_example()
    provider = _provider()

    hellos = {"en": "Hello", "es": "Hola", "fr": "Bonjour"}

    @function_calling(
        prompt=(
            "Extract the person's name and preferred language from the request. "
            "Use ISO language codes: en, es, or fr. Default to en."
        ),
        input_schema=UserRequest,
        client=client,
        provider=provider,
    )
    def greet(params: GreetParams) -> str:
        hello = hellos.get(params.language.lower(), "Hello")
        return f"{hello}, {params.name}!"

    @function_calling(
        prompt=(
            "Extract the topic and optional max_words limit from the request. "
            "Default max_words to 30 if not specified."
        ),
        input_schema=UserRequest,
        client=client,
        provider=provider,
    )
    def summarize(params: SummarizeParams) -> str:
        return f"Summary of '{params.topic}' in at most {params.max_words} words."

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result1 = greet(text="Please greet María in Spanish")
    print(f"greet    → {result1}")

    result2 = summarize(text="Give me a summary of quantum computing in 20 words")
    print(f"summarize → {result2}")


if __name__ == "__main__":
    main()
