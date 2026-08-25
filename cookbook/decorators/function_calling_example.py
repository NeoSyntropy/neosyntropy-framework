"""Simple @function_calling: extract parameters, then run the function.

Run::

    python cookbook/decorators/function_calling_example.py
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from neosyntropy import Client, function_calling

TESTS_ENV_PATH = Path(__file__).resolve().parents[2] / "tests" / ".env"


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


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class GreetParams(BaseModel):
    """Parameters the model must extract before greet() runs."""

    model_config = ConfigDict(extra="forbid")
    name: str
    language: str = "en"


def main() -> None:
    _load_tests_env()
    client = Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        project_id=_require_env("NEOSYNTROPY_PROJECT_ID"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com").strip()
        or "https://api.neosyntropy.com",
    )

    hellos = {"en": "Hello", "es": "Hola", "fr": "Bonjour"}

    @function_calling(
        prompt=(
            "Extract the person's name and preferred language from the request. "
            "Use ISO language codes: en, es, or fr. Default to en."
        ),
        input_schema=UserRequest,
        client=client,
    )
    def greet(params: GreetParams) -> str:
        hello = hellos.get(params.language.lower(), "Hello")
        return f"{hello}, {params.name}!"

    result = greet(text="Please greet María in Spanish")
    print(result) 


if __name__ == "__main__":
    main()
