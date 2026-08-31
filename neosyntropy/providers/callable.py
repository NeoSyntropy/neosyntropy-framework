"""Wrap any function or LLM client call as a provider."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


class CallableProvider:
    """Adapts ``fn(prompt, schema=None) -> str`` (sync or async) to Provider."""

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn
        self._accepts_schema = "schema" in inspect.signature(fn).parameters

    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        tools: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if self._accepts_schema and schema is not None:
            kwargs["schema"] = schema
        if "tools" in inspect.signature(self._fn).parameters and tools is not None:
            kwargs["tools"] = tools
        return self._fn(prompt, **kwargs)
