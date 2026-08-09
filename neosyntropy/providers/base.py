"""Provider protocol: the execution backend for nodes without handlers."""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    def generate(
        self, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> str | Awaitable[str]:
        """Generate a completion; may be sync or async."""
        ...


class ProviderRegistry:
    def __init__(self, providers: dict[str, Provider] | None = None):
        self._providers: dict[str, Provider] = dict(providers or {})

    def register(self, name: str, provider: Provider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> Provider:
        if name in self._providers:
            return self._providers[name]
        # If the provider is unknown locally but we have the default backend
        # provider, delegate to it — the backend owns open-model routing.
        if "neosyntropy/base" in self._providers:
            return self._providers["neosyntropy/base"]
        raise KeyError(
            f"Unknown provider {name!r}; registered: {sorted(self._providers)}"
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)
