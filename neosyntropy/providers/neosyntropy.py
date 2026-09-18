"""OpenAI-compatible NeoSyntropy chat provider."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class NeoSyntropyProviderError(RuntimeError):
    """Raised when the NeoSyntropy chat gateway rejects a request."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    raw: dict[str, Any]
    tool_calls: tuple[dict[str, Any], ...] = ()


def _content(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, (str, list)):
        return value
    return str(value)


def _message(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {
            "role": getattr(value, "type", getattr(value, "role", "user")),
            "content": getattr(value, "content", str(value)),
        }
    role = result.get("role", "user")
    role = {"human": "user", "ai": "assistant"}.get(role, role)
    result["role"] = role
    result["content"] = _content(result.get("content", ""))
    return result


class NeoSyntropyProvider:
    """Route chat traffic to ``/api/v1/chat/completions``.

    The class has no framework dependency. LangGraph and Agno adapters are
    optional wrappers around this provider, so the base package remains small.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.neosyntropy.com",
        project_id: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        root = base_url.rstrip("/")
        self.base_url = root if root.endswith("/api/v1") else f"{root}/api/v1"
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.timeout = timeout

    @classmethod
    def from_env(cls, *, model: str | None = None, **kwargs: Any) -> NeoSyntropyProvider:
        api_key = os.getenv("NEOSYNTROPY_API_KEY")
        if not api_key:
            raise ValueError("NEOSYNTROPY_API_KEY is not set")
        return cls(
            api_key,
            model or os.getenv("NEOSYNTROPY_MODEL", "default"),
            base_url=os.getenv("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com"),
            project_id=os.getenv("NEOSYNTROPY_PROJECT_ID"),
            **kwargs,
        )

    def _payload(self, messages: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": [_message(item) for item in messages],
        }
        if self.project_id:
            payload["metadata"] = {"project_id": self.project_id}
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return payload

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.project_id:
            headers["X-NeoSyntropy-Project-ID"] = self.project_id
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise NeoSyntropyProviderError(
                f"NeoSyntropy chat gateway returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise NeoSyntropyProviderError("NeoSyntropy chat gateway is unavailable") from exc
        if not isinstance(result, dict):
            raise NeoSyntropyProviderError("NeoSyntropy chat gateway returned invalid JSON")
        return result

    @staticmethod
    def _result(raw: dict[str, Any]) -> ChatResult:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = tuple(message.get("tool_calls") or ())
        return ChatResult(content=str(message.get("content") or ""), raw=raw, tool_calls=calls)

    def chat(self, messages: Iterable[Any], **kwargs: Any) -> ChatResult:
        return self._result(self._request(self._payload(messages, **kwargs)))

    def invoke(self, messages: Iterable[Any], **kwargs: Any) -> str:
        return self.chat(messages, **kwargs).content

    def generate(
        self, prompt: str, *, schema: dict[str, Any] | None = None, tools: Any = None
    ) -> str:
        kwargs: dict[str, Any] = {"tools": tools}
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        return self.invoke([{"role": "user", "content": prompt}], **kwargs)

    async def ainvoke(self, messages: Iterable[Any], **kwargs: Any) -> str:
        return await asyncio.to_thread(self.invoke, messages, **kwargs)

    async def achat(self, messages: Iterable[Any], **kwargs: Any) -> ChatResult:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def astream(self, messages: Iterable[Any], **kwargs: Any) -> AsyncIterator[str]:
        # The gateway may return a single normalized chunk while streaming is
        # enabled; this preserves a common async iterator contract for adapters.
        yield (await self.achat(messages, **kwargs)).content
