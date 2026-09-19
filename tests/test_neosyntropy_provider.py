"""NeoSyntropy chat provider keeps adapter kwargs out of the HTTP body."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import patch
from urllib.request import Request

import pytest

from neosyntropy.providers.neosyntropy import ChatResult, NeoSyntropyProvider


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _gateway_reply(content: str = "ok") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def test_chat_drops_framework_objects_and_keeps_api_fields() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data or b"{}")
        captured["timeout"] = timeout
        return _FakeResponse(_gateway_reply("hello"))

    provider = NeoSyntropyProvider("nsk_test", "default")
    with patch("neosyntropy.providers.neosyntropy.urllib.request.urlopen", fake_urlopen):
        result = provider.chat(
            [{"role": "user", "content": "Hi"}],
            assistant_message=object(),
            run_response=object(),
            compress_tool_results=False,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

    assert isinstance(result, ChatResult)
    assert result.content == "hello"
    assert captured["url"].endswith("/api/v1/chat/completions")
    body = captured["body"]
    assert body["model"] == "default"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
    assert body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert "assistant_message" not in body
    assert "run_response" not in body
    assert "compress_tool_results" not in body


def test_agno_style_invoke_kwargs_do_not_crash() -> None:
    """Agno Model.response() forwards assistant_message / run_response into invoke()."""

    class _Message:
        def __init__(self, role: str, content: str | None = None) -> None:
            self.role = role
            self.content = content

    def fake_urlopen(request: Request, timeout: float | None = None) -> _FakeResponse:
        json.loads(request.data or b"{}")
        return _FakeResponse(_gateway_reply())

    provider = NeoSyntropyProvider("nsk_test", "default")
    with patch("neosyntropy.providers.neosyntropy.urllib.request.urlopen", fake_urlopen):
        result = provider.chat(
            [_Message("user", "Hello")],
            assistant_message=_Message("assistant"),
            tools=[],
            tool_choice=None,
            run_response=BytesIO(b"not-json-payload"),
            compress_tool_results=False,
        )

    assert result.content == "ok"


def test_agno_adapter_invoke_accepts_framework_kwargs() -> None:
    pytest.importorskip("agno")
    from neosyntropy.integrations.agno.model import NeoSyntropyModel

    def fake_urlopen(request: Request, timeout: float | None = None) -> _FakeResponse:
        body = json.loads(request.data or b"{}")
        assert "assistant_message" not in body
        assert "run_response" not in body
        return _FakeResponse(_gateway_reply("from-agno"))

    model = NeoSyntropyModel(api_key="nsk_test", model="default")
    with patch("neosyntropy.providers.neosyntropy.urllib.request.urlopen", fake_urlopen):
        response = model.invoke(
            [{"role": "user", "content": "Hello"}],
            assistant_message=object(),
            run_response=object(),
            tools=None,
            tool_choice=None,
            compress_tool_results=False,
        )

    assert response.content == "from-agno"
