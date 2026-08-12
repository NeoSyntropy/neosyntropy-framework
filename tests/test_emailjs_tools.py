from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from neosyntropy.tools.emailjs import EmailJSClient, EmailJSError, EmailJSTools, SendEmailArgs
from neosyntropy.tools.registry import ToolRegistry


def test_send_posts_shell_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Request, timeout: float = 0) -> FakeResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("neosyntropy.tools.emailjs.urllib.request.urlopen", fake_urlopen)

    client = EmailJSClient(
        service_id="service_x",
        public_key="pub",
        private_key="priv",
        template_id="template_shell",
    )
    result = client.send(
        to_email="person@example.com",
        subject="Hello",
        html_body="<p>Hi</p>",
        text_body="Hi",
    )

    assert result == {"ok": "true", "to_email": "person@example.com"}
    assert captured["url"] == "https://api.emailjs.com/api/v1.0/email/send"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["service_id"] == "service_x"
    assert body["template_id"] == "template_shell"
    assert body["user_id"] == "pub"
    assert body["accessToken"] == "priv"
    assert body["template_params"] == {
        "to_email": "person@example.com",
        "subject": "Hello",
        "html_body": "<p>Hi</p>",
        "text_body": "Hi",
    }


def test_send_requires_credentials() -> None:
    client = EmailJSClient(
        service_id="",
        public_key="pub",
        private_key="priv",
        template_id="template_shell",
    )
    with pytest.raises(EmailJSError, match="service_id"):
        client.send(to_email="a@b.com", subject="s", html_body="<p>x</p>")


def test_send_maps_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: Request, timeout: float = 0) -> object:
        raise HTTPError(request.full_url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr("neosyntropy.tools.emailjs.urllib.request.urlopen", boom)
    client = EmailJSClient(
        service_id="service_x",
        public_key="pub",
        private_key="priv",
        template_id="template_shell",
    )
    with pytest.raises(EmailJSError, match="403"):
        client.send(to_email="a@b.com", subject="s", html_body="<p>x</p>")


def test_emailjs_tools_registers_send_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "neosyntropy.tools.emailjs.EmailJSClient.send",
        lambda self, **kwargs: {"ok": "true", "to_email": kwargs["to_email"]},
    )
    registry = ToolRegistry()
    EmailJSTools(
        service_id="service_x",
        public_key="pub",
        private_key="priv",
        template_id="template_shell",
    ).register(registry)

    assert "send_email" in registry.names()
    invocation = registry.invoke(
        "send_email",
        {
            "to_email": "person@example.com",
            "subject": "Hi",
            "html_body": "<p>Hi</p>",
        },
    )
    assert invocation.ok
    assert invocation.result == {"ok": "true", "to_email": "person@example.com"}
    SendEmailArgs.model_validate(
        {"to_email": "a@b.com", "subject": "s", "html_body": "<p>x</p>"}
    )
