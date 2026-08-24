from __future__ import annotations

import pytest

from neosyntropy.tools.email.email import EmailTools


def test_email_user_sends_email(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeSMTP:
        def __init__(self, host: str, port: int) -> None:
            captured["host"] = host
            captured["port"] = port

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def login(self, user: str, password: str) -> None:
            captured["user"] = user
            captured["password"] = password

        def send_message(self, msg: object) -> None:
            captured["msg"] = msg

    monkeypatch.setattr("smtplib.SMTP_SSL", FakeSMTP)

    tools = EmailTools(
        receiver_email="recipient@example.com",
        sender_name="Sender",
        sender_email="sender@example.com",
        sender_passkey="passkey123",
    )
    result = tools.email_user(subject="Hello Subject", body="Hello Body")

    assert result == "email sent successfully"
    assert captured["host"] == "smtp.gmail.com"
    assert captured["port"] == 465
    assert captured["user"] == "sender@example.com"
    assert captured["password"] == "passkey123"
    msg = captured["msg"]
    assert msg["Subject"] == "Hello Subject"
    assert msg["From"] == "Sender <sender@example.com>"
    assert msg["To"] == "recipient@example.com"


def test_email_user_requires_credentials() -> None:
    # Missing receiver_email
    tools_no_rec = EmailTools(
        receiver_email=None,
        sender_name="Sender",
        sender_email="sender@example.com",
        sender_passkey="passkey123",
    )
    assert "No receiver email provided" in tools_no_rec.email_user("s", "b")

    # Missing sender_email
    tools_no_sender = EmailTools(
        receiver_email="recipient@example.com",
        sender_name="Sender",
        sender_email=None,
        sender_passkey="passkey123",
    )
    assert "No sender email provided" in tools_no_sender.email_user("s", "b")

    # Missing sender_passkey
    tools_no_key = EmailTools(
        receiver_email="recipient@example.com",
        sender_name="Sender",
        sender_email="sender@example.com",
        sender_passkey=None,
    )
    assert "No sender passkey provided" in tools_no_key.email_user("s", "b")
