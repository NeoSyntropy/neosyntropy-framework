from __future__ import annotations

import os
import sys

from neosyntropy.tools.email.email import EmailTools


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    receiver_email = _require_env("RECIPIENT_EMAIL")
    sender_name = os.environ.get("SENDER_NAME", "NeoSyntropy Cookbook").strip() or "NeoSyntropy Cookbook"
    sender_email = _require_env("SENDER_EMAIL")
    sender_passkey = _require_env("SENDER_PASSKEY")

    tools = EmailTools(
        receiver_email=receiver_email,
        sender_name=sender_name,
        sender_email=sender_email,
        sender_passkey=sender_passkey,
    )

    subject = "NeoSyntropy cookbook email"
    body = (
        "This message was sent by the NeoSyntropy cookbook email example.\n\n"
        "It demonstrates the EmailTools toolkit with real SMTP credentials."
    )

    result = tools.email_user(subject=subject, body=body)
    print(result)


if __name__ == "__main__":
    sys.exit(main())
