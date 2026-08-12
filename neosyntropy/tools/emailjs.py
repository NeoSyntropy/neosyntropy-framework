"""EmailJS adapter and tool registration.

Agno-style external service toolkit: credentials are provided by the caller
(backend / app). Content is rendered elsewhere and sent through one shared
EmailJS shell template (``to_email`` / ``subject`` / ``html_body``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .registry import ToolRegistry, tool

EMAILJS_SEND_URL = "https://api.emailjs.com/api/v1.0/email/send"
# Cloudflare in front of api.emailjs.com rejects bare Python user-agents (CF 1010).
_EMAILJS_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (compatible; NeoSyntropyEmailJS/1.0; "
        "+https://neosyntropy.com)"
    ),
    "Origin": "http://localhost",
}


class EmailJSError(RuntimeError):
    """Raised when EmailJS is misconfigured or the API call fails."""


@dataclass(frozen=True, slots=True)
class EmailJSCredentials:
    service_id: str
    public_key: str
    private_key: str
    template_id: str

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("service_id", self.service_id),
                ("public_key", self.public_key),
                ("private_key", self.private_key),
                ("template_id", self.template_id),
            )
            if not value
        ]
        if missing:
            raise EmailJSError(
                "EmailJS is not configured. Missing: " + ", ".join(missing)
            )


class EmailJSClient:
    """Thin EmailJS REST client for the shared shell template."""

    def __init__(
        self,
        *,
        service_id: str,
        public_key: str,
        private_key: str,
        template_id: str,
        timeout: float = 10.0,
    ) -> None:
        self.credentials = EmailJSCredentials(
            service_id=service_id,
            public_key=public_key,
            private_key=private_key,
            template_id=template_id,
        )
        self.timeout = timeout

    def _payload(
        self,
        *,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
    ) -> dict[str, Any]:
        self.credentials.validate()
        template_params: dict[str, str] = {
            "to_email": to_email,
            "subject": subject,
            "html_body": html_body,
        }
        if text_body is not None:
            template_params["text_body"] = text_body
        return {
            "service_id": self.credentials.service_id,
            "template_id": self.credentials.template_id,
            "user_id": self.credentials.public_key,
            "accessToken": self.credentials.private_key,
            "template_params": template_params,
        }

    def send(
        self,
        *,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
    ) -> dict[str, str]:
        """Send via EmailJS (sync). Returns a small status dict for tool audits."""
        payload = self._payload(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
        request = urllib.request.Request(
            EMAILJS_SEND_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(_EMAILJS_HEADERS),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    raise EmailJSError(f"EmailJS delivery failed with status {status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise EmailJSError(
                f"EmailJS delivery failed with status {exc.code}{suffix}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmailJSError("EmailJS delivery failed") from exc
        return {"ok": "true", "to_email": to_email}

    async def asend(
        self,
        *,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
    ) -> dict[str, str]:
        """Send via EmailJS (async). Requires ``httpx``."""
        try:
            import httpx
        except ImportError as exc:
            raise EmailJSError(
                "httpx is required for EmailJSClient.asend; use send() or install httpx"
            ) from exc

        payload = self._payload(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    EMAILJS_SEND_URL,
                    json=payload,
                    headers=dict(_EMAILJS_HEADERS),
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = (exc.response.text or "").strip()
            suffix = f": {detail}" if detail else ""
            raise EmailJSError(
                f"EmailJS delivery failed with status {exc.response.status_code}{suffix}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmailJSError("EmailJS delivery failed") from exc
        return {"ok": "true", "to_email": to_email}


class SendEmailArgs(BaseModel):
    to_email: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    html_body: str = Field(description="HTML body (rendered before send)")
    text_body: str | None = Field(
        default=None,
        description="Optional plain-text body",
    )


class EmailJSTools:
    """Register EmailJS send_email on a ToolRegistry (Agno-style toolkit)."""

    def __init__(
        self,
        *,
        service_id: str,
        public_key: str,
        private_key: str,
        template_id: str,
        timeout: float = 10.0,
    ) -> None:
        self.client = EmailJSClient(
            service_id=service_id,
            public_key=public_key,
            private_key=private_key,
            template_id=template_id,
            timeout=timeout,
        )

    def register(self, registry: ToolRegistry | None = None) -> ToolRegistry:
        target = registry or ToolRegistry()

        @tool(registry=target, name="send_email")
        def send_email(args: SendEmailArgs) -> dict[str, str]:
            """Send an email through the configured EmailJS shell template."""
            return self.client.send(
                to_email=args.to_email,
                subject=args.subject,
                html_body=args.html_body,
                text_body=args.text_body,
            )

        return target
