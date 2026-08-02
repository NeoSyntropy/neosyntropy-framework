from __future__ import annotations

from pathlib import Path
from typing import Any

from neosyntropy import cli


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def test_login_opens_browser_and_saves_only_refresh_token(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    keyring = FakeKeyring()
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    responses = iter(
        [
            {
                "device_code": "device-secret",
                "user_code": "ABCDEFGH",
                "verification_uri_complete": "https://app.example/cli/approve?user_code=ABCDEFGH",
                "expires_in": 300,
                "interval": 1,
            },
            {"access_token": "short-lived", "refresh_token": "refresh-secret"},
        ]
    )
    monkeypatch.setattr(cli, "_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(cli, "_keyring", lambda: keyring)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: url.startswith("https://"))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_request",
        lambda method, api_url, path, **kwargs: (
            calls.append((method, path, kwargs.get("payload"))), next(responses)
        )[1],
    )

    assert cli.main(["--api-url", "https://api.example", "login"]) == 0
    assert keyring.get_password("neosyntropy", "default") == "refresh-secret"
    assert "short-lived" not in (tmp_path / "config.json").read_text(encoding="utf-8")
    assert calls == [
        ("POST", "/auth/device/authorize", {"client_name": calls[0][2]["client_name"]}),
        ("POST", "/auth/device/token", {"device_code": "device-secret"}),
    ]
    assert "CLI connected." in capsys.readouterr().out


def test_project_create_refreshes_session_and_selects_project(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    keyring = FakeKeyring()
    keyring.set_password("neosyntropy", "default", "old-refresh")
    responses = iter(
        [
            {"access_token": "new-access", "refresh_token": "new-refresh"},
            {"id": "project-1", "name": "Demo", "slug": "demo", "workspace_api_key": "nsk_secret"},
        ]
    )
    monkeypatch.setattr(cli, "_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(cli, "_keyring", lambda: keyring)
    monkeypatch.setattr(cli, "_request", lambda *args, **kwargs: next(responses))

    assert cli.main(["--api-url", "https://api.example", "project", "create", "Demo", "--use"]) == 0
    assert keyring.get_password("neosyntropy", "default") == "new-refresh"
    assert '"project_id": "project-1"' in (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "shown once" in capsys.readouterr().out
