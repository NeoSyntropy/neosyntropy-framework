"""Browser-authorized management commands for NeoSyntropy."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_API_URL = "https://api.neosyntropy.com"
KEYRING_SERVICE = "neosyntropy"


class CliError(RuntimeError):
    """An expected command-line error that should be shown without a traceback."""


def _config_path() -> Path:
    root = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "neosyntropy"
    return root / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise CliError(f"Configuration at {_config_path()} is invalid: {exc.msg}") from exc


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _keyring() -> Any:
    try:
        return importlib.import_module("keyring")
    except ImportError as exc:
        raise CliError(
            "Credential storage is unavailable. Reinstall neosyntropy with keyring."
        ) from exc


def _profile(args: argparse.Namespace) -> str:
    return args.profile


def _api_url(args: argparse.Namespace, config: dict[str, Any]) -> str:
    configured = config.get("profiles", {}).get(_profile(args), {}).get("api_url")
    return (
        args.api_url or os.getenv("NEOSYNTROPY_API_URL") or configured or DEFAULT_API_URL
    ).rstrip("/")


def _request(
    method: str,
    api_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    access_token: str | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    request = Request(f"{api_url}/api/v1{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = exc.reason
        raise CliError(f"Request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise CliError(f"Cannot reach NeoSyntropy: {exc.reason}") from exc
    return json.loads(content) if content else {}


def _save_refresh_token(profile: str, refresh_token: str) -> None:
    try:
        _keyring().set_password(KEYRING_SERVICE, profile, refresh_token)
    except Exception as exc:  # pragma: no cover - backend varies by operating system
        raise CliError(f"Could not save credentials in the system keychain: {exc}") from exc


def _refresh_token(profile: str) -> str:
    try:
        token = _keyring().get_password(KEYRING_SERVICE, profile)
    except Exception as exc:  # pragma: no cover - backend varies by operating system
        raise CliError(f"Could not read credentials from the system keychain: {exc}") from exc
    if not token:
        raise CliError("Not logged in. Run `neosyntropy login` first.")
    return token


def _access_token(args: argparse.Namespace, config: dict[str, Any]) -> tuple[str, str]:
    api_url = _api_url(args, config)
    pair = _request(
        "POST",
        api_url,
        "/auth/refresh",
        payload={"refresh_token": _refresh_token(_profile(args))},
    )
    refresh_token = pair["refresh_token"]
    _save_refresh_token(_profile(args), refresh_token)
    return api_url, pair["access_token"]


def login(args: argparse.Namespace) -> int:
    config = _load_config()
    api_url = _api_url(args, config)
    request = _request(
        "POST",
        api_url,
        "/auth/device/authorize",
        payload={"client_name": f"NeoSyntropy CLI ({platform.node() or 'local'})"},
    )
    print(f"Opening your browser to connect this CLI (code: {request['user_code']}).")
    if not webbrowser.open(request["verification_uri_complete"]):
        print(f"Open this URL manually: {request['verification_uri_complete']}")

    deadline = time.monotonic() + int(request["expires_in"])
    interval = max(1, int(request["interval"]))
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            pair = _request(
                "POST",
                api_url,
                "/auth/device/token",
                payload={"device_code": request["device_code"]},
            )
        except CliError as exc:
            if "authorization_pending" in str(exc) or "slow_down" in str(exc):
                continue
            if "expired_token" in str(exc):
                raise CliError(
                    "The browser connection expired. Run `neosyntropy login` again."
                ) from exc
            raise
        _save_refresh_token(_profile(args), pair["refresh_token"])
        profiles = config.setdefault("profiles", {})
        profiles[_profile(args)] = {
            **profiles.get(_profile(args), {}),
            "api_url": api_url,
        }
        _save_config(config)
        print("CLI connected.")
        return 0
    raise CliError("The browser connection expired. Run `neosyntropy login` again.")


def project_list(args: argparse.Namespace) -> int:
    config = _load_config()
    api_url, access_token = _access_token(args, config)
    projects = _request("GET", api_url, "/observability/projects", access_token=access_token)
    active_project = config.get("profiles", {}).get(_profile(args), {}).get("project_id")
    for project in projects:
        marker = "*" if project["id"] == active_project else " "
        print(f"{marker} {project['id']}  {project['name']} ({project['slug']})")
    return 0


def project_create(args: argparse.Namespace) -> int:
    config = _load_config()
    api_url, access_token = _access_token(args, config)
    payload = {"name": args.name}
    if args.slug:
        payload["slug"] = args.slug
    if args.description:
        payload["description"] = args.description
    project = _request(
        "POST", api_url, "/observability/projects", payload=payload, access_token=access_token
    )
    print(f"Created project {project['name']} ({project['id']}).")
    if workspace_key := project.get("workspace_api_key"):
        print("Workspace API key (shown once):", workspace_key)
    if args.use:
        _set_project(config, _profile(args), project["id"])
        print("Set as the active CLI project.")
    return 0


def _set_project(config: dict[str, Any], profile: str, project_id: str) -> None:
    profiles = config.setdefault("profiles", {})
    profiles[profile] = {**profiles.get(profile, {}), "project_id": project_id}
    _save_config(config)


def project_use(args: argparse.Namespace) -> int:
    config = _load_config()
    _set_project(config, _profile(args), args.project_id)
    print(f"Active project set to {args.project_id}.")
    return 0


def logout(args: argparse.Namespace) -> int:
    _keyring().delete_password(KEYRING_SERVICE, _profile(args))
    print("CLI credentials removed.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neosyntropy")
    parser.add_argument(
        "--profile", default="default", help="Stored credential profile (default: %(default)s)"
    )
    parser.add_argument("--api-url", help="NeoSyntropy API base URL")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login").set_defaults(handler=login)
    commands.add_parser("logout").set_defaults(handler=logout)
    project = commands.add_parser("project")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_commands.add_parser("list").set_defaults(handler=project_list)
    create = project_commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("--slug")
    create.add_argument("--description")
    create.add_argument("--use", action="store_true", help="Make the new project active")
    create.set_defaults(handler=project_create)
    use = project_commands.add_parser("use")
    use.add_argument("project_id")
    use.set_defaults(handler=project_use)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except CliError as exc:
        print(f"neosyntropy: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
