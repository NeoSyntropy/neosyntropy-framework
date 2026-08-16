from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from neosyntropy.backend import BackendClient, BackendError, Client, DEFAULT_API_URL


def test_client_accepts_explicit_project_id() -> None:
    client = Client(api_key="nsk_test", project_id="proj-1")
    assert client.project_id == "proj-1"
    assert client._as_backend().project_id == "proj-1"


def test_client_api_key_only_requires_create_or_project_id_before_run() -> None:
    client = Client(api_key="nsk_test")
    assert client.project_id is None
    with pytest.raises(ValueError, match="create_project"):
        client._as_backend()


def test_create_project_posts_when_slug_missing_and_binds_id() -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_get(self: BackendClient, path: str, **_: Any) -> list[dict[str, Any]]:
        calls.append(("GET", path, None))
        return []

    def fake_post(
        self: BackendClient, path: str, payload: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        calls.append(("POST", path, payload))
        return {"id": "proj-new", "name": payload["name"], "slug": payload["slug"]}

    client = Client(api_key="nsk_test")
    with (
        patch.object(BackendClient, "_get", fake_get),
        patch.object(BackendClient, "_post", fake_post),
    ):
        project = client.create_project("Support Bot", "support-bot")

    assert project["id"] == "proj-new"
    assert client.project_id == "proj-new"
    assert client._as_backend().project_id == "proj-new"
    assert calls == [
        ("GET", "/observability/projects", None),
        (
            "POST",
            "/observability/projects",
            {"name": "Support Bot", "slug": "support-bot"},
        ),
    ]


def test_create_project_reuses_existing_slug_without_post() -> None:
    existing = {"id": "proj-existing", "name": "Support Bot", "slug": "support-bot"}

    def fake_get(self: BackendClient, path: str, **_: Any) -> list[dict[str, Any]]:
        return [existing]

    def fake_post(self: BackendClient, *_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("create must not POST when slug exists")

    client = Client(api_key="nsk_test")
    with (
        patch.object(BackendClient, "_get", fake_get),
        patch.object(BackendClient, "_post", fake_post),
    ):
        project = client.create_project("Support Bot", "support-bot")

    assert project is existing
    assert client.project_id == "proj-existing"


def test_create_project_retries_list_on_conflict() -> None:
    listed: list[list[dict[str, Any]]] = [
        [],
        [{"id": "proj-race", "name": "Support Bot", "slug": "support-bot"}],
    ]

    def fake_get(self: BackendClient, path: str, **_: Any) -> list[dict[str, Any]]:
        return listed.pop(0)

    def fake_post(self: BackendClient, *_: Any, **__: Any) -> dict[str, Any]:
        raise BackendError("backend returned HTTP 409: slug exists", http_status=409)

    client = Client(api_key="nsk_test")
    with (
        patch.object(BackendClient, "_get", fake_get),
        patch.object(BackendClient, "_post", fake_post),
    ):
        project = client.create_project("Support Bot", "support-bot")

    assert project["id"] == "proj-race"
    assert client.project_id == "proj-race"


def test_list_projects_returns_backend_list() -> None:
    projects = [
        {"id": "a", "slug": "alpha"},
        {"id": "b", "slug": "beta"},
    ]

    def fake_get(self: BackendClient, path: str, **_: Any) -> list[dict[str, Any]]:
        assert path == "/observability/projects"
        return projects

    client = Client(api_key="nsk_test")
    with patch.object(BackendClient, "_get", fake_get):
        assert client.list_projects() == projects


def test_from_env_allows_api_key_without_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEOSYNTROPY_API_KEY", "nsk_env")
    monkeypatch.delenv("NEOSYNTROPY_PROJECT_ID", raising=False)
    monkeypatch.delenv("NEOSYNTROPY_API_URL", raising=False)
    monkeypatch.delenv("NEOSYNTROPY_ACCESS_TOKEN", raising=False)
    client = BackendClient.from_env()
    assert client is not None
    assert client.api_key == "nsk_env"
    assert client.project_id is None
    assert client.base_url == f"{DEFAULT_API_URL}/api/v1"


def test_from_env_still_accepts_explicit_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEOSYNTROPY_API_KEY", "nsk_env")
    monkeypatch.setenv("NEOSYNTROPY_PROJECT_ID", "proj-env")
    monkeypatch.delenv("NEOSYNTROPY_API_URL", raising=False)
    monkeypatch.delenv("NEOSYNTROPY_ACCESS_TOKEN", raising=False)
    client = BackendClient.from_env()
    assert client is not None
    assert client.project_id == "proj-env"
