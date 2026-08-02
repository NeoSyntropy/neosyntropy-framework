from __future__ import annotations

from typing import Any

from neosyntropy import (
    BackendClient,
    BackendError,
    BackendProvider,
    Candidate,
    RunContext,
    SemanticRouter,
    Topology,
)

from .conftest import build_graph


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def generate(self, prompt, *, schema=None, purpose="node", **extra):
        payload = {"prompt": prompt, "schema": schema, "purpose": purpose, **extra}
        self.calls.append(("generate", payload))
        return "generated"

    async def route(self, context, candidates, *, category="general"):
        self.calls.append(("route", {"category": category, "candidates": candidates}))
        from neosyntropy import RoutingPlan

        return RoutingPlan(
            reasoning="backend", topology=Topology.SEQUENTIAL, execution_plan=[[0]]
        )


def test_backend_adapters_do_not_accept_provider_or_model() -> None:
    import asyncio

    client = RecordingBackend()
    context = RunContext(request_id="req-1", intent="refund", current_state="Start")
    graph = build_graph()
    candidates = [
        Candidate(
            node_id=node.id,
            name=node.name,
            description=node.description,
            prerequisites=node.prerequisites,
            is_fallback=node.is_fallback,
        )
        for node in graph.nodes.values()
    ]

    plan = asyncio.run(SemanticRouter(client).route(context, candidates))
    generated = asyncio.run(BackendProvider(client).generate("hello", schema={"type": "object"}))

    assert plan.execution_plan == [[0]]
    assert generated == "generated"
    assert [name for name, _ in client.calls] == ["route", "generate"]
    assert all("provider" not in payload and "model" not in payload for _, payload in client.calls)


def test_backend_client_reads_api_key_and_project_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NEOSYNTROPY_API_URL", "https://api.example.test")
    monkeypatch.setenv("NEOSYNTROPY_API_KEY", "api-key")
    monkeypatch.setenv("NEOSYNTROPY_PROJECT_ID", "project-id")
    monkeypatch.delenv("NEOSYNTROPY_ACCESS_TOKEN", raising=False)

    client = BackendClient.from_env()

    assert client is not None
    assert client.api_key == "api-key"
    assert client.project_id == "project-id"
    assert client.access_token is None


def test_backend_client_env_retains_access_token_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("NEOSYNTROPY_API_URL", "https://api.example.test")
    monkeypatch.setenv("NEOSYNTROPY_ACCESS_TOKEN", "access-token")
    monkeypatch.delenv("NEOSYNTROPY_API_KEY", raising=False)
    monkeypatch.delenv("NEOSYNTROPY_PROJECT_ID", raising=False)

    client = BackendClient.from_env()

    assert client is not None
    assert client.access_token == "access-token"
    assert client.api_key is None


def test_backend_client_api_key_env_requires_project(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("NEOSYNTROPY_API_URL", "https://api.example.test")
    monkeypatch.setenv("NEOSYNTROPY_API_KEY", "api-key")
    monkeypatch.delenv("NEOSYNTROPY_PROJECT_ID", raising=False)
    monkeypatch.delenv("NEOSYNTROPY_ACCESS_TOKEN", raising=False)

    with pytest.raises(BackendError, match="NEOSYNTROPY_PROJECT_ID"):
        BackendClient.from_env()
