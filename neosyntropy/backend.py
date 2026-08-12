"""Client adapters for NeoSyntropy's backend-owned inference services."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core.context import RunContext
from .core.models import Candidate, RoutingPlan


class BackendError(RuntimeError):
    """The NeoSyntropy backend rejected or could not serve a request."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


INFERENCE_WARMING_CODE = "inference_warming"
INFERENCE_WARMING_MESSAGE = (
    "Inference is still warming up (GPU cold start, often 1–2 minutes). "
    "Retry shortly."
)

DEFAULT_API_URL = "https://api.neosyntropy.com"


class Client:
    """Project-scoped NeoSyntropy client for application code.

    Developers only supply an API key and project id. Pass this to
    :meth:`FSM.run` — control / inference plumbing stays inside the framework.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str,
        base_url: str = DEFAULT_API_URL,
        timeout: float = 180.0,
        # Neon round-trips often exceed 2s; short budgets orphan runs with 0 events.
        telemetry_timeout: float = 15.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not project_id:
            raise ValueError("project_id is required")
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = base_url
        self._backend = BackendClient(
            base_url,
            api_key=api_key,
            project_id=project_id,
            timeout=timeout,
            telemetry_timeout=telemetry_timeout,
        )

    def _as_backend(self) -> BackendClient:
        return self._backend


class BackendClient:
    """Authenticated client whose requests never name a model or provider."""

    def __init__(
        self,
        base_url: str,
        access_token: str | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        timeout: float = 180.0,
        telemetry_timeout: float = 15.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if not access_token and not api_key:
            raise ValueError("access_token or api_key is required")
        if api_key and not project_id:
            raise ValueError("project_id is required when api_key is used")
        if telemetry_timeout <= 0:
            raise ValueError("telemetry_timeout must be positive")
        base = base_url.rstrip("/")
        self.base_url = base if base.endswith("/api/v1") else f"{base}/api/v1"
        self.access_token = access_token
        self.api_key = api_key
        self.project_id = project_id
        self.timeout = timeout
        self.telemetry_timeout = telemetry_timeout

    @classmethod
    def from_env(cls) -> BackendClient | None:
        base_url = os.getenv("NEOSYNTROPY_API_URL")
        access_token = os.getenv("NEOSYNTROPY_ACCESS_TOKEN")
        api_key = os.getenv("NEOSYNTROPY_API_KEY")
        project_id = os.getenv("NEOSYNTROPY_PROJECT_ID")
        if not any((base_url, access_token, api_key, project_id)):
            return None
        if not base_url or not (access_token or api_key):
            raise BackendError(
                "NEOSYNTROPY_API_URL and either NEOSYNTROPY_ACCESS_TOKEN or "
                "NEOSYNTROPY_API_KEY must be set together"
            )
        if api_key and not project_id:
            raise BackendError(
                "NEOSYNTROPY_PROJECT_ID must be set with NEOSYNTROPY_API_KEY"
            )
        return cls(
            base_url,
            access_token,
            api_key=api_key,
            project_id=project_id,
        )

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post, path, payload)

    async def telemetry_run_started(
        self,
        *,
        request_id: str,
        initial_state: str,
        manifest: dict[str, Any],
        input: Any = None,
    ) -> str | None:
        payload: dict[str, Any] = {
            "external_id": request_id,
            "name": "control-cycle",
            "metadata": {
                "initial_state": initial_state,
                "graph": manifest,
            },
        }
        if input is not None:
            payload["input"] = input
        response = await self._telemetry_post("/telemetry/runs", payload)
        if response is None:
            return None
        run_id = response.get("id", response.get("run_id"))
        return str(run_id) if run_id is not None else None

    async def telemetry_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        external_id: str,
        sequence: int,
    ) -> None:
        await self._telemetry_post(
            f"/telemetry/runs/{run_id}/events",
            {
                "external_id": external_id,
                "sequence": sequence,
                "event_type": event_type,
                "payload": payload,
            },
        )

    async def telemetry_run_finished(
        self,
        run_id: str,
        *,
        status: str,
        final_state: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        wire_status = {
            "completed": "succeeded",
            "rejected": "cancelled",
            "failed": "failed",
            "cancelled": "cancelled",
            "succeeded": "succeeded",
        }.get(status, "failed")
        await self._telemetry_post(
            f"/telemetry/runs/{run_id}/finish",
            {
                "status": wire_status,
                "output": {"final_state": final_state, **(output or {})},
            },
        )

    async def _telemetry_post(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send bounded telemetry; all failures are intentionally discarded."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._post, path, payload, timeout=self.telemetry_timeout
                ),
                timeout=self.telemetry_timeout + 0.1,
            )
        except Exception:
            return None

    async def generate(
        self,
        prompt: str | None = None,
        *,
        schema: dict[str, Any] | None = None,
        purpose: str = "node",
        node: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate text. Prefer ``node`` + ``context`` so the backend builds the prompt.

        ``node`` may also accompany a prebuilt ``prompt`` (tool-loop continuations
        and extractors) so the backend still sees the node's provider identity
        (e.g. a Vertex model id).
        """
        payload: dict[str, Any] = {"purpose": purpose, "schema": schema}
        if node is not None:
            payload["node"] = node
            if context is not None:
                payload["context"] = context
            if tools is not None:
                payload["tools"] = tools
            if prompt:
                payload["prompt"] = prompt
        elif prompt:
            payload["prompt"] = prompt
        else:
            raise BackendError("generate requires prompt or node declaration")
        response = await self.post("/framework/inference", payload)
        text = response.get("text")
        if not isinstance(text, str):
            raise BackendError("backend inference response has no text")
        return text

    async def start_control_run(
        self,
        graph_manifest: dict[str, Any],
        request: dict[str, Any],
        *,
        category: str = "general",
    ) -> dict[str, Any]:
        """Start a backend-owned control run. Response never includes plan internals."""
        return await self.post(
            "/control/runs",
            {
                "graph": _control_api_graph(graph_manifest),
                "request": request,
                "category": category,
            },
        )

    async def submit_control_results(
        self,
        run_id: str,
        *,
        results: list[dict[str, Any]] | None = None,
        client_rejection: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if results is not None:
            payload["results"] = results
        if client_rejection is not None:
            payload["client_rejection"] = client_rejection
        return await self.post(f"/control/runs/{run_id}/results", payload)

    async def route(
        self,
        context: RunContext,
        candidates: list[Candidate],
        *,
        category: str = "general",
    ) -> RoutingPlan:
        """Legacy direct router call. Prefer :meth:`start_control_run`."""
        response = await self.post(
            "/framework/router",
            {
                "context": _wire_context(context),
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in candidates
                ],
                "category": category,
            },
        )
        try:
            return RoutingPlan.model_validate(response)
        except ValueError as exc:
            raise BackendError(f"backend returned an invalid routing plan: {exc}") from exc

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        token = self.api_key or self.access_token
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.project_id:
            headers["X-NeoSyntropy-Project-ID"] = self.project_id
        request = Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=json.dumps(payload, separators=(",", ":"), default=str).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                decoded = json.loads(response.read().decode())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            code: str | None = None
            try:
                payload = json.loads(detail)
                nested = payload.get("detail", detail)
                if isinstance(nested, dict):
                    code = nested.get("code")
                    detail = nested.get("message", nested)
                else:
                    detail = nested
            except json.JSONDecodeError:
                pass
            detail_text = str(detail)
            warming = (
                exc.code == 503
                or code == INFERENCE_WARMING_CODE
                or INFERENCE_WARMING_CODE in detail_text.lower()
                or "still loading the gpu model" in detail_text.lower()
                or "vllm adapter load failed" in detail_text.lower()
            )
            if warming:
                raise BackendError(
                    INFERENCE_WARMING_MESSAGE,
                    code=INFERENCE_WARMING_CODE,
                ) from exc
            raise BackendError(
                f"backend returned HTTP {exc.code}: {detail_text}",
                code=code if isinstance(code, str) else None,
            ) from exc
        except URLError as exc:
            raise BackendError(f"cannot reach NeoSyntropy backend: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("backend returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise BackendError("backend response must be a JSON object")
        return decoded


def _wire_context(context: RunContext) -> dict[str, Any]:
    """Strip fields the backend wire models reject (e.g. message metadata)."""
    payload = context.model_dump(mode="json")
    payload["history"] = [
        {"role": message["role"], "content": message["content"]}
        for message in payload.get("history", [])
    ]
    return payload


class BackendProvider:
    def __init__(self, client: BackendClient, *, purpose: str = "node") -> None:
        self.client = client
        self.purpose = purpose

    async def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        node: Any = None,
        context: Any = None,
        tools: Any = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "schema": schema,
            "purpose": self.purpose,
        }
        if node is not None:
            kwargs["node"] = _wire_node_declaration(node)
            if context is not None:
                kwargs["context"] = _wire_context(context)
                if tools is not None:
                    kwargs["tools"] = _wire_tool_catalog(tools)
                # First turn: backend assembles the prompt from declaration + context.
                return await self.client.generate(None, **kwargs)
            # Continuations / extractors: keep the prebuilt prompt; node carries
            # provider identity (Vertex model ids, etc.).
            return await self.client.generate(prompt, **kwargs)
        return await self.client.generate(prompt, **kwargs)


# SDK-local provider registry name. Selects BackendProvider on the client;
# the backend owns open-model routing via project reasoner defaults.
_LOCAL_PROVIDER_ALIASES = frozenset({"neosyntropy/base", "neosyntropy-base"})

# Opaque /control/runs ControlGraph forbids console-only node/graph fields.
_CONTROL_API_NODE_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "prerequisites",
        "is_fallback",
        "group",
        "tuned",
        "input_schema",
        "output_schema",
        "axioms",
    }
)
_CONTROL_API_GRAPH_FIELDS = frozenset(
    {
        "schema_version",
        "entry",
        "input_schema",
        "nodes",
        "edges",
        "groups",
        "routers",
        "allow_unlisted_transitions",
    }
)


def _control_api_graph(graph_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project a rich manifest onto the backend ControlGraph wire contract.

    Console manifests embed router stubs in ``nodes`` (``kind: "router"``) for
    display. The control API only accepts executable nodes there; routers belong
    exclusively in the ``routers`` id list.
    """
    nodes: list[dict[str, Any]] = []
    router_ids: list[str] = []
    seen_routers: set[str] = set()

    for item in graph_manifest.get("routers") or []:
        if isinstance(item, str) and item and item not in seen_routers:
            router_ids.append(item)
            seen_routers.add(item)

    for node in graph_manifest.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        if node.get("kind") == "router":
            rid = node.get("id")
            if isinstance(rid, str) and rid and rid not in seen_routers:
                router_ids.append(rid)
                seen_routers.add(rid)
            continue
        nodes.append({key: node[key] for key in _CONTROL_API_NODE_FIELDS if key in node})

    payload = {
        key: graph_manifest[key]
        for key in _CONTROL_API_GRAPH_FIELDS
        if key in graph_manifest and key not in {"nodes", "routers"}
    }
    payload["nodes"] = nodes
    payload["routers"] = router_ids
    payload.setdefault("schema_version", 1)
    payload.setdefault("edges", [])
    payload.setdefault("groups", [])
    payload.setdefault("allow_unlisted_transitions", False)
    return payload


def _wire_node_declaration(node: Any) -> dict[str, Any]:
    # input_schema is enforced client-side / on control-run graph nodes;
    # the /framework/inference NodePromptDeclaration wire does not carry it.
    payload = {
        "id": getattr(node, "id", ""),
        "name": getattr(node, "name", "") or "",
        "description": getattr(node, "description", "") or "",
        "prompt": getattr(node, "prompt", "") or "",
        "mode": getattr(node, "mode", None),
        "tools": list(getattr(node, "tools", ()) or ()),
        "output_schema": getattr(node, "output_schema", None),
    }
    # Only forward non-local provider ids (e.g. Vertex model names).
    # neosyntropy/base must not be sent — older backends forbid the field,
    # and newer ones treat anything other than neosyntropy/base as a Vertex model.
    provider = getattr(node, "provider", "neosyntropy/base") or "neosyntropy/base"
    if provider not in _LOCAL_PROVIDER_ALIASES:
        payload["provider"] = provider
    return payload


def _wire_tool_catalog(tools: Any) -> list[dict[str, Any]]:
    specs = tools.specs() if hasattr(tools, "specs") else ()
    return [
        {"name": spec.name, "description": spec.description or ""}
        for spec in specs
    ]
