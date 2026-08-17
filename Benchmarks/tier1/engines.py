"""Live engines for the Tier 1 NeoSyntropy vs LangGraph comparison."""

from __future__ import annotations

import os
import time
from typing import Any

from protocol import (
    DEFAULT_LG_MODEL,
    Case,
    Trace,
    landing_from_path,
    neosyntropy_usd,
    path_from_committed,
    token_usd,
)


def _output_from_ns(result: Any) -> dict[str, Any]:
    for step in reversed(getattr(result, "steps", []) or []):
        for item in reversed(getattr(step, "results", []) or []):
            if getattr(item, "status", "") != "succeeded":
                continue
            output = getattr(item, "output", None)
            if isinstance(output, dict) and output:
                return dict(output)
            if isinstance(output, str) and output.strip():
                return {"text": output}
    state = getattr(result, "state", None)
    return dict(state) if isinstance(state, dict) else {}


def _tools_from_ns(result: Any) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    denied: list[str] = []
    for step in getattr(result, "steps", []) or []:
        for item in getattr(step, "results", []) or []:
            for record in getattr(item, "tool_calls", []) or []:
                name = str(getattr(record, "tool", "") or "")
                if getattr(record, "denied", False):
                    denied.append(name)
                elif getattr(record, "ok", False) and name:
                    ok.append(name)
    return ok, denied


class NeoSyntropyEngine:
    system = "neosyntropy"

    def __init__(self, client: Any) -> None:
        self.client = client

    def run(self, case: Case) -> Trace:
        from fsm import NeoCodeActivation, fsm, registry

        started = time.perf_counter()
        try:
            result = fsm.run(
                NeoCodeActivation(
                    user_request=case.user_request,
                    is_headless=case.is_headless,
                    project_workspace=case.project_workspace,
                ),
                client=self.client,
                tools=registry,
            )
        except Exception as exc:  # noqa: BLE001 — surface engine failures as scored errors
            return Trace(
                case_id=case.id,
                system=self.system,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        committed = list(getattr(getattr(result, "audit", None), "committed_transitions", []) or [])
        path = path_from_committed(committed)
        if getattr(result, "final_state", None) == "End" and (not path or path[-1] != "End"):
            path.append("End")
        tools_ok, tools_denied = _tools_from_ns(result)
        return Trace(
            case_id=case.id,
            system=self.system,
            landing=landing_from_path(path),
            path=path,
            output=_output_from_ns(result),
            tools_ok=tools_ok,
            tools_denied=tools_denied,
            latency_ms=latency_ms,
            transitions=len(committed),
            usd=neosyntropy_usd(committed),
            error=(
                getattr(result, "rejection", None)
                if getattr(result, "rejected", False)
                else None
            ),
        )


class LangGraphEngine:
    system = "langgraph"

    def __init__(self, llm: Any, *, model_name: str = DEFAULT_LG_MODEL) -> None:
        from langgraph_app import LangGraphHarness

        self.model_name = model_name
        self.harness = LangGraphHarness(llm, model_name=model_name)

    def run(self, case: Case) -> Trace:
        started = time.perf_counter()
        try:
            result = self.harness.invoke(
                {
                    "user_request": case.user_request,
                    "is_headless": case.is_headless,
                    "project_workspace": case.project_workspace,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return Trace(
                case_id=case.id,
                system=self.system,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        path = list(result.get("path") or [])
        tokens_in = int(result.get("tokens_in") or 0)
        tokens_out = int(result.get("tokens_out") or 0)
        usd = float(result.get("usd") or 0.0)
        if usd == 0.0 and (tokens_in or tokens_out):
            usd = token_usd(tokens_in, tokens_out, model=self.model_name)
        return Trace(
            case_id=case.id,
            system=self.system,
            landing=str(result.get("landing") or landing_from_path(path)),
            path=path,
            output=dict(result.get("output") or {}),
            tools_ok=list(result.get("tools_ok") or []),
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            llm_calls=int(result.get("llm_calls") or 0),
            usd=usd,
            illegal_hops=int(result.get("illegal_hops") or 0),
        )


def neosyntropy_client_from_env() -> Any | None:
    api_key = os.getenv("NEOSYNTROPY_API_KEY")
    project_id = os.getenv("NEOSYNTROPY_PROJECT_ID")
    if not api_key or not project_id:
        return None
    from neosyntropy import Client

    kwargs: dict[str, Any] = {"api_key": api_key, "project_id": project_id}
    base_url = os.getenv("NEOSYNTROPY_API_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return Client(**kwargs)


def langgraph_llm_from_env(model_name: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=0)
