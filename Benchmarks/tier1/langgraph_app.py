"""LangGraph twin of ``fsm.py``: LLM phase classifier + analyst tools + stubs.

Same landings as NeoSyntropy (``AnalystPhase``, ``PlanStub``, …). Routing is a
full LLM structured call, not a trained ``SemanticRouter``. Unknown labels are
coerced to ``OutOfScope`` and counted as illegal hops.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from protocol import PHASE_ROUTES, map_phase_label, token_usd

ANALYST_PROMPT = (
    "You are a systems analyst. Analyze the provided project context, "
    "identify constraints, and determine system requirements. "
    "Use tools to read workspace files and append decisions to the memlog."
)
FINALIZE_PROMPT = "Generate a structured systems analysis report."
ROUTE_PROMPT = (
    "Pick the BMAD phase for this user request. "
    "Valid values: analysis, plan, solutioning, implementation, core, help, "
    "out_of_scope. Return only that label."
)


def _usage_tokens(message: Any) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        return int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0), int(
            meta.get("output_tokens") or meta.get("completion_tokens") or 0
        )
    return (
        int(getattr(meta, "input_tokens", 0) or getattr(meta, "prompt_tokens", 0) or 0),
        int(
            getattr(meta, "output_tokens", 0)
            or getattr(meta, "completion_tokens", 0)
            or 0
        ),
    )


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def stub_append_memlog(workspace: str, entry_type: str, text: str) -> dict[str, Any]:
    print(f"[TOOL] Appending {entry_type} to memlog: {text}")
    return {"status": "success", "appended": True, "workspace": workspace}


def stub_read_workspace_file(filepath: str) -> dict[str, Any]:
    print(f"[TOOL] Reading file: {filepath}")
    return {"status": "success", "content": "# Simulated workspace file"}


TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "append_memlog": stub_append_memlog,
    "read_workspace_file": stub_read_workspace_file,
}


def _run_tool(name: str, arguments: dict[str, Any] | None) -> str:
    args = dict(arguments or {})
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return json.dumps({"status": "error", "error": f"unknown tool {name}"})
    try:
        return json.dumps(impl(**args))
    except TypeError:
        allowed = impl.__code__.co_varnames[: impl.__code__.co_argcount]
        return json.dumps(impl(**{key: args[key] for key in args if key in allowed}))


def _invoke_structured(llm: Any, schema: Any, messages: list[Any]) -> tuple[Any, Any]:
    """Return ``(parsed, raw_message)``. Prefer include_raw so tokens survive."""
    try:
        structured = llm.with_structured_output(schema, include_raw=True)
        payload = structured.invoke(messages)
    except TypeError:
        parsed = llm.with_structured_output(schema).invoke(messages)
        return parsed, parsed
    if isinstance(payload, dict):
        return payload.get("parsed"), payload.get("raw") or payload.get("parsed")
    return payload, payload


def _estimate_tokens(parsed: Any, raw: Any) -> tuple[int, int]:
    tin, tout = _usage_tokens(raw)
    if tin or tout:
        return tin, tout
    tin, tout = _usage_tokens(parsed)
    if tin or tout:
        return tin, tout
    text = _text(parsed)
    return 32, max(1, len(text) // 4)


class LangGraphHarness:
    """Runnable graph. ``llm`` is a LangChain chat model (or a test double)."""

    def __init__(self, llm: Any, *, model_name: str = "gpt-4.1-mini") -> None:
        self.llm = llm
        self.model_name = model_name
        self._app = self._compile()

    def _compile(self) -> Any:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class GraphState(TypedDict, total=False):
            user_request: str
            is_headless: bool
            project_workspace: str
            landing: str
            path: list[str]
            output: dict[str, Any]
            tools_ok: list[str]
            notes: str
            tokens_in: int
            tokens_out: int
            llm_calls: int
            illegal_hops: int

        def add_usage(
            state: GraphState,
            parsed: Any,
            raw: Any | None = None,
            calls: int = 1,
        ) -> None:
            tin, tout = _estimate_tokens(parsed, raw if raw is not None else parsed)
            state["tokens_in"] = int(state.get("tokens_in") or 0) + tin
            state["tokens_out"] = int(state.get("tokens_out") or 0) + tout
            state["llm_calls"] = int(state.get("llm_calls") or 0) + calls

        def push(state: GraphState, node_id: str) -> None:
            path = list(state.get("path") or ["PhaseRouter"])
            if not path or path[-1] != node_id:
                path.append(node_id)
            state["path"] = path

        def phase_router(state: GraphState) -> GraphState:
            from typing import Literal

            from langchain_core.messages import HumanMessage, SystemMessage
            from pydantic import BaseModel, ConfigDict

            class RouteDecision(BaseModel):
                model_config = ConfigDict(extra="forbid")
                phase: Literal[
                    "analysis",
                    "plan",
                    "solutioning",
                    "implementation",
                    "core",
                    "help",
                    "out_of_scope",
                ]

            decision, raw_msg = _invoke_structured(
                self.llm,
                RouteDecision,
                [
                    SystemMessage(content=ROUTE_PROMPT),
                    HumanMessage(content=state["user_request"]),
                ],
            )
            add_usage(state, decision, raw_msg)
            raw = getattr(decision, "phase", None) or _text(decision)
            landing, illegal = map_phase_label(str(raw))
            state["landing"] = landing
            state["illegal_hops"] = int(state.get("illegal_hops") or 0) + illegal
            state["path"] = ["PhaseRouter", landing]
            return state

        def route_after(state: dict[str, Any]) -> str:
            return str(state.get("landing") or "OutOfScope")

        def analyst(state: GraphState) -> GraphState:
            from langchain_core.messages import (
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
            from langchain_core.tools import tool

            @tool
            def append_memlog(workspace: str, entry_type: str, text: str) -> str:
                """Append a decision or event to the .memlog.md audit trail."""
                return json.dumps(stub_append_memlog(workspace, entry_type, text))

            @tool
            def read_workspace_file(filepath: str) -> str:
                """Read a workspace file (e.g. existing brief or notes)."""
                return json.dumps(stub_read_workspace_file(filepath))

            bound = self.llm.bind_tools([append_memlog, read_workspace_file])
            messages: list[Any] = [
                SystemMessage(content=ANALYST_PROMPT),
                HumanMessage(content=state["user_request"]),
            ]
            tools_ok: list[str] = list(state.get("tools_ok") or [])
            notes = ""
            for _ in range(6):
                ai = bound.invoke(messages)
                add_usage(state, ai, ai)
                messages.append(ai)
                notes = _text(ai)
                calls = getattr(ai, "tool_calls", None) or []
                if not calls:
                    break
                for call in calls:
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                    args = (
                        call.get("args")
                        if isinstance(call, dict)
                        else getattr(call, "args", {})
                    )
                    call_id = (
                        call.get("id")
                        if isinstance(call, dict)
                        else getattr(call, "id", name)
                    )
                    if name:
                        tools_ok.append(str(name))
                    payload = _run_tool(str(name), dict(args or {}))
                    messages.append(
                        ToolMessage(content=payload, tool_call_id=str(call_id))
                    )
            push(state, "AnalystPhase")
            state["notes"] = notes
            state["tools_ok"] = tools_ok
            return state

        def finalize(state: GraphState) -> GraphState:
            from langchain_core.messages import HumanMessage, SystemMessage
            from pydantic import BaseModel, ConfigDict

            class FinalizeOutput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                status: str
                analysis_report: str

            result, raw_msg = _invoke_structured(
                self.llm,
                FinalizeOutput,
                [
                    SystemMessage(content=FINALIZE_PROMPT),
                    HumanMessage(
                        content=(
                            f"Request: {state['user_request']}\n"
                            f"Notes: {state.get('notes') or ''}"
                        )
                    ),
                ],
            )
            add_usage(state, result, raw_msg)
            push(state, "FinalizeNode")
            path = list(state.get("path") or [])
            path.append("End")
            state["path"] = path
            if hasattr(result, "model_dump"):
                state["output"] = result.model_dump()
            else:
                state["output"] = {
                    "status": str(getattr(result, "status", "") or ""),
                    "analysis_report": str(getattr(result, "analysis_report", "") or _text(result)),
                }
            return state

        def phase_stub(phase: str, node_id: str):
            def _node(state: GraphState) -> GraphState:
                from langchain_core.messages import HumanMessage, SystemMessage
                from pydantic import BaseModel, ConfigDict

                class PhaseStubOutput(BaseModel):
                    model_config = ConfigDict(extra="forbid")
                    phase: str

                result, raw_msg = _invoke_structured(
                    self.llm,
                    PhaseStubOutput,
                    [
                        SystemMessage(
                            content=f"Return JSON with phase set to {phase!r}."
                        ),
                        HumanMessage(content=state["user_request"]),
                    ],
                )
                add_usage(state, result, raw_msg)
                push(state, node_id)
                path = list(state.get("path") or [])
                path.append("End")
                state["path"] = path
                phase_value = getattr(result, "phase", None) or phase
                state["output"] = {"phase": str(phase_value)}
                return state

            _node.__name__ = node_id
            return _node

        def out_of_scope(state: GraphState) -> GraphState:
            from langchain_core.messages import HumanMessage, SystemMessage

            message = self.llm.invoke(
                [
                    SystemMessage(
                        content="Politely refuse out-of-scope requests and suggest a BMAD skill."
                    ),
                    HumanMessage(content=state["user_request"]),
                ]
            )
            add_usage(state, message, message)
            push(state, "OutOfScope")
            path = list(state.get("path") or [])
            path.append("End")
            state["path"] = path
            state["output"] = {"text": _text(message)}
            return state

        graph = StateGraph(GraphState)
        graph.add_node("PhaseRouter", phase_router)
        graph.add_node("AnalystPhase", analyst)
        graph.add_node("FinalizeNode", finalize)
        for label, node_id in PHASE_ROUTES.items():
            if node_id == "AnalystPhase":
                continue
            graph.add_node(node_id, phase_stub(label, node_id))
        graph.add_node("OutOfScope", out_of_scope)
        graph.add_edge(START, "PhaseRouter")
        graph.add_conditional_edges(
            "PhaseRouter",
            route_after,
            {
                "AnalystPhase": "AnalystPhase",
                "PlanStub": "PlanStub",
                "SolutioningStub": "SolutioningStub",
                "ImplementationStub": "ImplementationStub",
                "CoreStub": "CoreStub",
                "HelpStub": "HelpStub",
                "OutOfScope": "OutOfScope",
            },
        )
        graph.add_edge("AnalystPhase", "FinalizeNode")
        graph.add_edge("FinalizeNode", END)
        for node_id in PHASE_ROUTES.values():
            if node_id != "AnalystPhase":
                graph.add_edge(node_id, END)
        graph.add_edge("OutOfScope", END)
        return graph.compile()

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = {
            "user_request": payload["user_request"],
            "is_headless": bool(payload.get("is_headless", False)),
            "project_workspace": str(payload.get("project_workspace") or "."),
            "path": ["PhaseRouter"],
            "output": {},
            "tools_ok": [],
            "tokens_in": 0,
            "tokens_out": 0,
            "llm_calls": 0,
            "illegal_hops": 0,
        }
        result = self._app.invoke(state)
        result["usd"] = token_usd(
            int(result.get("tokens_in") or 0),
            int(result.get("tokens_out") or 0),
            model=self.model_name,
        )
        return result
