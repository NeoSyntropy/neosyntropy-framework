"""Append-only JSONL decision logging for training corpora.

Every routing decision is a JSONL line (sorted keys, compact separators,
append + flush under a lock) so interrupted writes never corrupt prior lines
and production runs keep producing router/SLM training data.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..core.context import RunContext
from ..core.models import Candidate, RoutingPlan


@runtime_checkable
class DecisionLogger(Protocol):
    def log_router_decision(
        self,
        context: RunContext,
        candidates: list[Candidate],
        plan: RoutingPlan,
    ) -> None: ...


class JsonlDecisionLogger:
    """Writes one routing decision per line, ``slm_decisions.jsonl`` style."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_router_decision(
        self,
        context: RunContext,
        candidates: list[Candidate],
        plan: RoutingPlan,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "node": context.current_state,
            "user_query": context.intent,
            "candidates": {
                candidate.name: candidate.description for candidate in candidates
            },
            "output": {
                "reasoning": plan.reasoning,
                "topology": plan.topology.value,
                "execution_plan": plan.execution_plan,
            },
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
