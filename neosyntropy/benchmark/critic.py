from __future__ import annotations

import abc
from typing import Any
from pydantic import BaseModel
from ..backend import BackendClient, BackendProvider


class CriticVerdict(BaseModel):
    passed: bool
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] | None = None


class Critic(abc.ABC):
    """Base class for FSM benchmark critics."""

    @abc.abstractmethod
    async def evaluate_node(
        self,
        target_node_id: str,
        input_state: dict[str, Any],
        actual_output: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        """Evaluate a node's output."""
        ...

    @abc.abstractmethod
    async def evaluate_router(
        self,
        target_router_id: str,
        input_state: dict[str, Any],
        actual_route: str,
        expected_route: str,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        """Evaluate a router's decision."""
        ...


class ExactMatchCritic(Critic):
    """Simple critic that strictly compares actual outputs with expected outputs."""

    async def evaluate_node(
        self,
        target_node_id: str,
        input_state: dict[str, Any],
        actual_output: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        if expected_output is None:
            return CriticVerdict(passed=True, score=1.0, reason="No expected output to match.")
        
        passed = (actual_output == expected_output)
        return CriticVerdict(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="Output matches expected exactly." if passed else f"Output {actual_output} != {expected_output}"
        )

    async def evaluate_router(
        self,
        target_router_id: str,
        input_state: dict[str, Any],
        actual_route: str,
        expected_route: str,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        passed = (actual_route == expected_route)
        return CriticVerdict(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="Route matches expected exactly." if passed else f"Route {actual_route} != {expected_route}"
        )


class BackendCritic(Critic):
    """Critic that delegates evaluation to the backend's judge endpoint."""

    def __init__(self, client: BackendClient, project_id: str, model: str = "gemini-2.5-flash"):
        self.client = client
        self.project_id = project_id
        self.model = model

    async def evaluate_node(
        self,
        target_node_id: str,
        input_state: dict[str, Any],
        actual_output: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        # Since this evaluates locally, we can either call a general /inference endpoint 
        # or rely on the runner to trigger the backend's judge on recorded samples.
        # For inline evaluation, we'll return a placeholder indicating backend evaluation is pending.
        return CriticVerdict(
            passed=False, 
            score=0.0, 
            reason="BackendCritic defers evaluation to the backend's /judge endpoint. Run evaluate_with_backend_judge."
        )

    async def evaluate_router(
        self,
        target_router_id: str,
        input_state: dict[str, Any],
        actual_route: str,
        expected_route: str,
        criteria: list[str] | None = None,
    ) -> CriticVerdict:
        return CriticVerdict(
            passed=False,
            score=0.0,
            reason="BackendCritic defers evaluation to the backend's /judge endpoint."
        )

