from __future__ import annotations

from typing import Any

from ..backend import BackendClient, BackendProvider
from ..core.graph import FSM
from ..core.models import RunRequest
from ..control.manager import ControlManager
from .dataset import BenchmarkDataset
from .critic import Critic
from .metrics import NodeAccuracyTracker, RouterAccuracyTracker, FullPathAccuracyTracker


class BenchmarkRunner:
    """Orchestrates FSM benchmark runs."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        fsm: FSM,
        critic: Critic,
        providers: list[str],
        client: BackendClient,
    ):
        self.dataset = dataset
        self.fsm = fsm
        self.critic = critic
        self.providers = providers
        self.client = client

    async def ensure_labels(self, project_id: str, *, model: str = "gemini-2.5-flash") -> None:
        """Run critic on unlabeled DB samples, then keep only gold labels."""
        unlabeled = self.dataset.unlabeled_ids_by_node()
        for node_id, sample_ids in unlabeled.items():
            if not sample_ids:
                continue
            await self.client.critic_eval_samples(
                project_id,
                node_id,
                sample_ids,
                model=model,
                delete_bad=True,
            )
            refreshed = await BenchmarkDataset.from_backend(
                self.client, project_id, node_id
            )
            self.dataset.node_cases = [
                case
                for case in self.dataset.node_cases
                if case.target_node_id != node_id
            ] + refreshed.node_cases
            self.dataset.router_cases = [
                case
                for case in self.dataset.router_cases
                if case.target_router_id != node_id
            ] + refreshed.router_cases
        self.dataset = self.dataset.labeled_only()

    async def run_all(self, project_id: str) -> dict[str, Any]:
        """Ensure labels, then run all benchmark tests across providers."""
        critic_model = getattr(self.critic, "model", "gemini-2.5-flash")
        await self.ensure_labels(project_id, model=str(critic_model or "gemini-2.5-flash"))

        results: dict[str, Any] = {}
        for provider in self.providers:
            results[provider] = await self._run_for_provider(provider)
        return results

    async def _run_for_provider(self, provider: str) -> dict[str, Any]:
        """Run the full dataset using a specific provider."""
        inference_model = None if provider in {"neosyntropy/base", "neosyntropy-base"} else provider
        provider_instance = BackendProvider(
            self.client, inference_model=inference_model
        )
        manager = ControlManager(
            self.fsm,
            backend=self.client,
            providers={
                "neosyntropy/base": provider_instance,
                provider: provider_instance,
            },
        )

        node_trackers: dict[str, NodeAccuracyTracker] = {}
        for case in self.dataset.node_cases:
            if case.target_node_id not in node_trackers:
                node_trackers[case.target_node_id] = NodeAccuracyTracker(case.target_node_id)

            req = RunRequest(
                current_state=case.target_node_id,
                state=case.input_state,
            )
            run_result = await manager.arun(req)
            actual_output = run_result.state
            if not isinstance(actual_output, dict):
                actual_output = {"value": actual_output}

            verdict = await self.critic.evaluate_node(
                target_node_id=case.target_node_id,
                input_state=case.input_state,
                actual_output=actual_output,
                expected_output=case.expected_output,
                criteria=case.criteria,
            )
            node_trackers[case.target_node_id].add_result(case.id, verdict)

        router_trackers: dict[str, RouterAccuracyTracker] = {}
        for case in self.dataset.router_cases:
            if case.target_router_id not in router_trackers:
                router_trackers[case.target_router_id] = RouterAccuracyTracker(
                    case.target_router_id
                )

            req = RunRequest(
                current_state=case.target_router_id,
                state=case.input_state,
            )
            run_result = await manager.arun(req)
            actual_route = run_result.current_state

            verdict = await self.critic.evaluate_router(
                target_router_id=case.target_router_id,
                input_state=case.input_state,
                actual_route=actual_route,
                expected_route=case.expected_route,
                criteria=case.criteria,
            )
            router_trackers[case.target_router_id].add_result(case.id, verdict)

        fsm_tracker = FullPathAccuracyTracker()
        for case in self.dataset.fsm_cases:
            current_state = self.fsm.entry_id
            state_data = case.input_state
            steps = 0
            while current_state != "End" and steps < 15:
                res = await manager.arun(
                    RunRequest(current_state=current_state, state=state_data)
                )
                current_state = res.current_state
                state_data = res.state
                steps += 1
                if not res.accepted:
                    break

            actual = state_data if isinstance(state_data, dict) else {"value": state_data}
            verdict = await self.critic.evaluate_node(
                target_node_id="FSM",
                input_state=case.input_state,
                actual_output=actual,
                expected_output=case.expected_output,
                criteria=case.criteria,
            )
            fsm_tracker.add_result(case.id, verdict)

        return {
            "nodes": node_trackers,
            "routers": router_trackers,
            "fsm": fsm_tracker,
        }

    async def trigger_tune_job(self, project_id: str, node_id: str) -> dict[str, Any]:
        """Trigger a tuning job for a specific node if accuracy isn't satisfactory."""
        return await self.client.start_tune_job(project_id, node_id)
