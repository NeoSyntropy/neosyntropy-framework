from __future__ import annotations

import asyncio
from typing import Any

from ..backend import BackendClient, BackendProvider
from ..core.graph import FSM
from ..core.models import RunRequest
from ..control.manager import ControlManager
from .dataset import BenchmarkDataset
from .critic import Critic, BackendCritic
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

    async def run_all(self, project_id: str) -> dict[str, Any]:
        """Run all benchmark tests across all specified providers."""
        results = {}
        for provider in self.providers:
            results[provider] = await self._run_for_provider(provider)
            
            # Automatically push critic results if they exist
            node_results = results[provider].get("nodes", {})
            for target_node, tracker in node_results.items():
                if tracker.results:
                    try:
                        await self.client.push_critic_results(
                            project_id=project_id,
                            node_id=target_node,
                            results={"items": tracker.results}
                        )
                    except Exception as exc:
                        print(f"Failed to push critic results for node {target_node}: {exc}")

        return results

    async def _run_for_provider(self, provider: str) -> dict[str, Any]:
        """Run the full dataset using a specific provider."""
        # Note: In a real implementation, we'd map `provider` to the framework's provider registry
        # and override the backend provider configuration.
        
        provider_instance = BackendProvider(self.client)
        # Using ControlManager for execution
        manager = ControlManager(
            self.fsm,
            backend=self.client,
            providers={"neosyntropy/base": provider_instance}
        )

        node_trackers: dict[str, NodeAccuracyTracker] = {}
        for case in self.dataset.node_cases:
            if case.target_node_id not in node_trackers:
                node_trackers[case.target_node_id] = NodeAccuracyTracker(case.target_node_id)
            
            # For a node test, we run from the target_node_id.
            req = RunRequest(
                current_state=case.target_node_id,
                state=case.input_state,
            )
            # Execute step
            run_result = await manager.arun(req)
            actual_output = run_result.state
            
            verdict = await self.critic.evaluate_node(
                target_node_id=case.target_node_id,
                input_state=case.input_state,
                actual_output=actual_output,
                expected_output=case.expected_output,
                criteria=case.criteria
            )
            node_trackers[case.target_node_id].add_result(case.id, verdict)

        router_trackers: dict[str, RouterAccuracyTracker] = {}
        for case in self.dataset.router_cases:
            if case.target_router_id not in router_trackers:
                router_trackers[case.target_router_id] = RouterAccuracyTracker(case.target_router_id)
            
            req = RunRequest(
                current_state=case.target_router_id,
                state=case.input_state,
            )
            run_result = await manager.arun(req)
            # The next state in run_result is the route taken
            actual_route = run_result.current_state
            
            verdict = await self.critic.evaluate_router(
                target_router_id=case.target_router_id,
                input_state=case.input_state,
                actual_route=actual_route,
                expected_route=case.expected_route,
                criteria=case.criteria
            )
            router_trackers[case.target_router_id].add_result(case.id, verdict)

        fsm_tracker = FullPathAccuracyTracker()
        for case in self.dataset.fsm_cases:
            req = RunRequest(
                current_state=self.fsm.entry_id,
                state=case.input_state,
            )
            # A full FSM run might require multiple steps; ControlManager.arun() executes one step.
            # We would need to loop until completion (e.g. state == "End").
            # For brevity in the benchmark runner, we execute the loop.
            current_state = self.fsm.entry_id
            state_data = case.input_state
            steps = 0
            while current_state != "End" and steps < 15:
                res = await manager.arun(RunRequest(current_state=current_state, state=state_data))
                current_state = res.current_state
                state_data = res.state
                steps += 1
                if not res.accepted:
                    break

            verdict = await self.critic.evaluate_node(
                target_node_id="FSM",
                input_state=case.input_state,
                actual_output=state_data,
                expected_output=case.expected_output,
                criteria=case.criteria
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
