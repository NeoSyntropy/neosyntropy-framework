from __future__ import annotations

import asyncio
import json
from typing import Any

from ..core.graph import FSM
from ..providers.base import Provider
from .dataset import BenchmarkDataset, RouterTestCase


class FSMSynthesizer:
    """Generates synthetic test cases covering an FSM's entry routes."""

    def __init__(self, fsm: FSM, provider: Provider):
        self.fsm = fsm
        self.provider = provider

    async def synthesize_entry_cases(
        self, samples_per_edge: int = 3
    ) -> BenchmarkDataset:
        """Synthesize entry-point inputs designed to achieve graph coverage."""
        router_cases = []
        entry_schema = self.fsm.input_schema
        if not entry_schema:
            raise ValueError("FSM missing input schema")

        entry_router = self.fsm.routers.get(self.fsm.entry_id)
        if not entry_router or not hasattr(entry_router, "routes"):
            # Currently focused on synthesizing paths from an entry router.
            return BenchmarkDataset()

        # Iterate over each semantic route
        for route_name, target in entry_router.routes.items():
            # In some routers, target might be a Node instance or string ID
            target_id = getattr(target, "id", str(target))
            target_desc = route_name
            target_node = self.fsm.nodes.get(target_id)
            if target_node and target_node.description:
                target_desc = target_node.description

            for _ in range(samples_per_edge):
                prompt = (
                    f"Generate a realistic JSON payload matching the workflow's input schema. "
                    f"The payload MUST represent a user request aiming to achieve this goal or follow this route: '{target_desc}'. "
                    f"Route label: '{route_name}'. "
                    f"Return ONLY valid JSON matching the schema."
                )

                res = self.provider.generate(prompt=prompt, schema=entry_schema)
                if asyncio.iscoroutine(res):
                    res = await res

                payload = res
                if isinstance(res, str):
                    try:
                        payload = json.loads(res)
                    except json.JSONDecodeError:
                        payload = {"text": res}

                prefill_prompt = f"Provide a brief preliminary analysis (prefill) for this user request: {json.dumps(payload)}"
                prefill_res = self.provider.generate(prompt=prefill_prompt, schema=None)
                if asyncio.iscoroutine(prefill_res):
                    prefill_res = await prefill_res
                prefill_text = prefill_res if isinstance(prefill_res, str) else json.dumps(prefill_res)

                router_cases.append(
                    RouterTestCase(
                        input_state=payload,
                        target_router_id=self.fsm.entry_id,
                        prefill=prefill_text,
                        expected_route=route_name,
                        critic_json={"labeled": True, "synthesized": True},
                    )
                )

        return BenchmarkDataset(router_cases=router_cases)
