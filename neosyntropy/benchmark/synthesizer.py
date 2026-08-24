"""FSM synthesis: generates diverse, schema-aware test cases from node declarations."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..core.graph import FSM
from .dataset import BenchmarkDataset, NodeTestCase, RouterTestCase

# Model aliases that mean "our base inference" — synthesis must NEVER use these.
_NEOSYNTROPY_ALIASES = frozenset({"neosyntropy/base", "neosyntropy-base", ""})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SynthesisNodeStub:
    """Minimal node declaration that routes generation to provider model."""

    def __init__(self, node_id: str, provider: str) -> None:
        self.id = node_id
        self.provider = provider
        self.name = ""
        self.description = ""
        self.prompt = ""
        self.mode = "schema_extraction"
        self.tools = ()
        self.output_schema = None


async def _generate_sample_from_seed(
    generate_fn: Any,  # async callable(prompt, *, schema) -> str | dict
    node_prompt: str,
    schema: dict[str, Any],
    seed: dict[str, Any],
    index: int,
    sem: asyncio.Semaphore,
    retries: int = 5,
) -> dict[str, Any]:
    """Generate one sample using the given seed."""
    seed_json = json.dumps(seed, indent=2)
    gen_prompt = (
        f"Generate a realistic input for this AI workflow node.\n"
        f"Node task: \"{node_prompt}\"\n\n"
        f"Here is a real historical interaction (seed):\n"
        f"```json\n{seed_json}\n```\n\n"
        f"Generate a NEW, highly diverse synthetic variation of this interaction.\n"
        f"Keep the structure identical but completely change the names, contexts, tone, specific values, and scenario details.\n"
        f"Ensure this variation is distinctly different from the seed.\n\n"
        "Return ONLY valid JSON that matches the input schema exactly. "
        "Do not add explanation or markdown."
    )

    async with sem:
        last_exc: BaseException | None = None
        for attempt in range(retries):
            try:
                raw = await generate_fn(gen_prompt, schema=schema)

                if isinstance(raw, dict):
                    return raw
                text = str(raw).strip()
                if text.startswith("```"):
                    text = "\n".join(
                        line for line in text.splitlines()
                        if not line.startswith("```")
                    ).strip()
                return json.loads(text)
            except Exception as exc:
                last_exc = exc
                is_warmup = (
                    "warming" in str(exc).lower()
                    or "503" in str(exc)
                    or getattr(exc, "http_status", None) == 503
                )
                if is_warmup and attempt < retries - 1:
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                raise
        raise last_exc or RuntimeError(f"Sample {index} failed after {retries} attempts")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FSMSynthesizer:
    """Generates diverse synthetic test cases using a Model Garden foundation model.
    
    This is a Standalone Synthesizer that strictly uses real runs (seeds) to 
    generate diverse synthetic data, avoiding the low-diversity problems of 
    zero-shot axis generation.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        fsm: FSM,
        client: Any,          # BackendClient — typed as Any to avoid circular import
        *,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if model.lower() in _NEOSYNTROPY_ALIASES:
            raise ValueError(
                f"FSMSynthesizer model cannot be {model!r}. "
                "Synthesis requires a foundation model such as 'gemini-2.5-flash'. "
                "The synthesized data is what gets used to tune neosyntropy/base."
            )
        self.fsm = fsm
        self.model = model
        self._client = client

    def _synthesis_node_stub(self, node_id: str = "synthesizer") -> dict[str, Any]:
        """Minimal node declaration that routes generation to ``self.model``."""
        stub = _SynthesisNodeStub(node_id, self.model)
        return {
            "id": stub.id,
            "name": stub.name,
            "description": stub.description,
            "prompt": stub.prompt,
            "mode": stub.mode,
            "tools": list(stub.tools),
            "output_schema": stub.output_schema,
            "provider": stub.provider,
        }

    async def _generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> Any:
        """Route a synthesis generation call to Model Garden via the node stub."""
        return await self._client.generate(
            prompt,
            schema=schema,
            purpose="node",
            node=self._synthesis_node_stub(),
        )

    # ------------------------------------------------------------------
    # Standalone Synthesis
    # ------------------------------------------------------------------

    async def generate_diverse(
        self,
        node_id: str,
        seed_runs: list[dict[str, Any]],
        count: int = 5,
        *,
        concurrency: int = 8,
    ) -> BenchmarkDataset:
        """Synthesize ``count`` diverse input samples for a specific node using seed runs.

        The LLM acts as a "Teacher", taking real seeds and mutating them into
        new, highly diverse scenarios.
        """
        if not seed_runs:
            raise ValueError("Seed runs must be provided for diverse generation. Zero-shot generation is not supported.")
            
        node = self.fsm.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in FSM")

        schema = node.input_schema or {}
        node_prompt = getattr(node, "prompt", "") or ""

        # Distribute the seeds across the requested count
        selected_seeds = [seed_runs[i % len(seed_runs)] for i in range(count)]
        
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            _generate_sample_from_seed(self._generate, node_prompt, schema, seed, i, sem)
            for i, seed in enumerate(selected_seeds)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        node_cases: list[NodeTestCase] = []
        errors: list[tuple[int, BaseException]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((i, result))
            else:
                node_cases.append(
                    NodeTestCase(
                        input_state=result,
                        target_node_id=node_id,
                        critic_json={"synthesized": True, "seed_index": i % len(seed_runs)},
                    )
                )

        if errors:
            raise RuntimeError(
                f"{len(errors)}/{count} samples failed. First: {errors[0][1]}"
            )

        return BenchmarkDataset(node_cases=node_cases)

    # Alias so existing code (like reasoning node tests) using the old method name can still work if updated with seeds,
    # though they should be migrated to generate_diverse.
    synthesize_node_cases = generate_diverse
    synthesize_reasoning_cases = generate_diverse

    # ------------------------------------------------------------------
    # Router synthesis (SemanticRouter entry)
    # ------------------------------------------------------------------

    async def synthesize_entry_cases(
        self, seed_runs: list[dict[str, Any]], samples_per_edge: int = 3
    ) -> BenchmarkDataset:
        """Synthesize entry-point inputs designed to achieve route coverage, using seeds."""
        if not seed_runs:
            raise ValueError("Seed runs must be provided for diverse entry generation.")
            
        entry_schema = self.fsm.input_schema
        if not entry_schema:
            raise ValueError("FSM missing input schema")

        entry_router = self.fsm.routers.get(self.fsm.entry_id)
        if not entry_router or not hasattr(entry_router, "routes"):
            return BenchmarkDataset()

        router_prompt = getattr(entry_router, "description", "") or (
            "Route user queries to the appropriate handling node."
        )

        router_cases: list[RouterTestCase] = []

        for route_name, target in entry_router.routes.items():
            target_id = getattr(target, "id", str(target))
            target_node = self.fsm.nodes.get(target_id)
            target_desc = (
                (target_node.description if target_node else None)
                or route_name
            )

            route_prompt = (
                f"{router_prompt} "
                f"This specific route handles: '{target_desc}' (label: '{route_name}')."
            )
            
            selected_seeds = [seed_runs[i % len(seed_runs)] for i in range(samples_per_edge)]
            
            sem = asyncio.Semaphore(8)
            tasks = [
                _generate_sample_from_seed(
                    self._generate, route_prompt, entry_schema, seed, i, sem
                )
                for i, seed in enumerate(selected_seeds)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    continue  # skip failed samples for router (best-effort)
                router_cases.append(
                    RouterTestCase(
                        input_state=result,
                        target_router_id=self.fsm.entry_id,
                        expected_route=route_name,
                        critic_json={"labeled": True, "synthesized": True, "seed_index": i % len(seed_runs)},
                    )
                )

        return BenchmarkDataset(router_cases=router_cases)
