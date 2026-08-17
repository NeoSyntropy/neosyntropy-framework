import asyncio
from typing import Any

from pydantic import BaseModel

from neosyntropy import FSM, OpenInput, SchemaNode
from neosyntropy.benchmark.synthesizer import FSMSynthesizer
from neosyntropy.routing.declarations import SemanticRouter


class MockProvider:
    def __init__(self, fixed_response: str | dict):
        self.fixed_response = fixed_response
        self.prompts_received = []

    def generate(
        self, prompt: str, *, schema: dict[str, Any] | None = None, tools: Any = None
    ) -> str | dict:
        self.prompts_received.append(prompt)
        return self.fixed_response


class FakeInput(BaseModel):
    user_request: str


def test_synthesize_entry_cases():
    # Setup mock FSM
    router = SemanticRouter(
        id="EntryRouter",
        input_schema=FakeInput,
        routes={
            "analysis_route": "AnalystPhase",
            "help_route": "HelpNode",
        },
        fallback_node="FallbackNode",
        provider="dummy",
    )
    
    n1 = SchemaNode(id="AnalystPhase", input_schema=OpenInput, output_schema=OpenInput, prompt="x", provider="dummy")
    n2 = SchemaNode(id="HelpNode", input_schema=OpenInput, output_schema=OpenInput, prompt="x", provider="dummy")
    n3 = SchemaNode(id="FallbackNode", is_fallback=True, input_schema=OpenInput, output_schema=OpenInput, prompt="x", provider="dummy")
    
    fsm = FSM(entry=router, nodes=[n1, n2, n3])
    
    # Setup mock provider
    mock_payload = {"user_request": "I need a simulated payload."}
    provider = MockProvider(fixed_response=mock_payload)
    
    # Synthesize cases
    synthesizer = FSMSynthesizer(fsm=fsm, provider=provider)
    dataset = asyncio.run(synthesizer.synthesize_entry_cases(samples_per_edge=2))
    
    assert len(dataset.router_cases) == 4  # 2 routes * 2 samples
    assert len(dataset.node_cases) == 0
    assert len(dataset.fsm_cases) == 0
    
    routes_found = [case.expected_route for case in dataset.router_cases]
    assert routes_found.count("analysis_route") == 2
    assert routes_found.count("help_route") == 2
    
    for case in dataset.router_cases:
        assert case.target_router_id == "EntryRouter"
        assert case.input_state == mock_payload
        assert case.critic_json == {"labeled": True, "synthesized": True}
        
    # Check that prompts instructed the LLM properly
    assert len(provider.prompts_received) == 4
    for prompt in provider.prompts_received:
        assert "user request aiming to achieve this goal" in prompt
