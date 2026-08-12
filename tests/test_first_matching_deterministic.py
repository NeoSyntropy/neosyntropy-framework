"""DeterministicRouter: first matching rule wins (not unique-match-only)."""

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    DeterministicRouter,
    FSM,
    OpenInput,
    SchemaNode,
    TextOutput,
)


class Activation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_request: str = ""


def test_first_matching_deterministic_prefers_earlier_rule():
    skill = SchemaNode(
        id="SkillNode",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="skill",
    )
    fallback = SchemaNode(
        id="FallbackNode",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    entry = DeterministicRouter(
        id="Entry",
        input_schema=Activation,
        rules=[
            (lambda ctx: ctx.state.get("target_skill") == "prd", skill),
            (lambda ctx: True, fallback),
        ],
    )
    fsm = FSM(entry=entry, nodes=[skill, fallback], routers=[entry])

    first = fsm.first_matching_deterministic("Entry", {"target_skill": "prd"})
    assert first is not None
    assert first.target == "SkillNode"

    all_matching = fsm.matching_deterministic("Entry", {"target_skill": "prd"})
    assert [e.target for e in all_matching] == ["SkillNode", "FallbackNode"]
