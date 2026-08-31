"""Live test that runs the FSM reasoning-node cookbook.

Run::

    pytest tests/test_Cookbook_reasoning_node_prompt_tools_example.py -s
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

COOKBOOK = (
    Path(__file__).resolve().parents[1]
    / "cookbook"
    / "fsm"
    / "reasoning_node_prompt_tools_example.py"
)


def _load_cookbook():
    spec = importlib.util.spec_from_file_location(
        "Cookbook_reasoning_node_prompt_tools_example",
        COOKBOOK,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_Cookbook_reasoning_node_prompt_tools_example():
    cookbook = _load_cookbook()
    result = cookbook.main()

    assert result is not None
    assert not result.rejected, result.rejection
    assert result.final_state == "End"

    route_output = None
    for step in result.steps:
        for item in step.results:
            if item.node_id == "RouteIntent":
                route_output = item.output
    assert route_output is not None
    assert route_output.get("lane") == "shipping"
    assert "confidence" in route_output
    assert route_output.get("summary")


if __name__ == "__main__":
    test_Cookbook_reasoning_node_prompt_tools_example()
