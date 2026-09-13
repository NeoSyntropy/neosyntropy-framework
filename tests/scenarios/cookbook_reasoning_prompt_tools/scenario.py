"""Scenario: Cookbook: reasoning node with prompt tools.

Copy of ``cookbook/fsm/reasoning_node_prompt_tools_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    RouteIntent --> End
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_reasoning_prompt_tools"
SCENARIO_TITLE = "Cookbook: reasoning node with prompt tools"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
