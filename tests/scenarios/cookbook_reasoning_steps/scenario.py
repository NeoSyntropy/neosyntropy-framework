"""Scenario: Cookbook: reasoning node steps.

Copy of ``cookbook/fsm/reasoning_node_steps_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    SupportDecision_step_0 --> SupportDecision_step_1 --> SupportDecision_step_2 --> End
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_reasoning_steps"
SCENARIO_TITLE = "Cookbook: reasoning node steps"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
