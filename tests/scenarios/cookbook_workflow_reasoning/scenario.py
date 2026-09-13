"""Scenario: Cookbook: @workflow reasoning.

Copy of ``cookbook/decorators/workflow_reasoning_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    lookup_sku --> check_stock --> ExtractParams --> place_order
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_workflow_reasoning"
SCENARIO_TITLE = "Cookbook: @workflow reasoning"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
