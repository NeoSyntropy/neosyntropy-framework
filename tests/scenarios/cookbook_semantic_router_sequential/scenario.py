"""Scenario: Cookbook: semantic router (sequential).

Copy of ``cookbook/fsm/semantic_router_sequential_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    CaptureRequest --> SupportIntent --> InvestigateBilling --> ResolveRequest --> End
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_semantic_router_sequential"
SCENARIO_TITLE = "Cookbook: semantic router (sequential)"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
