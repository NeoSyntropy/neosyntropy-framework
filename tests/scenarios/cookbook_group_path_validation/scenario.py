"""Scenario: Cookbook: group path validation.

Copy of ``cookbook/validation/group_path_validation_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    ExtractTicket --> ClassifyUrgency --> ValidateTriage --> End
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_group_path_validation"
SCENARIO_TITLE = "Cookbook: group path validation"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
