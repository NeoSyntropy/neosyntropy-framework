"""Scenario: Cookbook: FSM path KPI.

Copy of ``cookbook/kpi/fsm_path_kpi_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    ParseQuery --> GenerateAnswer --> PathScore --> End
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_fsm_path_kpi"
SCENARIO_TITLE = "Cookbook: FSM path KPI"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
