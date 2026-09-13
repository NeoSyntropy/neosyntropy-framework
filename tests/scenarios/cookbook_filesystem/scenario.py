"""Scenario: Cookbook: filesystem tools.

Copy of ``cookbook/tools/filesystem_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    write_file --> ls --> read_file
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_filesystem"
SCENARIO_TITLE = "Cookbook: filesystem tools"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
