"""Scenario: Cookbook: web search tools.

Copy of ``cookbook/tools/web_search_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    web_search --> extract_text
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_web_search"
SCENARIO_TITLE = "Cookbook: web search tools"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
