"""Scenario: Cookbook: FileSystemKnowledge retrieval.

Copy of ``cookbook/knowledge/retrieval_example.py`` for the scenario map. The graph is
loaded from the cookbook; deliveries run it offline and persist results.

Live backend APIs: ``BACKEND_APIS`` and ``tests/scenarios/BACKEND.md``.

```mermaid
flowchart LR
    FileSystemKnowledge.search --> hits
```
"""

from tests.scenarios.cookbook_support import SPECS, build_from_spec

SCENARIO_ID = "cookbook_knowledge_retrieval"
SCENARIO_TITLE = "Cookbook: FileSystemKnowledge retrieval"
BACKEND_APIS = SPECS[SCENARIO_ID].backend_apis


def build_scenario():
    return build_from_spec(SCENARIO_ID)
