# Tier 1 neo-code FSM vs LangGraph

Shallow BMAD path (1–2 steps). Mirrors neo-code `PhaseRouter` plus
`bmad-agent-analyst`. Other phases are stub SchemaNodes so the router still
has six legal hops.

```text
PhaseRouter (SemanticRouter, provider=neosyntropy/base)
  analysis        → AnalystPhase (ReasoningNode) → FinalizeNode (SchemaNode) → End
  plan            → PlanStub (SchemaNode) → End
  solutioning     → SolutioningStub (SchemaNode) → End
  implementation  → ImplementationStub (SchemaNode) → End
  core            → CoreStub (SchemaNode) → End
  help            → HelpStub (SchemaNode) → End
  fallback        → OutOfScope (SchemaNode) → End
```

`analysis` / `plan` / … are route **labels**. Compiled landing ids are
`AnalystPhase`, `PlanStub`, …

## What is compared

Same gold file (`datasets/cases.jsonl`) on two engines:

| Metric | How it is scored |
|---|---|
| **Task accuracy** | Exact `expected_route` + `expected_path` subsequence + schema fields + required tools. No LLM judge. |
| **Latency** | Wall clock per case; table reports p50 / p95 / mean. |
| **Money** | NeoSyntropy: committed hops (`$0.005` trained router, `$0.01` glm node). LangGraph: input/output tokens × model list price. |

LangGraph uses a full LLM as the phase classifier (`with_structured_output`).
NeoSyntropy uses the trained `neosyntropy/base` router; illegal labels cannot
commit. LangGraph unknown labels are coerced to `OutOfScope` and counted as
**illegal hops**.

Accuracy is the gate. Latency and USD are the comparison at that gate.

## Run the comparison

```bash
# from neosyntropy-framework
pip install -e ".[bench]"   # langgraph + langchain-openai

python Benchmarks/tier1/compare.py
python Benchmarks/tier1/compare.py --systems neosyntropy --limit 4
python Benchmarks/tier1/compare.py --json results.json
```

Live NeoSyntropy needs `NEOSYNTROPY_API_KEY` and `NEOSYNTROPY_PROJECT_ID`
(see `tests/.env.example`). Live LangGraph needs `OPENAI_API_KEY`, or
`OPENAI_BASE_URL` pointing at an OpenAI-compatible endpoint (set
`LANGGRAPH_MODEL=glm-5-maas` to match the NeoSyntropy node model).

Without credentials the CLI skips that system and exits 2 if both are skipped.
Unit tests never call a backend.

## Single-engine FSM run

```python
from neosyntropy import Client
from fsm import NeoCodeActivation, fsm, registry

client = Client(api_key="...", project_id="...")
result = fsm.run(
    NeoCodeActivation(user_request="Analyze constraints for a CI secret scanner."),
    client=client,
    tools=registry,
)
```

Compile check (no backend):

```bash
python -c "import sys; sys.path.insert(0, 'Benchmarks/tier1'); from fsm import build_fsm; g=build_fsm(); print(g.entry_id, sorted(g.router_ids))"
```
