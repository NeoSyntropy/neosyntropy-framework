# NeoSyntropy Framework

A deterministic control layer for AI workflows. Models propose what should
happen next; a finite-state graph defines what is allowed to happen.

Core primitives — start with the model-backed nodes. That is the point of the
framework: you drop tiny, scoped AI models into ordinary code paths
(`SchemaNode` / `ReasoningNode`), while routers and `ControlManager` keep every
proposal inside a fail-closed graph.

| Primitive | Role |
|---|---|
| [`SchemaNode`](docs/concepts.md#schemanode--constrained-json) | Provider-backed extraction: a small model returns constrained JSON for `output_schema`, no tools |
| [`ReasoningNode`](docs/concepts.md#reasoningnode--tools--notes) | Provider-backed reasoning: a small model may call allow-listed tools and write plain-text notes |
| [`CombineNode`](docs/concepts.md#combinenode--reasoning-then-schema) | Authoring unit that expands to reasoning → schema FSM states |
| [`Node`](docs/concepts.md#node--executable-capability) | Executable capability (Python handler or provider-backed), never a workflow position |
| [`Edge`](docs/concepts.md#edge--one-permitted-movement) | One permitted movement: `deterministic`, `semantic`, or `fallback` |
| [`Group`](docs/concepts.md#group--organization-and-optional-authored-subgraph) | Named node collection; optional `entry`, internal routers, and `add_edge` that compile into the FSM |
| [`DeterministicRouter`](docs/concepts.md#deterministicrouter--hard-rules) | First matching `(predicate, target)` rule wins; compiles to deterministic edges |
| [`SemanticRouter`](docs/concepts.md#semanticrouter--labeled-intent-routes) | Model picks among labeled targets (`routes={label: node_or_group}`); still validated against the graph |
| [`ControlManager`](docs/concepts.md#controlmanager--the-pipeline-as-one-object) | The whole cycle: deterministic → semantic router → fallback → validate → execute → commit |

Site docs (synced from [`docs/site/framework-docs.json`](docs/site/framework-docs.json)):
[Nodes](https://docs.neosyntropy.com/concepts/nodes) ·
[Model-backed nodes](https://docs.neosyntropy.com/concepts/model-nodes) ·
[Routers](https://docs.neosyntropy.com/concepts/routers) ·
[Edges](https://docs.neosyntropy.com/concepts/edges) ·
[Groups](https://docs.neosyntropy.com/concepts/groups) ·
[Control manager](https://docs.neosyntropy.com/concepts/control-manager)

## Install

```bash
pip install neosyntropy
```

From a local checkout:

```bash
pip install -e .          # editable install
pip install -e ".[dev]"   # with pytest + ruff + build tools
```

### Publish a release to PyPI

Publishing is automated by `.github/workflows/publish.yml` (tests → build → upload).

One-time PyPI setup (Trusted Publishing, no API token):

1. Create a GitHub Environment named `pypi` on this repo (Settings → Environments).
2. On [PyPI publishing settings](https://pypi.org/manage/account/publishing/), add a **pending publisher**:
   - Project name: `neosyntropy`
   - Owner: `NeoSyntropy`
   - Repository: `neosyntropy-framework`
   - Workflow: `publish.yml`
   - Environment name: `pypi`

Then bump `version` in `pyproject.toml`, commit, tag, and push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pushing a `v*` tag (or publishing a GitHub Release) runs the workflow and uploads to PyPI.

## CLI login and projects

Installing the package also installs the `neosyntropy` command, while preserving
the normal Python import API. Connect a terminal through your existing browser
session, then create or select a project:

```bash
neosyntropy login
neosyntropy project create "Support automation" --use
neosyntropy project list
```

`login` opens the browser for approval and keeps the refresh credential in the
operating system keychain. Its configuration file contains only the API URL and
selected project ID. Use `--api-url` for a self-hosted API and `--profile` for
separate accounts:

```bash
neosyntropy --api-url http://localhost:8000 --profile development login
neosyntropy --profile development project use <project-id>
neosyntropy logout
```

## Quickstart

```python
from neosyntropy import Client, EmptyOutput, TextOutput, Workflow, node

@node(id="VerifyIdentity", output_schema=EmptyOutput)
def verify_identity(ctx):
    """Verify the requester owns the order."""
    return ctx.result(output={}, state_updates={"verified": True})

@node(id="IssueRefund", prerequisites=("VerifyIdentity",), output_schema=EmptyOutput)
def issue_refund(ctx):
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")

@node(id="OutOfScope", is_fallback=True, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope for this workflow."})

fsm = Workflow(
    [verify_identity, issue_refund],
    fallback=out_of_scope,
)

client = Client(api_key="your-api-key", project_id="your-project-id")
result = fsm.run({"intent": "refund my order", "current_state": "Start"}, client=client)

print(result.final_state)                    # one committed transition
print(result.audit.committed_transitions)    # ["Start->VerifyIdentity"]
```


Every cycle returns a `RunResult` with a full `AuditRecord`: the proposed
plan, the candidates, every gate check, and the committed transitions. A
rejection (illegal plan, illegal transition, failed guard) is a
normal outcome — `result.rejected` is set, nothing was committed for the
offending step, and the audit explains why.

### The entry contract

A node declares what it returns; a graph declares what it takes in.
`input_schema` documents and enforces the state a run must supply when it
starts at `Start`:

```python
class RefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    currency: str = "USD"

graph = FSM(nodes=[...], edges=[...], input_schema=RefundRequest)
```

Fields with defaults stay optional and unknown keys are refused, so no caller
can smuggle state into the workflow. The check is the first gate of the cycle:
input the graph never accepted is rejected before selection, routing, or any
node runs, and the audit records it as an `InputSchema` check. Cycles that
resume mid-workflow are not re-checked — that state is what the workflow
itself produced. A pydantic model or a raw JSON Schema object both work, and
the schema travels in `graph_manifest(graph)` so the console can show what the
entry point expects.

## Routing and control

When backend credentials are configured, **the backend owns the control
cycle**: candidate selection, routing, plan validation, and state commits.
The client only defines the graph and executes local handlers. Responses never
include topology, candidates, execution plans, providers, or model names.

- **Backend control** (default with credentials): `POST /control/runs` +
  `POST /control/runs/{id}/results`. The SDK loops: receive opaque execute
  steps → run local handlers → submit results → accept commits/rejections.
- **Authoring routers** (instead of hand-written edges):
  `DeterministicRouter(id, rules=[(predicate, target), ...])` and
  `SemanticRouter(id, routes={label: group_or_node}, fallback_node=...)`.
  Pass them to `FSM(..., routers=[...], entry=auth_router)`.
- **Authoring groups** (subgraph that compiles into the FSM):
  `@billing.node(...)`, `billing.routers = [...]`, `billing.entry = "ValidateCard"`,
  `billing.add_edge("ValidateCard", "BillingLogic")`. Pass `groups=[billing]`
  (or target the group from a `SemanticRouter`); nodes, routers, and edges
  merge into the parent graph. With `entry` set, a semantic edge to the group
  lands on that entry node.
- **`PreferredPathRouter`** (offline runtime): takes exactly one matching
  deterministic edge, else a unique semantic target, else the fallback edge.
- Node generation for handler-less nodes may still use `/framework/inference`;
  selection/routing stay behind the control API.

Set `NEOSYNTROPY_API_URL` with `NEOSYNTROPY_API_KEY` + `NEOSYNTROPY_PROJECT_ID`
(or `NEOSYNTROPY_ACCESS_TOKEN`). `ControlManager(graph)` discovers them
automatically.


## Observability

When a backend client is configured, `ControlManager` reports the control
lifecycle to:

- `POST /api/v1/telemetry/runs`
- `POST /api/v1/telemetry/runs/{id}/events`
- `POST /api/v1/telemetry/runs/{id}/finish`

Telemetry is bounded and best-effort: an unavailable or slow observer never
changes execution, validation, state commits, returned results, or raised
execution errors. Events cover plan proposals, step start/completion,
committed transitions, rejection/failure, and finish.

By default the run's debug payloads are captured so the console can replay
the FSM step by step (and the data can later feed training):

- the run input (intent, initial state, state snapshot, history, metadata),
- each step's input (`current_state` plus the pre-step state snapshot),
- each step's output (node results, state updates, the post-step state, and
  the rejection reason when a gate fails),
- the run output (final state, final state snapshot, committed transitions).

Oversized step payloads are truncated client-side so events are stored rather
than dropped. Pass `ControlManager(graph, capture_payloads=False)` for
sanitized lifecycle-only telemetry: then only a visualization manifest leaves
the process (graph input schema, node IDs, display names, descriptions,
prompts, modes, tool allow-list names, output schemas, groups, fallback
markers, and typed edges) and handlers, guards, providers, metadata,
request intent, history, state, outputs, and errors are excluded.

Use `graph_manifest(graph)` to inspect the manifest payload or provide a
custom `RunObserver` with `ControlManager(graph, observer=...)`. See
[`examples/observability.py`](examples/observability.py).

## Tools

```python
from pydantic import BaseModel
from neosyntropy import tool

class AddToCartArgs(BaseModel):
    product_id: str
    quantity: int

@tool
def add_to_cart(args: AddToCartArgs) -> dict:
    """Add a quantity of a product to the active cart."""
    ...
```

Tools are capabilities on nodes (`@node(..., tools=("add_to_cart",))`), never
graph vertices. Handlers call them through `ctx.tools.invoke(...)`, which
enforces the node's allow-list fail-closed and logs every invocation.

### Model-driven tool calling

Provider-backed nodes use explicit constructors. A reasoning node may call
tools; a schema node returns constrained JSON; a combine node expands into
reasoning then schema FSM states:

```python
ReasoningNode(
    id="Support",
    input_schema=OpenInput,
    prompt="Help the customer with their order.",
    tools=("lookup_order",),
)

SchemaNode(
    id="Ticket",
    input_schema=OpenInput,
    output_schema=SupportTicket,
    prompt="Extract a support ticket as JSON.",
)

CombineNode(
    id="Clearance",
    input_schema=OpenInput,
    tools=("lookup_order",),
    output_schema=SupportTicket,
    prompt="Gather evidence, then extract a ticket.",
)
```

Python handlers stay on `@node`. The model reasons, emits `<TOOL:lookup_order>`
with no arguments, a parameter extractor fills a JSON object constrained by the
tool's pydantic schema, arguments are validated before the tool runs, and the
outcome is reinjected so reasoning continues. Only the node's own tools appear
in its prompt.

The default extractor uses the same provider; plug a trained edge extractor in
with `ControlManager(graph, extractor=my_extractor)` — anything implementing
`async extract(messages, tool) -> ToolCall` works, which is the same contract
the trained extractors already speak.

Guarantees in this loop: a proposed tool that the node does not declare is
denied and never executed (the refusal is fed back so the model can recover),
invalid arguments never reach the tool, duplicate calls are rejected, and the
loop is bounded by `max_tool_calls`. Every attempt lands in
`NodeResult.tool_calls`, for a complete audit trail.

## Example and docs

- [`examples/refund_workflow.py`](examples/refund_workflow.py) — end-to-end
  workflow with guards, tools, a rejection, and a fallback.
- [`docs/concepts.md`](docs/concepts.md) — methodology: model-backed nodes,
  routers, proposal vs permission, fail-closed gates, wire contracts.
- Site concepts:
  [model-backed nodes](https://docs.neosyntropy.com/concepts/model-nodes),
  [routers](https://docs.neosyntropy.com/concepts/routers),
  [nodes](https://docs.neosyntropy.com/concepts/nodes),
  [edges](https://docs.neosyntropy.com/concepts/edges),
  [groups](https://docs.neosyntropy.com/concepts/groups),
  [control manager](https://docs.neosyntropy.com/concepts/control-manager).
- [`docs/site/framework-docs.json`](docs/site/framework-docs.json) — canonical
  website docs (Get started, Core concepts, Control API). CI syncs this file
  into the frontend repo; see [`docs/site/README.md`](docs/site/README.md).

## Tests

```bash
pytest
ruff check .
```
