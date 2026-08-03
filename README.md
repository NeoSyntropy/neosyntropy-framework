# NeoSyntropy Framework

A deterministic control layer for AI workflows. Models propose what should
happen next; a finite-state graph defines what is allowed to happen.

Core primitives — start with the model-backed nodes. That is the point of the
framework: you drop tiny, scoped AI models into ordinary code paths
(`SchemaNode` / `ReasoningNode`), while routers and `ControlManager` keep every
proposal inside a fail-closed graph.

| Primitive | Role |
|---|---|
| [`SchemaNode`](docs/concepts-explained.md#3-schemanode--constrained-json-no-tools) | Provider-backed extraction: a small model returns constrained JSON for `output_schema`, no tools |
| [`ReasoningNode`](docs/concepts-explained.md#4-reasoningnode--tools--plain-text-notes) | Provider-backed reasoning: a small model may call allow-listed tools and write plain-text notes |
| [`CombineNode`](docs/concepts-explained.md#5-combinenode--reasoning-then-schema) | Authoring unit that expands to reasoning → schema FSM states |
| [`Node`](docs/concepts-explained.md#2-node--node--executable-capability) | Executable capability (Python handler or provider-backed), never a workflow position |
| [`DeterministicRouter`](docs/concepts-explained.md#7-deterministicrouter--hard-rules) | First matching `(predicate, target)` rule wins; compiles to deterministic edges |
| [`SemanticRouter`](docs/concepts-explained.md#8-semanticrouter--labeled-intent-routes) | Model picks among labeled targets (`routes={label: node_or_group}`); still validated against the graph |
| [`Edge`](docs/concepts-explained.md#6-edge--one-permitted-movement) | One permitted movement: `deterministic`, `semantic`, or `fallback` |
| [`Group`](docs/concepts-explained.md#9-group--named-subgraph-optional) | Named node collection; optional `entry`, internal routers, and `add_edge` that compile into the FSM |
| [`ControlManager`](docs/concepts-explained.md#11-controlmanager--one-control-cycle) | The whole cycle: deterministic → semantic router → fallback → validate → execute → commit |

Each concept explained (what / when / example):
[`docs/concepts-explained.md`](docs/concepts-explained.md)

Site docs:
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

## CLI login and projects

Installing the package also installs the `neosyntropy` command, while preserving
the normal Python import API. Connect a terminal through your existing browser
session, then create or select a project:

```bash
neosyntropy login
neosyntropy project create "Support automation" --use
neosyntropy project list
```

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

Every cycle returns a `RunResult` with a full `AuditRecord`. A rejection is a
normal outcome — `result.rejected` is set and the audit explains why.

More detail: [`docs/concepts-explained.md`](docs/concepts-explained.md) ·
[`examples/refund_workflow.py`](examples/refund_workflow.py)
