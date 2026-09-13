# Remote execution and graph governance

Credentials alone do not hand the control loop to the backend. Two
environment flags, both **literal uppercase `TRUE` only**, opt into
publication, telemetry, and backend-owned control.

| Flag | Enables | Also implies |
|---|---|---|
| `NEOSYNTROPY_MONITOR=TRUE` | Graph **structure** registration and run observers | — |
| `NEO_REMOTE_EXECUTION=TRUE` | Executable publication, backend-owned control, `FSM.load` | Monitoring |

`true`, `True`, `1`, `yes`, and `on` stay **off**. See
[`neosyntropy/_features.py`](../neosyntropy/_features.py) and
[`tests/test_feature_flags.py`](../tests/test_feature_flags.py).

Root [`.env.example`](../.env.example) documents the two flags.
Cookbook scripts load [`tests/.env`](../tests/.env.example) for API
credentials.

---

## Three run modes

```text
fsm.run(..., client=client)
        │
        ▼
  ControlManager
        │
        ├─ no flags
        │     local PreferredPathRouter
        │     SchemaNode / ReasoningNode may still call backend inference
        │
        ├─ NEOSYNTROPY_MONITOR=TRUE
        │     same local control
        │     register source-free graph structure
        │     honor RunObserver / telemetry (best-effort)
        │
        └─ NEO_REMOTE_EXECUTION=TRUE
              BackendSemanticRouter + POST /api/v1/control/runs
              extract + upload code bundles
              publish a recoverable revision
              cache a local snapshot under .neosyntropy/
```

A configured `Client` without either flag never starts a backend control
run. Tests assert this: `ControlManager(graph, backend=backend)` keeps
`_backend is None` until `NEO_REMOTE_EXECUTION=TRUE`.

---

## Publish then load

### 1. Publish on first run

With the remote flag and a bound project, the first `fsm.run(..., client=client)`
(or `ControlManager(graph, client=client).run(...)`):

1. Builds a structure manifest (`graph_manifest`).
2. Extracts handlers, tools, router predicates, and edge guards into
   content-addressed gzip JSON bundles
   (`application/vnd.neosyntropy.remote.v1+json+gzip`).
3. Registers the structure, uploads missing bundles, publishes a recovery
   revision.
4. Downloads the server snapshot and writes
   `.neosyntropy/<project_id>/graphs/<graph_id>/`.

If any **required** artifact is not publishable, registration raises
(`graph cannot be registered for remote recovery: …`) instead of uploading
a partial graph.

```python
import os
from neosyntropy import Client, EmptyOutput, OpenInput, TextOutput, Workflow, node

os.environ["NEO_REMOTE_EXECUTION"] = "TRUE"

client = Client(api_key="nsk_...", base_url="https://api.neosyntropy.com")
client.create_project(name="Support Bot", slug="support-bot")

@node(id="Verify", input_schema=OpenInput, output_schema=EmptyOutput)
def verify(ctx):
    return ctx.result(output={}, state_updates={"verified": True})

@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope."})

fsm = Workflow([verify], fallback=out_of_scope)
result = fsm.run({"text": "hello"}, client=client)
print(fsm.graph_id)  # set after a successful publish
```

Re-runs skip republish when `graph._remote_snapshot_revision` already
matches the packaged revision.

### 2. Load a stored graph

```python
import os
from neosyntropy import Client, FSM

os.environ["NEO_REMOTE_EXECUTION"] = "TRUE"
client = Client(api_key="nsk_...", base_url="https://api.neosyntropy.com")
client.create_project(name="Support Bot", slug="support-bot")

loaded = FSM.load(graph_id, client=client)
result = loaded.run({"text": "hello"}, client=client)
```

`FSM.load`:

- Raises **before any network call** when the remote flag is unset.
- Prefers `recovery_manifest` over the structure-only `manifest`.
- Checks `structure_hash` and `revision` (`recovery_revision`).
- Validates `runtime_compat` (CPython `>=3.10,<4` and a compatible
  `neosyntropy` minor) and each artifact's `dependency_lock`.
- Hydrates handlers from the local snapshot when it matches; otherwise
  downloads bundles and writes `.neosyntropy/`.

`FSM.from_manifest(manifest, code_callables=...)` reconstructs a compiled
graph in-process. Use it in tests; use `FSM.load` for a stored UUID.

---

## What the backend owns vs what stays local

When remote execution is on:

| Backend | Client SDK |
|---|---|
| Candidate selection, semantic routing, plan validation, commits | Local `@node` handlers and tool bodies |
| Opaque `awaiting_execution` steps (`node` ids only) | Extract, upload, and hydrate code bundles |
| Rejection reasons | Unique deterministic hops and edge **guards** (evaluated locally; first matching deterministic hop short-circuits to the offline cycle because the backend cannot see guards) |

Responses never include topology, candidates, scores, providers, or model
names. That contract is the control API (`POST /api/v1/control/runs`,
`POST /api/v1/control/runs/{run_id}/results`).

Monitor-only mode registers **structure** (no bundles) and is best-effort:
a slow or down observer never changes execution, validation, or commits.

---

## Snapshots

Layout (cwd-relative unless `project_root=` is passed):

```text
.neosyntropy/
  .gitignore          # * plus !.gitignore — do not commit snapshots
  <project_id>/graphs/<graph_id>/
    structure.json
    recovery.json
    artifacts.json
    artifacts/<sha256>.json.gz
```

`scripts/run_remote_cookbooks.py` sets `NEO_REMOTE_EXECUTION=TRUE` and
prints new vs existing snapshot directories after the FSM / KPI /
validation / decorator cookbooks.

---

## Extractability constraints

The extractor walks project source (stops at `pyproject.toml` / `.git`;
skips `.venv` and `site-packages`). A graph is **not recoverable** when a
required artifact is blocked:

| Reason | Typical cause | Fix |
|---|---|---|
| `unsupported_closure_binding` | Closure captures a callable, socket, DB client, or other non-JSON value | Close over JSON-compatible constants (`str`, `int`, `float`, `bool`, `None`, `bytes`, lists/tuples/sets of those, string-key dicts) |
| `vfs_file_limit_exceeded` | Transitive local files exceed 200 | Shrink the handler's local import closure |
| Missing source | Lambda / nested helper the extractor cannot locate | Use a named top-level function |

`CombineNode` cannot be the dedicated fallback. `Workflow` requires
exactly one fallback node.

---

## Client surface used by this path

```python
client = Client(
    api_key="nsk_...",                    # or access_token=...
    project_id=None,                      # or pass an existing id
    base_url="https://api.neosyntropy.com",  # must include http(s)://
    timeout=180.0,
    telemetry_timeout=15.0,               # default 15s; keep >0
)
client.create_project(name="…", slug="…")  # get-or-create by slug; binds project_id
client.list_projects()
client.get_graph(graph_id)
client.get_graph_snapshot(graph_id)
client.get_graph_code_bundles(graph_id)
```

`BackendClient.from_env()` reads `NEOSYNTROPY_API_URL`,
`NEOSYNTROPY_API_KEY` or `NEOSYNTROPY_ACCESS_TOKEN`, and optional
`NEOSYNTROPY_PROJECT_ID`. A bare host such as `api.neosyntropy.com`
raises: `base_url must be an HTTP or HTTPS URL`. The SDK appends `/api/v1`
when the URL does not already end with it.

---

## Pitfalls

| Symptom | Cause | What to do |
|---|---|---|
| `remote graph loading is disabled` | `FSM.load` without the remote flag | Set `NEO_REMOTE_EXECUTION=TRUE` (uppercase) |
| Control stays local despite a `Client` | Flag unset; credentials only enable inference | Set the remote flag for backend-owned control |
| Observer / `ControlManager(..., observer=)` is ignored | Monitor flag off (and remote flag off) | Set `NEOSYNTROPY_MONITOR=TRUE` or the remote flag |
| `graph cannot be registered for remote recovery` | Required handler/guard not extractable | See extractability table |
| `stored graph runtime is incompatible` | Python or `neosyntropy` minor mismatch | Hydrate on CPython 3.10+ and a matching framework minor |
| `ControlManager.run() cannot be called from a running event loop` | Sync `run()` inside async code | `await ControlManager.arun(...)` |
| `Inference is still warming up` (`inference_warming`) | GPU cold start (often 1–2 minutes) | Retry; cookbooks can set `NEOSYNTROPY_PROVIDER` (for example `gemini-2.5-flash`) when local GPU inference is down |
| `project_id is required` | Client has no bound project | `create_project(name, slug)` or pass `project_id=` |
| `FSM did not reach End within N cycles` | Loop or missing `End` edge | Inspect `result.audit`; raise `max_cycles` (default 32) only after the graph is correct |
| Telemetry timeout orphans a run | `telemetry_timeout` too small vs a slow DB | Client default is 15s; do not drop it to ~2s |

`fsm.run` defaults: `until_end=True` (loop until `End` or rejection),
`max_cycles=32`. Pass `until_end=False` for a single control cycle.

---

## Related

- [`concepts-explained.md`](concepts-explained.md) — primitives and `fsm.run()` arguments
- [`concepts.md`](concepts.md) — fail-closed methodology
- Site: [control manager](https://docs.neosyntropy.com/concepts/control-manager) ·
  [configuration](https://docs.neosyntropy.com/configuration)
- [`scripts/run_remote_cookbooks.py`](../scripts/run_remote_cookbooks.py)
