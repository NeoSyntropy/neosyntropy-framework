# Remote execution and graph recovery

Executable graph publication is **opt-in**. API credentials alone do **not**
hand control of routing or commits to the backend.

Verified against `neosyntropy/_features.py`, `neosyntropy/control/manager.py`,
`neosyntropy/core/graph.py` (`FSM.load`), and `neosyntropy/cloud/remote/`.

## Feature flags

Both gates require the **literal uppercase string** `TRUE`. `true`, `1`, and
`yes` are ignored.

| Variable | Effect |
|---|---|
| `NEO_REMOTE_EXECUTION=TRUE` | Publish extractable handlers, upload code bundles, run the backend-owned control loop, and enable `FSM.load`. Also implies monitoring. |
| `NEOSYNTROPY_MONITOR=TRUE` | Register graph **structure** (governance / telemetry) only. Does not switch routing or commits to the backend. |

```bash
# Enable executable publication + backend-owned control
export NEO_REMOTE_EXECUTION=TRUE

# Governance only (structure registration, no remote control loop)
export NEOSYNTROPY_MONITOR=TRUE
```

`.env.example` documents the same contract.

## Credentials vs backend-owned control

`ControlManager` can discover a `BackendClient` from `NEOSYNTROPY_API_KEY` /
`NEOSYNTROPY_ACCESS_TOKEN` (and optional `NEOSYNTROPY_PROJECT_ID`). That is
enough for **inference** (`BackendProvider`) and optional **monitoring**.

Backend-owned **select → route → validate → commit** runs only when
`NEO_REMOTE_EXECUTION=TRUE` **and** a backend client is resolved. The manager
comment states this explicitly: a configured backend may still provide model
inference locally.

```text
credentials set, flag off
      → PreferredPathRouter locally
      → BackendProvider for SchemaNode / ReasoningNode inference
      → optional structure registration if NEOSYNTROPY_MONITOR=TRUE

NEO_REMOTE_EXECUTION=TRUE + credentials
      → BackendSemanticRouter + POST /api/v1/control/runs
      → client executes local handlers, posts results
      → graph code extracted, bundled, and snapshotted
```

Without credentials, `BackendClient.from_env()` returns `None` and the manager
stays fully local.

## Publish on first remote run

When the flag is on, the first `ControlManager` run that does not already have
a matching local snapshot:

1. Builds a source-free **structure** manifest (`graph_manifest`).
2. Extracts handlers, tools, and edge guards into gzip JSON **bundles**.
3. Registers the structure, uploads missing bundles, publishes a recovery
   revision.
4. Writes a verified snapshot under `.neosyntropy/`.

If any required callable is not extractable, the run **raises**
`RuntimeError("graph cannot be registered for remote recovery: …")` with the
`recovery_issues` list. There is no silent fallback to local control once the
flag is on.

## `FSM.load`

Reload a previously published graph by UUID:

```python
import os
from neosyntropy import Client, FSM

os.environ["NEO_REMOTE_EXECUTION"] = "TRUE"

client = Client(api_key="...", project_id="...")
fsm = FSM.load("graph-uuid", client=client)
```

Constraints (fail closed):

- `NEO_REMOTE_EXECUTION` must be `TRUE`, or `FSM.load` raises `RuntimeError`.
- `client` must be a NeoSyntropy `Client` with `get_graph` (and bundle
  download when artifacts are not cached).
- The stored record must include a recoverable revision (`recovery_manifest`
  or `code_artifacts` on the structure manifest).
- `structure_hash` and `recovery_revision` must match the published revision.

Loaded graphs hydrate node handlers, router predicates, and edge guards from
the verified bundles. Callables that were never extractable cannot be
recovered.

## Local snapshots

Snapshots live under the process working directory:

```text
.neosyntropy/
  .gitignore          # ignores everything except itself
  {project_id}/
    graphs/
      {graph_id}/
        structure.json
        recovery.json
        artifacts.json
        artifacts/{sha256}.json.gz
```

`project_id` and `graph_id` must be 1–255 ASCII letters, digits, dots, hyphens,
or underscores (not `.` / `..`). Bundle keys are lowercase SHA-256 of the
**compressed** bytes. A cache miss or hash mismatch re-fetches from the
backend.

Do not commit `.neosyntropy/` — the directory gitignores its own contents.

## Extractability limits

`extract_callable_artifact` walks the handler's source closure into a virtual
filesystem (max **200** local files). A graph is recoverable only when every
required callable is `extractable` and not `truncated`.

| Failure | Why | Fix |
|---|---|---|
| `unsupported_closure_binding` | Closure captures a non-JSON value (live client, socket, unserializable object) | Bind JSON-compatible constants, or look up clients inside the handler |
| `vfs_file_limit_exceeded` | Transitive local imports exceed 200 files | Shrink the import closure; keep handlers in a small package |
| source not found | Lambda, interactive session, or installed binary with no file | Move the callable to an inspectable `.py` file |

`neosyntropy` itself is treated as a runtime package (not bundled). Third-party
imports are recorded on the artifact as `external_imports` plus a dependency
lock; they are not vendored into the VFS.

## Client URL

`BackendClient` (used by `Client`) requires `base_url` to start with `http://`
or `https://`. A host-only value raises `ValueError`.

```python
from neosyntropy import Client

# Hosted
client = Client(api_key="...", base_url="https://api.neosyntropy.com")

# Self-hosted / VPC
client = Client(api_key="...", base_url="http://127.0.0.1:8000")
```

If the URL does not already end in `/api/v1`, the client appends it.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `remote graph loading is disabled` | `NEO_REMOTE_EXECUTION` is unset or not exactly `TRUE` |
| Control still routes locally despite an API key | Flag is off; credentials only enable inference / optional monitor |
| `graph cannot be registered for remote recovery` | A handler, tool, or guard is not extractable (see table above) |
| `base_url must be an HTTP or HTTPS URL` | Missing `http://` or `https://` scheme |
| `entry cannot be the removed synthetic 'Start' state` | Graphs start at `entry.id`; do not author a `Start` vertex |
| `@node … requires input_schema` / `output_schema` | Both schemas are required; use `OpenInput` / `EmptyOutput` when unconstrained |
| Monitor events never appear | `NEOSYNTROPY_MONITOR` is not `TRUE` and remote execution is off; observers are ignored |

Related: [`concepts-explained.md`](concepts-explained.md#11-controlmanager--one-control-cycle) ·
[`.env.example`](../.env.example)
