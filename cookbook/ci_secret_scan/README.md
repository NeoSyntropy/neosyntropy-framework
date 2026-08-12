# CI secret scan cookbook (per-file FSM loop)

Simple pattern:

1. **List** all file paths under a directory  
2. **Loop** `fsm.run(...)` once per path  
3. **SchemaNode** returns `{ file_path, api_keys: [...], clean }`

```text
for path in list_source_files(repo):
    fsm.run(FilePathInput(file_path=path))
        InvestigateFile (tools) → ListApiKeys (SchemaNode)
```

## Run

```bash
# bundled sample (expects secrets in leaky_client.py)
python cookbook/ci_secret_scan/run_example.py

# your checkout
python cookbook/ci_secret_scan/run_example.py --repo .
```

Credentials: `tests/.env` (see `tests/.env.example`).

## GitHub Actions

Copy `github-actions/secret-scan.yml` into `.github/workflows/` and set
`NEOSYNTROPY_API_KEY` / `NEOSYNTROPY_PROJECT_ID` repo secrets.

Exit code `1` if any file report has a non-empty `api_keys` list.
