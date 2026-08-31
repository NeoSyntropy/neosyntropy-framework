# Site docs (canonical)

`framework-docs.json` is the source of truth for NeoSyntropy product docs that
appear on the website under Get started, Core concepts, and API reference.

## Sync to the frontend

On every push to `main` that changes this directory, GitHub Actions copies the
JSON into `neo_syntropy_frontend` at `app/content/framework-docs.json` and
opens a pull request.

### One-time setup (framework repo)

1. Create a GitHub PAT (or fine-grained token) with **contents: write** and
   **pull requests: write** on the frontend repository.
2. Add it as repository secret `DOCS_SYNC_TOKEN` on
   `NeoSyntropy/neosyntropy-framework`.
3. Optionally set repository variable `FRONTEND_REPO` (default
   `aharrar/neo_syntropy_frontend`).

Until the secret is set, the workflow no-ops.

### Local preview

```bash
cp docs/site/framework-docs.json \
  ../neo_syntropy_frontend/app/content/framework-docs.json
```

The frontend imports that file from `app/content/docs.ts`.
