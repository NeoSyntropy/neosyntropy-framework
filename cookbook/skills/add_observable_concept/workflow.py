"""add_observable_concept — main workflow orchestrator.

Generates all files needed to introduce a new NeoSyntropy observable concept
across all three repositories (framework, backend, frontend) following the
15-step process defined in neo_syntropy_backend/thoughts/add-new-concept.md.

Quick start
-----------
    from pathlib import Path
    from cookbook.skills.add_observable_concept import ConceptDefinition, generate_concept, write_files

    defn = ConceptDefinition(
        concept="agent",
        concept_label="Agent",
        concept_tab_label="Agents",
        concept_description="AI agents registered by workflows in this project.",
        concept_icon_path="M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z",
        concept_natural_key="name",
        next_migration_number="0019",
        prev_migration_number="0018",
        framework_repo_path=Path("/path/to/neosyntropy-framework"),
        backend_repo_path=Path("/path/to/neo_syntropy_backend"),
        frontend_repo_path=Path("/path/to/neo_syntropy_frontend"),
    )

    result = generate_concept(defn)        # pure generation — nothing written to disk

    if result.success:
        write_files(result)                # actually write create/overwrite files
        print_report(result)
    else:
        for err in result.errors:
            print("ERROR:", err)

CLI
---
    python -m cookbook.skills.add_observable_concept.workflow --help
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from cookbook.skills.add_observable_concept.schemas import (
    ConceptDefinition,
    GeneratedFile,
    WorkflowResult,
)
from cookbook.skills.add_observable_concept.validators import validate_concept_definition

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    """Load raw template text from the templates/ directory."""
    path = _TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def render(template: str, defn: ConceptDefinition) -> str:
    """Replace all ``{PLACEHOLDER}`` markers in *template* with their values.

    Substitutions are applied longest-key-first so that e.g.
    ``{CONCEPT_LABEL}`` is replaced before the shorter ``{CONCEPT}``,
    preventing partial substitution artefacts.
    """
    result = template
    for placeholder, value in defn.substitutions.items():
        result = result.replace(placeholder, value)
    return result


# ---------------------------------------------------------------------------
# Step generators — each returns a list[GeneratedFile]
# ---------------------------------------------------------------------------

def _step1_framework_manifest(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 1: neosyntropy/monitor/{concept}/manifest.py  (CREATE)"""
    tmpl = _load_template("framework_manifest.py.tmpl")
    path = (
        defn.framework_repo_path
        / "neosyntropy" / "monitor" / defn.concept / "manifest.py"
    )
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=1,
        description=f"Framework manifest generator for {defn.concept_label}",
    )]


def _step2_framework_schemas(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 2: neosyntropy/monitor/{concept}/schemas.py  (CREATE/OVERWRITE)"""
    tmpl = _load_template("framework_schemas.py.tmpl")
    path = (
        defn.framework_repo_path
        / "neosyntropy" / "monitor" / defn.concept / "schemas.py"
    )
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="overwrite",
        step=2,
        description=f"Pydantic wire schemas for {defn.concept_label} monitor layer",
    )]


def _step3_framework_observer(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 3: neosyntropy/monitor/{concept}/observer.py  (OVERWRITE stub)"""
    tmpl = _load_template("framework_observer.py.tmpl")
    path = (
        defn.framework_repo_path
        / "neosyntropy" / "monitor" / defn.concept / "observer.py"
    )
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="overwrite",
        step=3,
        description=f"{defn.concept_label}Observer using BackendConceptReporter",
    )]


def _step3_framework_init(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 3 companion: neosyntropy/monitor/{concept}/__init__.py  (CREATE if missing)"""
    path = (
        defn.framework_repo_path
        / "neosyntropy" / "monitor" / defn.concept / "__init__.py"
    )
    if path.exists():
        return []
    return [GeneratedFile(
        path=path,
        content="",
        action="create",
        step=3,
        description=f"Package init for neosyntropy.monitor.{defn.concept}",
    )]


def _step4_decorator_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 4: edit snippet for neosyntropy/core/decorators.py.

    Produces an 'edit-snippet' file so the developer knows exactly what to add
    and where.  The workflow does NOT auto-edit decorators.py because it
    requires context-aware insertion into a 300+ line file.
    """
    helper_snippet = textwrap.dedent(f"""\
        def _build_{defn.concept}_manifests(
            obj: Any,
        ) -> list[dict[str, Any]]:
            \"\"\"Return serialisable manifests for one or more {defn.concept_label} instances.\"\"\"
            from neosyntropy.monitor.{defn.concept}.manifest import {defn.concept}_manifest

            if obj is None:
                return []
            items = obj if isinstance(obj, (list, tuple)) else [obj]
            manifests = []
            for item in items:
                try:
                    manifests.append({defn.concept}_manifest(item))
                except Exception:
                    pass
            return manifests
    """)

    registration_snippet = textwrap.dedent(f"""\
            # Register {defn.concept} manifests at decoration time (add after knowledge block):
            for m in _build_{defn.concept}_manifests({defn.concept}):
                _register_concept_fire_and_forget(
                    client, project_id, "{defn.concept}", m
                )
    """)

    signature_snippet = textwrap.dedent(f"""\
        # Add to both @function_calling and @workflow signatures:
        {defn.concept}: Any = None,
    """)

    full_snippet = "\n".join([
        "# ── FILE: neosyntropy/core/decorators.py ──────────────────────────────────",
        "",
        "# 1. Add this helper function after _build_knowledge_manifests():",
        helper_snippet,
        "",
        "# 2. Add this registration block inside both @function_calling and @workflow,",
        "#    right after the existing 'knowledge' registration loop:",
        registration_snippet,
        "",
        "# 3. Add to BOTH decorator parameter lists:",
        signature_snippet,
    ])

    path = (
        defn.framework_repo_path
        / "neosyntropy" / "core" / f"decorators.{defn.concept}.patch.txt"
    )
    return [GeneratedFile(
        path=path,
        content=full_snippet,
        action="edit-snippet",
        step=4,
        description=(
            f"Edit instructions for decorators.py — add {defn.concept}= param "
            f"to @workflow and @function_calling"
        ),
    )]


def _step5_backend_model(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 5: src/models/{concept}.py  (CREATE)"""
    tmpl = _load_template("backend_model.py.tmpl")
    path = defn.backend_repo_path / "src" / "models" / f"{defn.concept}.py"
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=5,
        description=f"SQLAlchemy model for {defn.concept_label}",
    )]


def _step6_backend_models_init_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 6: edit snippet for src/models/__init__.py."""
    snippet = textwrap.dedent(f"""\
        # ── FILE: src/models/__init__.py ────────────────────────────────────────────
        # Add this import line alongside the other model imports:
        from src.models.{defn.concept} import {defn.concept_label}

        # Add to __all__:
        "{defn.concept_label}",
    """)
    path = (
        defn.backend_repo_path / "src" / "models" / f"__init__.{defn.concept}.patch.txt"
    )
    return [GeneratedFile(
        path=path,
        content=snippet,
        action="edit-snippet",
        step=6,
        description=f"Edit instructions for src/models/__init__.py — register {defn.concept_label}",
    )]


def _step7_backend_schemas(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 7: src/schemas/{concept}.py  (CREATE)"""
    tmpl = _load_template("backend_schemas.py.tmpl")
    path = defn.backend_repo_path / "src" / "schemas" / f"{defn.concept}.py"
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=7,
        description=f"Wire schemas for the {defn.concept_label} API",
    )]


def _step8_backend_operations(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 8: src/operations/{concept}.py  (CREATE)"""
    tmpl = _load_template("backend_operations.py.tmpl")
    path = defn.backend_repo_path / "src" / "operations" / f"{defn.concept}.py"
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=8,
        description=f"DB operations for {defn.concept_label} (upsert / list / get)",
    )]


def _step9_backend_api(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 9: src/apis/{concept}.py  (CREATE)"""
    tmpl = _load_template("backend_api.py.tmpl")
    path = defn.backend_repo_path / "src" / "apis" / f"{defn.concept}.py"
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=9,
        description=f"FastAPI router for {defn.concept_label} (3 routes)",
    )]


def _step10_backend_main_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 10: edit snippet for src/main.py."""
    snippet = textwrap.dedent(f"""\
        # ── FILE: src/main.py ───────────────────────────────────────────────────────
        # 1. Add this import after the graphs_router import line:
        from src.apis.{defn.concept} import router as {defn.concept}_router

        # 2. Add this include after the graphs_router include line:
        app.include_router({defn.concept}_router, prefix="/api/v1")
    """)
    path = defn.backend_repo_path / "src" / f"main.{defn.concept}.patch.txt"
    return [GeneratedFile(
        path=path,
        content=snippet,
        action="edit-snippet",
        step=10,
        description=f"Edit instructions for src/main.py — register {defn.concept} router",
    )]


def _step11_alembic_migration(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 11: alembic/versions/{NEXT}_{concept}s.py  (CREATE)"""
    tmpl = _load_template("backend_migration.py.tmpl")
    path = (
        defn.backend_repo_path
        / "alembic" / "versions"
        / f"{defn.next_migration_number}_{defn.concept}s.py"
    )
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=11,
        description=f"Alembic migration: create {defn.concept}s table",
    )]


def _step12_frontend_registry_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 12: edit snippet for lib/concept-registry.ts."""
    snippet = textwrap.dedent(f"""\
        // ── FILE: lib/concept-registry.ts ───────────────────────────────────────────
        // Add this object to the CONCEPT_REGISTRY array (enabled: false until Step 15):
        {{
          id: "{defn.concept}",
          label: "{defn.concept_tab_label}",
          description: "{defn.concept_description}",
          iconPath: "{defn.concept_icon_path}",
          apiBasePath: "{defn.url_slug}",
          enabled: false,   // flip to true in Step 15 after backend is deployed
        }},
    """)
    path = (
        defn.frontend_repo_path / "lib" / f"concept-registry.{defn.concept}.patch.txt"
    )
    return [GeneratedFile(
        path=path,
        content=snippet,
        action="edit-snippet",
        step=12,
        description=f"Edit instructions for lib/concept-registry.ts — add {defn.concept}",
    )]


def _step13_frontend_panel(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 13: app/components/console/panels/{ConceptLabel}Panel.tsx  (CREATE)"""
    tmpl = _load_template("frontend_panel.tsx.tmpl")
    path = (
        defn.frontend_repo_path
        / "app" / "components" / "console" / "panels"
        / f"{defn.concept_label}Panel.tsx"
    )
    return [GeneratedFile(
        path=path,
        content=render(tmpl, defn),
        action="create",
        step=13,
        description=f"{defn.concept_label}Panel React component",
    )]


def _step14_console_client_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 14: edit snippet for app/components/console/ConsoleClient.tsx."""
    snippet = textwrap.dedent(f"""\
        // ── FILE: app/components/console/ConsoleClient.tsx ──────────────────────────
        // Edit A — add import at the top (with other panel imports):
        import {defn.concept_label}Panel from "@/app/components/console/panels/{defn.concept_label}Panel";

        // Edit B — add to the CONCEPT_PANELS map:
        "{defn.concept}": {defn.concept_label}Panel,
    """)
    path = (
        defn.frontend_repo_path
        / "app" / "components" / "console"
        / f"ConsoleClient.{defn.concept}.patch.txt"
    )
    return [GeneratedFile(
        path=path,
        content=snippet,
        action="edit-snippet",
        step=14,
        description=(
            f"Edit instructions for ConsoleClient.tsx — import + register "
            f"{defn.concept_label}Panel"
        ),
    )]


def _step15_enable_snippet(defn: ConceptDefinition) -> list[GeneratedFile]:
    """Step 15: reminder snippet — flip enabled: true after deploy."""
    snippet = textwrap.dedent(f"""\
        // ── FILE: lib/concept-registry.ts ───────────────────────────────────────────
        // ⚠  Do this ONLY after Steps 10 + 11 have been deployed to production.
        // Change the entry added in Step 12:
        //   enabled: false,
        // to:
        //   enabled: true,
        //
        // This makes the '{defn.concept_tab_label}' tab visible in the project console.
    """)
    path = (
        defn.frontend_repo_path / "lib" / f"concept-registry.{defn.concept}.enable.txt"
    )
    return [GeneratedFile(
        path=path,
        content=snippet,
        action="edit-snippet",
        step=15,
        description=f"Reminder: flip enabled: true for {defn.concept} after backend deploy",
    )]


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

_ALL_STEPS = [
    _step1_framework_manifest,
    _step2_framework_schemas,
    _step3_framework_observer,
    _step3_framework_init,
    _step4_decorator_snippet,
    _step5_backend_model,
    _step6_backend_models_init_snippet,
    _step7_backend_schemas,
    _step8_backend_operations,
    _step9_backend_api,
    _step10_backend_main_snippet,
    _step11_alembic_migration,
    _step12_frontend_registry_snippet,
    _step13_frontend_panel,
    _step14_console_client_snippet,
    _step15_enable_snippet,
]


def generate_concept(defn: ConceptDefinition) -> WorkflowResult:
    """Generate all files for a new observable concept.

    This is a **pure** function — it does not write anything to disk.
    Inspect ``result.files`` then call :func:`write_files` to materialise them.

    Args:
        defn: Fully populated :class:`ConceptDefinition`.

    Returns:
        :class:`WorkflowResult` containing generated files and any errors.
    """
    errors = validate_concept_definition(defn)
    if errors:
        return WorkflowResult(concept=defn.concept, errors=errors)

    all_files: list[GeneratedFile] = []
    gen_errors: list[str] = []

    for step_fn in _ALL_STEPS:
        try:
            all_files.extend(step_fn(defn))
        except Exception as exc:
            gen_errors.append(f"{step_fn.__name__}: {exc}")

    return WorkflowResult(
        concept=defn.concept,
        files=all_files,
        errors=gen_errors,
    )


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def write_files(result: WorkflowResult, *, dry_run: bool = False) -> None:
    """Write all *create* and *overwrite* files from *result* to disk.

    *edit-snippet* files are written to disk as ``*.patch.txt`` / ``*.enable.txt``
    helper files so developers can apply the changes manually.

    Args:
        result:  Completed :class:`WorkflowResult` from :func:`generate_concept`.
        dry_run: If ``True``, print what would be written without touching disk.
    """
    if not result.success:
        raise RuntimeError(
            "WorkflowResult has errors; fix them before writing files:\n"
            + "\n".join(f"  • {e}" for e in result.errors)
        )

    for gf in result.files:
        if dry_run:
            print(f"[DRY RUN] {gf}")
            continue

        gf.path.parent.mkdir(parents=True, exist_ok=True)
        gf.path.write_text(gf.content, encoding="utf-8")
        print(f"  [OK] {gf}")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(result: WorkflowResult) -> None:
    """Print a human-readable summary of the workflow result to stdout."""
    sep = "-" * 72

    print(f"\n{sep}")
    print(f"  add_observable_concept: {result.concept!r}")
    print(sep)

    if result.errors:
        print("\n[ERRORS]")
        for err in result.errors:
            print(f"  [X] {err}")
        print()
        return

    if result.warnings:
        print("\n[WARNINGS]")
        for w in result.warnings:
            print(f"  [!] {w}")

    create_files = [f for f in result.files if f.action in ("create", "overwrite")]
    patch_files = [f for f in result.files if f.action == "edit-snippet"]

    if create_files:
        print(f"\n[FILES TO CREATE/OVERWRITE]  ({len(create_files)})")
        for gf in create_files:
            print(f"  +  {gf.path}")

    if patch_files:
        print(f"\n[MANUAL EDITS REQUIRED]  ({len(patch_files)})")
        for gf in patch_files:
            print(f"  ~  {gf.path}")
            print(f"       -> {gf.description}")

    print(f"\n[CHECKLIST]")
    for line in result.checklist:
        print(f"  {line}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cookbook.skills.add_observable_concept.workflow",
        description="Scaffold all layers for a new NeoSyntropy observable concept.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python -m cookbook.skills.add_observable_concept.workflow \\
                --concept agent \\
                --concept-label Agent \\
                --concept-tab-label Agents \\
                --concept-description "AI agents registered in this project." \\
                --concept-icon-path "M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2z" \\
                --concept-natural-key name \\
                --next-migration 0019 \\
                --prev-migration 0018 \\
                --framework-repo /path/to/neosyntropy-framework \\
                --backend-repo /path/to/neo_syntropy_backend \\
                --frontend-repo /path/to/neo_syntropy_frontend \\
                --write
        """),
    )
    parser.add_argument("--concept", required=True, help="snake_case concept id (e.g. 'agent')")
    parser.add_argument("--concept-label", required=True, help="PascalCase class name (e.g. 'Agent')")
    parser.add_argument("--concept-tab-label", required=True, help="UI tab label (e.g. 'Agents')")
    parser.add_argument("--concept-description", required=True, help="One-sentence description")
    parser.add_argument("--concept-icon-path", required=True, help="SVG path d= value")
    parser.add_argument("--concept-natural-key", default="name", help="Natural key field (default: name)")
    parser.add_argument("--next-migration", required=True, help="Next migration number (e.g. '0019')")
    parser.add_argument("--prev-migration", required=True, help="Previous migration number (e.g. '0018')")
    parser.add_argument("--url-slug", default=None, help="URL path slug (default: concept + 's')")
    parser.add_argument(
        "--framework-repo", required=True,
        help="Absolute path to neosyntropy-framework repo root"
    )
    parser.add_argument(
        "--backend-repo", required=True,
        help="Absolute path to neo_syntropy_backend repo root"
    )
    parser.add_argument(
        "--frontend-repo", required=True,
        help="Absolute path to neo_syntropy_frontend repo root"
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write files to disk (default: dry-run only)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    defn = ConceptDefinition(
        concept=args.concept,
        concept_label=args.concept_label,
        concept_tab_label=args.concept_tab_label,
        concept_description=args.concept_description,
        concept_icon_path=args.concept_icon_path,
        concept_natural_key=args.concept_natural_key,
        next_migration_number=args.next_migration,
        prev_migration_number=args.prev_migration,
        url_slug=args.url_slug,
        framework_repo_path=Path(args.framework_repo),
        backend_repo_path=Path(args.backend_repo),
        frontend_repo_path=Path(args.frontend_repo),
    )

    result = generate_concept(defn)
    print_report(result)

    if not result.success:
        return 1

    write_files(result, dry_run=not args.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
