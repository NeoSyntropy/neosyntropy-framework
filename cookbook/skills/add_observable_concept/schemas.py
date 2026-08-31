"""Input / output schemas for the add_observable_concept skill."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class ConceptDefinition(BaseModel):
    """All variables needed to scaffold a new observable concept across all three repos.

    Mirrors the placeholder table in thoughts/add-new-concept.md:
        {CONCEPT}                 → concept
        {CONCEPT_LABEL}           → concept_label
        {CONCEPT_TAB_LABEL}       → concept_tab_label
        {CONCEPT_DESCRIPTION}     → concept_description
        {CONCEPT_ICON_PATH}       → concept_icon_path
        {CONCEPT_NATURAL_KEY}     → concept_natural_key
        {NEXT_MIGRATION_NUMBER}   → next_migration_number
        {PREV_MIGRATION_NUMBER}   → prev_migration_number
    """

    # -- Core identity --------------------------------------------------------
    concept: str = Field(
        description="snake_case concept id used in Python, DB, and URLs. e.g. 'knowledge'"
    )
    concept_label: str = Field(
        description="PascalCase class name. e.g. 'Knowledge'"
    )
    concept_tab_label: str = Field(
        description="Human-readable UI tab label. e.g. 'Knowledge'"
    )
    concept_description: str = Field(
        description="One-sentence empty-state description shown in the console panel."
    )
    concept_icon_path: str = Field(
        description="SVG path d= attribute value for a 24×24 viewBox icon."
    )
    concept_natural_key: str = Field(
        description=(
            "The field that uniquely identifies a record within a project and is used "
            "for upsert. e.g. 'name'. Must be a valid Python identifier."
        )
    )

    # -- Migration numbers ----------------------------------------------------
    next_migration_number: str = Field(
        description="Next sequential alembic revision number, zero-padded to 4 digits. e.g. '0018'"
    )
    prev_migration_number: str = Field(
        description="The previous migration's revision id. e.g. '0017'"
    )

    # -- URL slug -------------------------------------------------------------
    url_slug: str | None = Field(
        default=None,
        description=(
            "URL path segment used in the FastAPI router. Defaults to concept + 's'. "
            "Override when the natural plural is irregular (e.g. 'knowledge_bases')."
        ),
    )

    # -- Repo roots -----------------------------------------------------------
    framework_repo_path: Path = Field(
        description="Absolute path to the neosyntropy-framework repo root."
    )
    backend_repo_path: Path = Field(
        description="Absolute path to the neo_syntropy_backend repo root."
    )
    frontend_repo_path: Path = Field(
        description="Absolute path to the neo_syntropy_frontend repo root."
    )

    @model_validator(mode="after")
    def _defaults(self) -> "ConceptDefinition":
        if self.url_slug is None:
            self.url_slug = f"{self.concept}s"
        return self

    # -- Convenience ----------------------------------------------------------

    @property
    def substitutions(self) -> dict[str, str]:
        """Ordered placeholder → value mapping for template rendering.

        Longer keys come first so that e.g. ``{CONCEPT_LABEL}`` is replaced
        before the shorter ``{CONCEPT}``, preventing partial substitution.
        """
        assert self.url_slug is not None  # set by validator
        return {
            "{CONCEPT_LABEL}": self.concept_label,
            "{CONCEPT_TAB_LABEL}": self.concept_tab_label,
            "{CONCEPT_DESCRIPTION}": self.concept_description,
            "{CONCEPT_ICON_PATH}": self.concept_icon_path,
            "{CONCEPT_NATURAL_KEY}": self.concept_natural_key,
            "{NEXT_MIGRATION_NUMBER}": self.next_migration_number,
            "{PREV_MIGRATION_NUMBER}": self.prev_migration_number,
            "{URL_SLUG}": self.url_slug,
            "{CONCEPT}": self.concept,  # must come last (shortest key)
        }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GeneratedFile(BaseModel):
    """A single file that the workflow produces."""

    path: Path
    content: str
    action: Literal["create", "overwrite", "edit-snippet"]
    step: int
    description: str

    def __str__(self) -> str:
        verb = {"create": "CREATE", "overwrite": "OVERWRITE", "edit-snippet": "EDIT"}.get(
            self.action, self.action.upper()
        )
        return f"[Step {self.step:02d}] {verb} {self.path}"


class WorkflowResult(BaseModel):
    """Full output of the add_observable_concept workflow."""

    concept: str
    files: list[GeneratedFile] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def checklist(self) -> list[str]:
        """Return a markdown-style checklist showing which steps were generated."""
        step_labels = {
            1:  f"monitor/{{concept}}/manifest.py created",
            2:  f"monitor/{{concept}}/schemas.py filled in",
            3:  f"monitor/{{concept}}/observer.py updated",
            4:  f"@workflow / @function_calling accept {{concept}}= param",
            5:  f"src/models/{{concept}}.py created",
            6:  f"src/models/__init__.py updated",
            7:  f"src/schemas/{{concept}}.py created",
            8:  f"src/operations/{{concept}}.py created",
            9:  f"src/apis/{{concept}}.py created",
            10: f"src/main.py updated",
            11: f"Alembic migration created",
            12: f"lib/concept-registry.ts entry added (enabled: false)",
            13: f"{{concept_label}}Panel.tsx created",
            14: f"ConsoleClient.tsx import + CONCEPT_PANELS entry added",
            15: f"enabled: true after deploy",
        }
        completed = {f.step for f in self.files}
        lines = []
        for n, label in step_labels.items():
            mark = "x" if n in completed else " "
            concept = self.concept
            label = label.replace("{concept}", concept).replace("{concept_label}", concept.title())
            lines.append(f"- [{mark}] Step {n:02d} — {label}")
        return lines
