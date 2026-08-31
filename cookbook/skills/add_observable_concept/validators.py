"""Validation helpers for the add_observable_concept skill.

Each function returns a list of error strings (empty list = valid).
Call ``validate_concept_definition`` to run all checks at once.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cookbook.skills.add_observable_concept.schemas import ConceptDefinition

# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------

_SNAKE_RE = re.compile(r'^[a-z][a-z0-9_]*$')
_PASCAL_RE = re.compile(r'^[A-Z][a-zA-Z0-9]+$')
_MIGRATION_RE = re.compile(r'^\d{4}$')
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def validate_snake_case(value: str, field: str) -> list[str]:
    """Ensure *value* is lowercase snake_case starting with a letter."""
    if not value:
        return [f"'{field}' must not be empty"]
    if not _SNAKE_RE.match(value):
        return [
            f"'{field}' = {value!r} must be snake_case "
            f"(lowercase letters/digits/underscores, start with a letter)"
        ]
    return []


def validate_pascal_case(value: str, field: str) -> list[str]:
    """Ensure *value* is PascalCase (starts with uppercase, no underscores)."""
    if not value:
        return [f"'{field}' must not be empty"]
    if not _PASCAL_RE.match(value):
        return [
            f"'{field}' = {value!r} must be PascalCase "
            f"(start with uppercase letter, no underscores, e.g. 'MyThing')"
        ]
    return []


def validate_migration_number(value: str, field: str) -> list[str]:
    """Ensure *value* is a 4-digit zero-padded decimal string."""
    if not _MIGRATION_RE.match(value):
        return [
            f"'{field}' = {value!r} must be a 4-digit zero-padded number "
            f"(e.g. '0018')"
        ]
    return []


def validate_natural_key(value: str, field: str) -> list[str]:
    """Ensure *value* is a snake_case Python identifier."""
    errors: list[str] = []
    if not value:
        errors.append(f"'{field}' must not be empty")
        return errors
    if not _IDENTIFIER_RE.match(value):
        errors.append(
            f"'{field}' = {value!r} must be a valid Python identifier"
        )
    if not _SNAKE_RE.match(value):
        errors.append(
            f"'{field}' = {value!r} should be snake_case (all lowercase)"
        )
    return errors


def validate_migration_sequence(next_num: str, prev_num: str) -> list[str]:
    """Ensure *next_num* is exactly *prev_num* + 1."""
    try:
        n = int(next_num)
        p = int(prev_num)
        if n != p + 1:
            return [
                f"next_migration_number ({next_num}) must be exactly one greater than "
                f"prev_migration_number ({prev_num}); got {n} vs {p}"
            ]
    except ValueError:
        pass  # format errors reported separately
    return []


def validate_not_empty(value: str, field: str) -> list[str]:
    if not value or not value.strip():
        return [f"'{field}' must not be empty"]
    return []


# ---------------------------------------------------------------------------
# Repo-path validators
# ---------------------------------------------------------------------------

_REPO_MARKERS: dict[str, str] = {
    "framework_repo_path": "neosyntropy/__init__.py",
    "backend_repo_path": "alembic.ini",
    "frontend_repo_path": "next.config.ts",
}


def validate_repo_path(path: Path, field: str) -> list[str]:
    """Check that *path* exists and contains the expected repo marker file."""
    errors: list[str] = []
    if not path.exists():
        errors.append(f"'{field}' path does not exist: {path}")
        return errors
    marker = _REPO_MARKERS.get(field)
    if marker and not (path / marker).exists():
        errors.append(
            f"'{field}' does not look like the expected repo "
            f"(missing {marker}): {path}"
        )
    return errors


# ---------------------------------------------------------------------------
# Collision check
# ---------------------------------------------------------------------------

def validate_no_collision(defn: "ConceptDefinition") -> list[str]:
    """Return errors if any primary output artifact already exists on disk.

    A pre-existing file means the concept may already be implemented; the skill
    should not silently overwrite it.
    """
    errors: list[str] = []

    checks: list[tuple[Path, str]] = [
        (
            defn.framework_repo_path / "neosyntropy" / "monitor" / defn.concept / "manifest.py",
            "framework manifest.py",
        ),
        (
            defn.backend_repo_path / "src" / "models" / f"{defn.concept}.py",
            "backend model",
        ),
        (
            defn.backend_repo_path / "alembic" / "versions" / f"{defn.next_migration_number}_{defn.concept}s.py",
            "alembic migration",
        ),
    ]
    for artifact_path, label in checks:
        if artifact_path.exists():
            errors.append(
                f"Collision detected — {label} already exists: {artifact_path}\n"
                f"  If you intended to update an existing concept, remove the file first."
            )
    return errors


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def validate_concept_definition(defn: "ConceptDefinition") -> list[str]:
    """Run all validations and return a combined list of error messages.

    An empty list means the definition is ready to be scaffolded.
    """
    errors: list[str] = []

    errors.extend(validate_snake_case(defn.concept, "concept"))
    errors.extend(validate_pascal_case(defn.concept_label, "concept_label"))
    errors.extend(validate_snake_case(defn.concept_natural_key, "concept_natural_key"))
    errors.extend(validate_natural_key(defn.concept_natural_key, "concept_natural_key"))
    errors.extend(validate_migration_number(defn.next_migration_number, "next_migration_number"))
    errors.extend(validate_migration_number(defn.prev_migration_number, "prev_migration_number"))
    errors.extend(validate_migration_sequence(defn.next_migration_number, defn.prev_migration_number))
    errors.extend(validate_not_empty(defn.concept_description, "concept_description"))
    errors.extend(validate_not_empty(defn.concept_icon_path, "concept_icon_path"))
    errors.extend(validate_not_empty(defn.concept_tab_label, "concept_tab_label"))

    # Repo paths
    errors.extend(validate_repo_path(defn.framework_repo_path, "framework_repo_path"))
    errors.extend(validate_repo_path(defn.backend_repo_path, "backend_repo_path"))
    errors.extend(validate_repo_path(defn.frontend_repo_path, "frontend_repo_path"))

    # Collision check (only when paths are valid)
    if not errors:
        errors.extend(validate_no_collision(defn))

    return errors
