from pathlib import Path


def safe_join_relative_path(base_dir: str | Path, path: str) -> Path:
    """Resolve ``path`` under ``base_dir`` and reject traversal, including
    sibling directories whose names merely share a prefix with the base.
    """
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        from neosyntropy.exceptions import PathSecurityError

        raise PathSecurityError(f"Path traversal detected: {path}")
    return target
