import os
from pathlib import Path

def safe_join_relative_path(base_dir: str, path: str) -> Path:
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        from neosyntropy.exceptions import PathSecurityError
        raise PathSecurityError(f"Path traversal detected: {path}")
    return target
