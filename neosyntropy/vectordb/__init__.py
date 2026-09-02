# Backward-compatibility shim — the vector DB adapters have moved to
# neosyntropy.databases.vector. This re-export keeps existing imports working.
from neosyntropy.databases.vector.base import VectorDb

__all__ = ["VectorDb"]
