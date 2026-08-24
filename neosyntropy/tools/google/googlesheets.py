"""Backward-compatibility stub. Use neosyntropy.tools.google.sheets instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.googlesheets' is deprecated. "
    "Use 'from neosyntropy.tools.google.sheets import GoogleSheetsTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.sheets import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.sheets import GoogleSheetsTools  # noqa: F811, E402

__all__ = ["GoogleSheetsTools"]
