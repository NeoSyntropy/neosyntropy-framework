"""Backward-compatibility stub. Use neosyntropy.tools.google.drive instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.google_drive' is deprecated. "
    "Use 'from neosyntropy.tools.google.drive import GoogleDriveTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.drive import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.drive import GoogleDriveTools  # noqa: F811, E402

__all__ = ["GoogleDriveTools"]
