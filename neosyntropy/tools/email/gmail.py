"""Backward-compatibility stub. Use neosyntropy.tools.google.gmail instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.gmail' is deprecated. Use 'from neosyntropy.tools.google.gmail import GmailTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.gmail import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.gmail import GmailTools  # noqa: F811, E402

__all__ = ["GmailTools"]
