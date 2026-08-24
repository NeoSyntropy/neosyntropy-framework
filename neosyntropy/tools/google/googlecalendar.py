"""Backward-compatibility stub. Use neosyntropy.tools.google.calendar instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.googlecalendar' is deprecated. "
    "Use 'from neosyntropy.tools.google.calendar import GoogleCalendarTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.calendar import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.calendar import GoogleCalendarTools  # noqa: F811, E402

__all__ = ["GoogleCalendarTools"]
