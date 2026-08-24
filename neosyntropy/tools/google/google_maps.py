"""Backward-compatibility stub. Use neosyntropy.tools.google.maps instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.google_maps' is deprecated. "
    "Use 'from neosyntropy.tools.google.maps import GoogleMapTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.maps import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.maps import GoogleMapTools  # noqa: F811, E402

__all__ = ["GoogleMapTools"]
