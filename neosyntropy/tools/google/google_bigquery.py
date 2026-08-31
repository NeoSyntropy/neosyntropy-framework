"""Backward-compatibility stub. Use neosyntropy.tools.google.bigquery instead."""

import warnings

warnings.warn(
    "Importing from 'neosyntropy.tools.google_bigquery' is deprecated. "
    "Use 'from neosyntropy.tools.google.bigquery import GoogleBigQueryTools' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neosyntropy.tools.google.bigquery import *  # noqa: F401, F403, E402
from neosyntropy.tools.google.bigquery import GoogleBigQueryTools, _clean_sql  # noqa: F811, E402

__all__ = ["GoogleBigQueryTools", "_clean_sql"]
