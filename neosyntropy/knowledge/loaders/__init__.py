"""Remote content loaders for Knowledge.

This module provides loaders for various cloud storage providers:
- S3Loader: AWS S3  (→ neosyntropy.databases.storage)
- GCSLoader: Google Cloud Storage  (→ neosyntropy.databases.storage)
- AzureBlobLoader: Azure Blob Storage  (→ neosyntropy.databases.storage)
- SharePointLoader: Microsoft SharePoint
- GitHubLoader: GitHub repositories

All loaders inherit from BaseLoader which provides common utilities for
computing content names, creating content entries, and merging metadata.
"""

from neosyntropy.databases.storage.azure_blob import AzureBlobLoader
from neosyntropy.databases.storage.gcs import GCSLoader
from neosyntropy.databases.storage.s3 import S3Loader
from neosyntropy.knowledge.loaders.base import BaseLoader, FileToProcess
from neosyntropy.knowledge.loaders.github import GitHubLoader
from neosyntropy.knowledge.loaders.sharepoint import SharePointLoader

__all__ = [
    "BaseLoader",
    "FileToProcess",
    "S3Loader",
    "GCSLoader",
    "SharePointLoader",
    "GitHubLoader",
    "AzureBlobLoader",
]
