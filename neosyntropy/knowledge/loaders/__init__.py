"""Remote content loaders for Knowledge.

This module provides loaders for various cloud storage providers:
- S3Loader: AWS S3
- GCSLoader: Google Cloud Storage
- SharePointLoader: Microsoft SharePoint
- GitHubLoader: GitHub repositories
- AzureBlobLoader: Azure Blob Storage

All loaders inherit from BaseLoader which provides common utilities for
computing content names, creating content entries, and merging metadata.
"""

from neosyntropy.knowledge.loaders.azure_blob import AzureBlobLoader
from neosyntropy.knowledge.loaders.base import BaseLoader, FileToProcess
from neosyntropy.knowledge.loaders.gcs import GCSLoader
from neosyntropy.knowledge.loaders.github import GitHubLoader
from neosyntropy.knowledge.loaders.s3 import S3Loader
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
