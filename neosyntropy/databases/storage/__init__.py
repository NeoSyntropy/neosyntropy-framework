from typing import Any

__all__ = [
    "S3Loader",
    "GCSLoader",
    "AzureBlobLoader",
]


def __getattr__(name: str) -> Any:
    if name == "S3Loader":
        from neosyntropy.databases.storage.s3 import S3Loader

        return S3Loader
    if name == "GCSLoader":
        from neosyntropy.databases.storage.gcs import GCSLoader

        return GCSLoader
    if name == "AzureBlobLoader":
        from neosyntropy.databases.storage.azure_blob import AzureBlobLoader

        return AzureBlobLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
