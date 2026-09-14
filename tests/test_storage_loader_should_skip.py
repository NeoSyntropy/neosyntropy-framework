"""Regression: cloud storage loaders must use Knowledge._should_skip's 3-arg contract."""

from __future__ import annotations

import ast
from pathlib import Path

from neosyntropy.databases.storage.s3 import S3Object
from neosyntropy.knowledge.content import Content, ContentStatus
from neosyntropy.knowledge.knowledge import Knowledge
from neosyntropy.knowledge.remote_content.remote_content import S3Content

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER_FILES = (
    REPO_ROOT / "neosyntropy/databases/storage/s3.py",
    REPO_ROOT / "neosyntropy/databases/storage/gcs.py",
    REPO_ROOT / "neosyntropy/databases/storage/azure_blob.py",
    REPO_ROOT / "neosyntropy/knowledge/loaders/sharepoint.py",
    REPO_ROOT / "neosyntropy/knowledge/loaders/github.py",
)


class _ExistingHashVectorDb:
    """Vector DB that reports every content hash as already stored."""

    def content_hash_exists(self, content_hash: str) -> bool:
        return True


class _FakeS3Bucket:
    name = "docs"

    def object(self, name: str) -> S3Object:
        return S3Object(bucket_name=self.name, name=name)


def _should_skip_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_should_skip":
            calls.append(node)
    return calls


def test_storage_loaders_call_should_skip_with_content_upsert_and_flag():
    """GitHub already used the 3-arg contract; the other loaders must match it.

    The previous 2-arg call ``_should_skip(content_entry.content_hash, skip_if_exists)``
    raises TypeError on every S3/GCS/Azure/SharePoint file after the contents-db insert.
    """
    for path in LOADER_FILES:
        calls = _should_skip_calls(path)
        assert calls, f"{path.name} should invoke _should_skip"
        for call in calls:
            assert len(call.args) == 3, (
                f"{path.name}:{call.lineno} must call "
                "_should_skip(content_entry, upsert, skip_if_exists)"
            )
            first = call.args[0]
            assert not (
                isinstance(first, ast.Attribute) and first.attr == "content_hash"
            ), f"{path.name}:{call.lineno} still passes content_hash instead of Content"


def test_s3_load_skip_if_exists_does_not_crash():
    knowledge = Knowledge(vector_db=_ExistingHashVectorDb())
    content = Content(
        name="readme",
        remote_content=S3Content(bucket=_FakeS3Bucket(), key="readme.md"),
    )

    knowledge._load_from_s3(content, upsert=False, skip_if_exists=True)

    assert len(knowledge.contents) == 1
    assert knowledge.contents[0].status == ContentStatus.COMPLETED
    assert knowledge.contents[0].file_type == "s3"
    assert knowledge.contents[0].id == knowledge.contents[0].content_hash
