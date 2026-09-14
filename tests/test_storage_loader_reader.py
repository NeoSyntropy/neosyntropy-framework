"""Cloud loaders must be able to select a reader and ingest a file.

``_select_reader_by_uri`` was documented on Knowledge but never implemented,
so every S3/GCS/Azure/SharePoint/GitHub ingest that did not take the skip path
raised AttributeError after writing a PROCESSING row.
"""

from __future__ import annotations

from io import BytesIO

from neosyntropy.knowledge.content import Content, ContentStatus
from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.knowledge import Knowledge
from neosyntropy.knowledge.reader.text_reader import TextReader
from neosyntropy.knowledge.remote_content.remote_content import S3Content


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _S3Resource:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def get(self) -> dict[str, _Body]:
        return {"Body": _Body(self._data)}


class _ReadableS3Object:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.bucket_name = "docs"
        self._data = data

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket_name}/{self.name}"

    def get_resource(self) -> _S3Resource:
        return _S3Resource(self._data)


class _ReadableS3Bucket:
    name = "docs"

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def object(self, name: str) -> _ReadableS3Object:
        return _ReadableS3Object(name, self._objects[name])


class _RecordingVectorDb:
    def __init__(self) -> None:
        self.inserted: list[tuple[str, list[Document]]] = []

    def content_hash_exists(self, content_hash: str) -> bool:
        return False

    def insert(self, content_hash: str, documents: list[Document], filters=None) -> None:
        self.inserted.append((content_hash, list(documents)))


def test_select_reader_by_uri_prefers_explicit_reader():
    knowledge = Knowledge()
    explicit = TextReader(chunk=False)
    selected = knowledge._select_reader_by_uri("s3://docs/readme.md", explicit)
    assert selected is explicit


def test_select_reader_by_uri_infers_reader_from_extension():
    knowledge = Knowledge()
    selected = knowledge._select_reader_by_uri("s3://docs/notes.txt")
    assert selected is not None
    assert selected.__class__.__name__ == "TextReader"

    selected = knowledge._select_reader_by_uri("readme.md")
    assert selected is not None
    assert selected.__class__.__name__ == "MarkdownReader"


def test_s3_load_inserts_file_without_storage_directory():
    """Happy-path S3 ingest used to crash twice: missing reader helper, then
    download to a non-existent ``storage/`` directory for non-PDF objects.
    """
    vector_db = _RecordingVectorDb()
    knowledge = Knowledge(vector_db=vector_db)
    content = Content(
        name="readme",
        remote_content=S3Content(
            bucket=_ReadableS3Bucket({"readme.txt": b"hello from s3"}),
            key="readme.txt",
        ),
        reader=TextReader(chunk=False),
    )

    knowledge._load_from_s3(content, upsert=False, skip_if_exists=False)

    assert len(knowledge.contents) == 1
    assert knowledge.contents[0].status == ContentStatus.COMPLETED
    assert knowledge.contents[0].file_type == "s3"
    assert len(vector_db.inserted) == 1
    _hash, documents = vector_db.inserted[0]
    assert documents
    assert "hello from s3" in documents[0].content


def test_select_reader_by_uri_can_read_bytesio():
    knowledge = Knowledge()
    reader = knowledge._select_reader_by_uri("policy.txt")
    assert reader is not None
    docs = reader.read(BytesIO(b"renewals are billed monthly"), name="policy.txt")
    assert docs
    assert "renewals are billed monthly" in docs[0].content
