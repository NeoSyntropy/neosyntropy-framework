from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neosyntropy.cloud.remote.snapshot import read_graph_snapshot, write_graph_snapshot


def _inputs(bundle: bytes = b"\x1f\x8bexact compressed bytes") -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, bytes],
]:
    digest = hashlib.sha256(bundle).hexdigest()
    graph: dict[str, object] = {
        "id": "graph-1",
        "project_id": "project-1",
        "manifest": {"entry": "Start", "unicode": "\u03bb"},
        "recovery_manifest": {"revision": "r1", "recoverable": True},
    }
    associations: list[dict[str, object]] = [
        {
            "owner_role": "node_handler",
            "artifact": {"id": "artifact-1", "sha256": digest},
        }
    ]
    return graph, associations, {digest: bundle}


def test_writes_complete_snapshot_and_preserves_bundle_bytes(tmp_path: Path) -> None:
    graph, associations, bundles = _inputs()

    target = write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    assert target == tmp_path / ".neosyntropy/project-1/graphs/graph-1"
    assert json.loads((target / "structure.json").read_bytes()) == graph["manifest"]
    assert json.loads((target / "recovery.json").read_bytes()) == graph["recovery_manifest"]
    assert json.loads((target / "artifacts.json").read_bytes()) == associations
    digest, bundle = next(iter(bundles.items()))
    assert (target / "artifacts" / f"{digest}.json.gz").read_bytes() == bundle
    assert (tmp_path / ".neosyntropy/.gitignore").read_text() == (
        "# NeoSyntropy local snapshots\n*\n!.gitignore\n"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "../escape"),
        ("project_id", "project/escape"),
        ("id", r"graph\escape"),
        ("id", ".."),
        ("id", ""),
    ],
)
def test_rejects_unsafe_path_ids_without_writing(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    graph, associations, bundles = _inputs()
    graph[field] = value

    with pytest.raises(ValueError):
        write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    assert not (tmp_path / ".neosyntropy").exists()


def test_rejects_invalid_or_mismatched_hashes_without_writing(tmp_path: Path) -> None:
    graph, associations, bundles = _inputs()
    digest = next(iter(bundles))
    associations[0]["artifact"] = {"sha256": digest.upper()}

    with pytest.raises(ValueError, match="lowercase 64-character"):
        write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)
    assert not (tmp_path / ".neosyntropy").exists()

    associations[0]["artifact"] = {"sha256": digest}
    with pytest.raises(ValueError, match="does not match"):
        write_graph_snapshot(graph, associations, {digest: b"tampered"}, project_root=tmp_path)
    assert not (tmp_path / ".neosyntropy").exists()


def test_requires_bundle_for_each_association(tmp_path: Path) -> None:
    graph, associations, _ = _inputs()

    with pytest.raises(ValueError, match="hashes differ"):
        write_graph_snapshot(graph, associations, {}, project_root=tmp_path)

    assert not (tmp_path / ".neosyntropy").exists()


def test_prunes_stale_blobs_after_success(tmp_path: Path) -> None:
    graph, associations, bundles = _inputs()
    target = write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)
    stale = target / "artifacts" / f"{'f' * 64}.json.gz"
    stale.write_bytes(b"old")

    write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    assert not stale.exists()


def test_failed_replacement_does_not_prune_stale_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, associations, bundles = _inputs()
    target = write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)
    stale = target / "artifacts" / f"{'f' * 64}.json.gz"
    stale.write_bytes(b"old")

    import neosyntropy.cloud.remote.snapshot as snapshot

    real_replace = snapshot.os.replace
    calls = 0

    def fail_during_commit(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(snapshot.os, "replace", fail_during_commit)
    with pytest.raises(OSError, match="simulated"):
        write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    assert stale.read_bytes() == b"old"
    assert not list(tmp_path.rglob("*.tmp"))


def test_preserves_existing_gitignore_content(tmp_path: Path) -> None:
    graph, associations, bundles = _inputs()
    ignore = tmp_path / ".neosyntropy/.gitignore"
    ignore.parent.mkdir()
    ignore.write_text("custom-entry\n", encoding="utf-8")

    write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    assert ignore.read_text(encoding="utf-8") == (
        "custom-entry\n# NeoSyntropy local snapshots\n*\n!.gitignore\n"
    )


def test_reads_matching_verified_snapshot_without_server_download(
    tmp_path: Path,
) -> None:
    graph, associations, bundles = _inputs()
    write_graph_snapshot(graph, associations, bundles, project_root=tmp_path)

    snapshot = read_graph_snapshot(graph, project_root=tmp_path)

    assert snapshot == {
        "graph": graph,
        "artifacts": associations,
        "bundles": bundles,
    }


def test_cache_miss_for_changed_revision_or_tampered_bundle(tmp_path: Path) -> None:
    graph, associations, bundles = _inputs()
    target = write_graph_snapshot(
        graph,
        associations,
        bundles,
        project_root=tmp_path,
    )
    changed = dict(graph)
    changed["recovery_manifest"] = {"revision": "r2", "recoverable": True}
    assert read_graph_snapshot(changed, project_root=tmp_path) is None

    digest = next(iter(bundles))
    (target / "artifacts" / f"{digest}.json.gz").write_bytes(b"tampered")
    assert read_graph_snapshot(graph, project_root=tmp_path) is None
