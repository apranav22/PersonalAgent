import hashlib
import json

import pytest

import sync


@pytest.fixture
def export(tmp_path, monkeypatch):
    exp = tmp_path / "export"
    exp.mkdir()
    monkeypatch.setattr(sync, "EXPORT_DIR", exp)
    monkeypatch.setattr(sync, "HASHES_FILE", tmp_path / "_hashes.json")

    def write(notes: dict[str, str]):
        (exp / "manifest.txt").write_text(
            "".join(f"{nid}|||Title {nid}\n" for nid in notes), encoding="utf-8"
        )
        for nid, html in notes.items():
            (exp / f"{nid}.html").write_text(html, encoding="utf-8")

    return exp, write


def test_first_run_marks_everything_changed_and_stores_sha256(export):
    exp, write = export
    write({"p1": "<div>one</div>", "p2": "<div>two</div>"})

    changed, deleted = sync.get_diff()
    assert sorted(changed) == ["p1", "p2"]
    assert deleted == []

    stored = json.loads(sync.HASHES_FILE.read_text())
    assert stored["p1"] == hashlib.sha256(b"<div>one</div>").hexdigest()


def test_only_modified_notes_are_reprocessed(export):
    exp, write = export
    write({"p1": "one", "p2": "two", "p3": "three"})
    sync.get_diff()

    assert sync.get_diff() == ([], [])  # nothing changed

    (exp / "p2.html").write_text("two, edited", encoding="utf-8")
    changed, deleted = sync.get_diff()
    assert changed == ["p2"]
    assert deleted == []


def test_removed_notes_are_reported_deleted(export):
    exp, write = export
    write({"p1": "one", "p2": "two"})
    sync.get_diff()

    write({"p1": "one"})  # p2 dropped from manifest
    changed, deleted = sync.get_diff()
    assert changed == []
    assert deleted == ["p2"]
    assert "p2" not in json.loads(sync.HASHES_FILE.read_text())


def test_dry_run_does_not_write_hashes(export):
    _, write = export
    write({"p1": "one"})
    changed, _ = sync.get_diff(dry_run=True)
    assert changed == ["p1"]
    assert not sync.HASHES_FILE.exists()


def test_manifest_entry_without_html_is_skipped(export):
    exp, write = export
    write({"p1": "one"})
    with open(exp / "manifest.txt", "a", encoding="utf-8") as f:
        f.write("ghost|||No file\n")
    changed, _ = sync.get_diff()
    assert changed == ["p1"]
