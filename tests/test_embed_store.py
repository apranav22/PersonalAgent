"""LanceDB upsert/delete and vector search, using a deterministic fake encoder
in place of the sentence-transformer so no model download is needed."""

import numpy as np
import pytest

import embed
from conftest import make_chunk

DIM = 8


class FakeModel:
    """Maps each text to a fixed unit vector based on a keyword."""

    KEYS = ["bread", "gym", "books", "travel", "music", "code", "work", "food"]

    def _vec(self, text):
        v = np.full(DIM, 0.01, dtype=np.float32)
        for i, k in enumerate(self.KEYS):
            if k in text.lower():
                v[i] = 1.0
        return v / np.linalg.norm(v)

    def encode(self, texts, **_):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def table(tmp_path):
    import lancedb

    db = lancedb.connect(str(tmp_path / "lancedb"))
    return db, embed.get_or_create_table(db, DIM)


def _ids(table):
    return sorted(table.to_arrow().column("note_id").to_pylist())


def test_upsert_replaces_rows_for_changed_note(table):
    _, tbl = table
    model = FakeModel()
    n = embed.embed_and_upsert(tbl, [
        make_chunk("bread", 0, "bread one"),
        make_chunk("bread", 1, "bread two"),
        make_chunk("gym", 0, "gym"),
    ], model)
    assert n == 3 and tbl.count_rows() == 3

    # Re-embedding a changed note replaces its rows rather than duplicating them.
    embed.embed_and_upsert(tbl, [make_chunk("bread", 0, "bread rewritten")], model)
    assert _ids(tbl) == ["bread", "gym"]
    assert embed.embed_and_upsert(tbl, [], model) == 0


def test_delete_notes_handles_quotes_in_ids(table):
    _, tbl = table
    model = FakeModel()
    embed.embed_and_upsert(tbl, [make_chunk("it's", 0, "x"), make_chunk("gym", 0, "y")], model)
    embed.delete_notes(tbl, ["it's"])
    assert _ids(tbl) == ["gym"]


def test_reopening_table_with_wrong_dimension_fails(table):
    db, _ = table
    assert embed.get_or_create_table(db, DIM).count_rows() == 0
    with pytest.raises(RuntimeError, match="dimension"):
        embed.get_or_create_table(db, DIM * 2)


def test_semantic_search_returns_nearest_chunk(table, fresh_retrieve, monkeypatch):
    _, tbl = table
    r = fresh_retrieve
    model = FakeModel()
    chunks = [make_chunk(k, 0, f"notes about {k}") for k in FakeModel.KEYS]
    embed.embed_and_upsert(tbl, chunks, model)

    # fresh_retrieve points retrieve.DB_DIR at the same tmp_path/lancedb
    monkeypatch.setattr(r, "load_embed_model", lambda: model)
    hits = r.semantic_search("where did I put the travel plan", top_k=3)
    assert hits[0][0]["note_id"] == "travel"
    assert len(hits) == 3


def test_semantic_search_missing_table_raises(fresh_retrieve):
    with pytest.raises(RuntimeError, match="run embed.py first"):
        fresh_retrieve.semantic_search("q", top_k=3)


def test_load_chunks_filters_by_note_id(tmp_path, monkeypatch):
    from conftest import write_jsonl

    f = tmp_path / "chunks.jsonl"
    write_jsonl(f, [make_chunk("a", 0, "x"), make_chunk("b", 0, "y")])
    monkeypatch.setattr(embed, "CHUNKS_FILE", f)
    assert [c["note_id"] for c in embed.load_chunks()] == ["a", "b"]
    assert [c["note_id"] for c in embed.load_chunks(note_ids=["b"])] == ["b"]
