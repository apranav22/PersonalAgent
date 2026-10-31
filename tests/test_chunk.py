import chunk
from chunk import (
    MAX_TOKENS,
    approx_tokens,
    chunk_note,
    merge_small,
    recursive_split,
    split_by_headings,
)


def test_approx_tokens_is_chars_over_four():
    assert approx_tokens("") == 0
    assert approx_tokens("a" * 400) == 100


def test_split_by_headings_keeps_heading_with_section():
    md = "intro text\n## Section A\nalpha\n### Sub B\nbeta"
    assert split_by_headings(md) == ["intro text", "## Section A\nalpha", "### Sub B\nbeta"]


def test_recursive_split_respects_max_tokens_and_keeps_content():
    paras = [f"paragraph {i} " + ("x" * 1000) for i in range(10)]  # ~250 tokens each
    text = "\n\n".join(paras)
    assert approx_tokens(text) > MAX_TOKENS

    out = recursive_split(text, ["\n\n", "\n"])
    assert len(out) > 1
    assert all(approx_tokens(c) <= MAX_TOKENS for c in out)
    for i in range(10):
        assert sum(f"paragraph {i} " in c for c in out) == 1


def test_recursive_split_short_text_untouched():
    assert recursive_split("short", ["\n\n"]) == ["short"]


def test_merge_small_combines_undersized_neighbours():
    a, b, big = "a" * 100, "b" * 100, "c" * 1900
    merged = merge_small([a, b, big])
    assert merged == [f"{a}\n\n{b}", big]


def _write_note(tmp_path, name, content):
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_chunk_note_prepends_title_and_numbers_chunks(tmp_path):
    p = _write_note(tmp_path, "n1", "# Original H1\n\nSome body text about sourdough.")
    recs = chunk_note("n1", "Bread Notes", str(p), modified_at=1700000000)
    assert len(recs) == 1
    r = recs[0]
    assert r["text"] == "# Bread Notes\n\nSome body text about sourdough."
    assert "Original H1" not in r["text"]  # leading H1 is replaced by manifest title
    assert (r["note_id"], r["chunk_index"], r["chunk_total"]) == ("n1", 0, 1)
    assert r["modified_at"] == 1700000000
    assert r["tokens_approx"] == approx_tokens(r["text"])


def test_chunk_note_long_note_splits_on_headings_and_sentences(tmp_path):
    sentence = "This is a fairly ordinary sentence about my weekly plans. "
    sections = [f"## Part {i}\n" + sentence * 40 for i in range(4)]  # ~580 tokens each
    p = _write_note(tmp_path, "long", "# T\n\n" + "intro\n" + "\n".join(sections))
    recs = chunk_note("long", "Long Note", str(p))

    assert len(recs) >= 4
    assert [r["chunk_index"] for r in recs] == list(range(len(recs)))
    assert all(r["chunk_total"] == len(recs) for r in recs)
    header = "# Long Note\n\n"
    for r in recs:
        assert r["text"].startswith(header)
        assert approx_tokens(r["text"][len(header):]) <= MAX_TOKENS
    assert r["modified_at"] == 0


def test_chunk_note_empty_or_missing(tmp_path):
    p = _write_note(tmp_path, "empty", "# Only a title\n")
    assert chunk_note("empty", "Only a title", str(p)) == []
    assert chunk_note("nope", "Missing", str(tmp_path / "missing.md")) == []


def test_load_existing_chunks_groups_and_skips_malformed(tmp_path, monkeypatch, capsys):
    f = tmp_path / "chunks.jsonl"
    f.write_text(
        '{"note_id": "a", "chunk_index": 0}\n'
        "not json\n"
        '{"note_id": "a", "chunk_index": 1}\n'
        '{"note_id": "b", "chunk_index": 0}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(chunk, "CHUNKS_FILE", f)
    by_note = chunk.load_existing_chunks()
    assert {k: len(v) for k, v in by_note.items()} == {"a": 2, "b": 1}
    assert "skipping malformed" in capsys.readouterr().out


def test_write_then_load_chunks_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(chunk, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    data = {"x": [{"note_id": "x", "chunk_index": 0, "text": "héllo"}]}
    chunk.write_chunks(data)
    assert chunk.load_existing_chunks() == data
