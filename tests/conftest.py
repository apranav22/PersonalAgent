"""Shared fixtures. The pipeline modules live in notes-agent/ and import each
other as siblings (``from utils import ...``), so put that dir on sys.path."""

import json
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent / "notes-agent"
sys.path.insert(0, str(AGENT_DIR))


def make_chunk(note_id: str, idx: int, text: str, title: str | None = None) -> dict:
    """A chunk record in the same shape chunk.py writes to chunks.jsonl."""
    title = title or note_id
    return {
        "note_id": note_id,
        "title": title,
        "modified_at": 0,
        "source_path": f"/fake/{note_id}.md",
        "chunk_index": idx,
        "chunk_total": 1,
        "tokens_approx": len(text) // 4,
        "text": f"# {title}\n\n{text}",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture
def fresh_retrieve(tmp_path, monkeypatch):
    """retrieve.py with file paths pointed at tmp_path and its caches cleared."""
    import retrieve

    monkeypatch.setattr(retrieve, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(retrieve, "GRAPH_FILE", tmp_path / "graph.json")
    monkeypatch.setattr(retrieve, "DB_DIR", tmp_path / "lancedb")
    monkeypatch.setattr(retrieve, "_chunks_cache", [])
    monkeypatch.setattr(retrieve, "_chunks_mtime", 0.0)
    monkeypatch.setattr(retrieve, "_graph_cache", {})
    monkeypatch.setattr(retrieve, "_graph_mtime", 0.0)
    monkeypatch.setattr(retrieve, "_bm25", None)
    monkeypatch.setattr(retrieve, "_bm25_mtime", 0.0)
    return retrieve
