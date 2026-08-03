#!/usr/bin/env python3
"""
Step 3: Chunk markdown notes into retrieval-sized pieces.

Strategy:
  - Split on ## / ### headings first (semantic boundaries)
  - Recursively split oversized sections on \n\n → \n
  - Merge undersized chunks into the next one
  - Prepend "# {title}" to every chunk for embedding context

Output: notes-agent/chunks.jsonl  (one chunk per line)
  Fields: note_id, title, source_path, chunk_index, chunk_total, text

Sync-aware: pass --changed / --deleted to only update affected notes.
"""

import argparse
import json
import re
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).parent
NOTES_DIR = AGENT_DIR / "notes_md"
INDEX_FILE = NOTES_DIR / "_index.json"
CHUNKS_FILE = AGENT_DIR / "chunks.jsonl"

MIN_TOKENS = 50
MAX_TOKENS = 500
TARGET_TOKENS = 300


def approx_tokens(text: str) -> int:
    return len(text) // 4


def split_by_headings(md: str) -> list[str]:
    """Split markdown on ## or ### headings, keeping the heading with its section."""
    parts = re.split(r"(?=\n#{2,3} )", md)
    return [p.strip() for p in parts if p.strip()]


def recursive_split(text: str, separators: list[str]) -> list[str]:
    """Split text using the first separator that produces chunks within MAX_TOKENS.
    Falls back through the separator list."""
    if approx_tokens(text) <= MAX_TOKENS or not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]
    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if approx_tokens(candidate) <= MAX_TOKENS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself may be oversized — recurse
            if approx_tokens(part) > MAX_TOKENS:
                chunks.extend(recursive_split(part.strip(), rest))
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if c]


def merge_small(chunks: list[str]) -> list[str]:
    """Merge consecutive undersized chunks into the next chunk."""
    merged = []
    buffer = ""
    for chunk in chunks:
        if approx_tokens(buffer + " " + chunk) <= MAX_TOKENS:
            buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    return merged


def chunk_note(
    note_id: str, title: str, source_path: str, modified_at: int = None
) -> list[dict]:
    path = Path(source_path)
    if not path.exists():
        return []

    md = path.read_text(encoding="utf-8", errors="replace")

    # Strip the leading # Title line — we'll re-add it to every chunk
    lines = md.splitlines()
    if lines and lines[0].startswith("# "):
        body = "\n".join(lines[1:]).strip()
    else:
        body = md.strip()

    if not body:
        return []

    # 1. Split on headings
    sections = (
        split_by_headings(body) if "\n## " in body or "\n### " in body else [body]
    )

    # 2. Recursively split oversized sections
    raw_chunks = []
    for section in sections:
        if approx_tokens(section) > MAX_TOKENS:
            raw_chunks.extend(recursive_split(section, [". ", "\n\n", "\n"]))
        else:
            raw_chunks.append(section)

    # 3. Merge undersized chunks
    raw_chunks = [c for c in raw_chunks if c.strip()]
    final_chunks = merge_small(raw_chunks)

    # 4. Build output records — prepend title to each chunk
    header = f"# {title}"
    records = []
    for i, text in enumerate(final_chunks):
        full_text = f"{header}\n\n{text}"
        records.append(
            {
                "note_id": note_id,
                "title": title,
                "modified_at": int(modified_at or 0),
                "source_path": source_path,
                "chunk_index": i,
                "chunk_total": len(final_chunks),
                "tokens_approx": approx_tokens(full_text),
                "text": full_text,
            }
        )

    return records


def load_existing_chunks() -> dict[str, list[dict]]:
    """Load chunks.jsonl grouped by note_id."""
    by_note: dict[str, list[dict]] = {}
    if not CHUNKS_FILE.exists():
        return by_note
    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            by_note.setdefault(rec["note_id"], []).append(rec)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARN: skipping malformed chunk line: {e}")
            continue
    return by_note


def write_chunks(by_note: dict[str, list[dict]]) -> None:
    lines = []
    for chunks in by_note.values():
        for chunk in chunks:
            lines.append(json.dumps(chunk, ensure_ascii=False))
    CHUNKS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", nargs="*", help="Note IDs to re-chunk")
    parser.add_argument("--deleted", nargs="*", help="Note IDs to remove")
    parser.add_argument("--all", action="store_true", help="Re-chunk all notes")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print("_index.json not found — run convert_html.py first")
        sys.exit(1)

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in index}

    by_note = load_existing_chunks()

    # Determine which notes to process
    if args.all or not CHUNKS_FILE.exists():
        to_chunk = list(by_id.keys())
    elif args.changed:
        to_chunk = args.changed
    else:
        # Default: chunk everything not yet chunked
        to_chunk = [nid for nid in by_id if nid not in by_note]

    to_delete = set(args.deleted or [])

    # Remove deleted
    for nid in to_delete:
        by_note.pop(nid, None)

    # Re-chunk changed/new
    total_chunks = 0
    for note_id in to_chunk:
        if note_id not in by_id:
            continue
        entry = by_id[note_id]
        chunks = chunk_note(
            note_id, entry["title"], entry["path"], entry.get("modified_at")
        )
        if chunks:
            by_note[note_id] = chunks
            total_chunks += len(chunks)
        else:
            by_note.pop(note_id, None)

    write_chunks(by_note)

    total = sum(len(v) for v in by_note.values())
    print(f"Chunked {len(to_chunk)} notes → {total_chunks} new chunks")
    print(f"Total chunks in store: {total}")
    print(f"Output: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
