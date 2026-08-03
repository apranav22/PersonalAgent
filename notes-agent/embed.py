#!/usr/bin/env python3
"""
Step 5: Embed chunks into LanceDB vector store.

Model: all-MiniLM-L6-v2 (384-dim, fast, runs fully local)
Store: notes-agent/lancedb/  (file-based, no server needed)
Table: notes_chunks

Sync-aware:
  --changed note_1 note_2 ...  re-embed only these notes
  --deleted note_1 note_2 ...  remove these notes from the store
  --all                        re-embed everything (first run)

Each row in the table mirrors a chunk from chunks.jsonl + its vector.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")  # suppress urllib3/ssl noise

AGENT_DIR   = Path(__file__).parent
CHUNKS_FILE = AGENT_DIR / "chunks.jsonl"
DB_DIR      = AGENT_DIR / "lancedb"

MODEL_NAME  = "all-MiniLM-L6-v2"
TABLE_NAME  = "notes_chunks"


def load_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model ({MODEL_NAME})...")
    return SentenceTransformer(MODEL_NAME)


def get_or_create_table(db, dim: int):
    import pyarrow as pa
    schema = pa.schema([
        pa.field("note_id",      pa.utf8()),
        pa.field("title",        pa.utf8()),
        pa.field("modified_at",  pa.int64()),
        pa.field("source_path",  pa.utf8()),
        pa.field("chunk_index",  pa.int32()),
        pa.field("chunk_total",  pa.int32()),
        pa.field("tokens_approx",pa.int32()),
        pa.field("text",         pa.utf8()),
        pa.field("vector",       pa.list_(pa.float32(), dim)),
    ])
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        # Validate dimension matches current model
        existing_dim = table.schema.field("vector").type.list_size
        if existing_dim != dim:
            raise RuntimeError(
                f"Vector store has dimension {existing_dim} but model produces {dim}. "
                f"Delete lancedb/ and run embed.py --all to rebuild."
            )
        return table
    else:
        return db.create_table(TABLE_NAME, schema=schema)


def load_chunks(note_ids: Optional[list] = None) -> list:
    if not CHUNKS_FILE.exists():
        return []
    chunks = []
    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        if note_ids is None or chunk["note_id"] in note_ids:
            chunks.append(chunk)
    return chunks


def embed_and_upsert(table, chunks: list[dict], model) -> int:
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(chunks)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

    import pyarrow as pa
    rows = []
    for chunk, vec in zip(chunks, vectors):
        rows.append({
            "note_id":       chunk["note_id"],
            "title":         chunk["title"],
            "modified_at":   int(chunk.get("modified_at") or 0),
            "source_path":   chunk["source_path"],
            "chunk_index":   chunk["chunk_index"],
            "chunk_total":   chunk["chunk_total"],
            "tokens_approx": chunk["tokens_approx"],
            "text":          chunk["text"],
            "vector":        vec.tolist(),
        })

    # Delete existing rows for these notes before inserting updated ones
    note_ids = list({c["note_id"] for c in chunks})
    for nid in note_ids:
        try:
            table.delete(f"note_id = '{nid}'")
        except Exception as e:
            if "empty" not in str(e).lower() and "no such" not in str(e).lower():
                print(f"  WARN: delete failed for {nid}: {e}")

    table.add(rows)
    return len(rows)


def delete_notes(table, note_ids: list[str]) -> None:
    if not note_ids:
        return
    for nid in note_ids:
        try:
            table.delete(f"note_id = '{nid}'")
        except Exception as e:
            if "empty" not in str(e).lower() and "no such" not in str(e).lower():
                print(f"  WARN: delete failed for {nid}: {e}")
    print(f"Deleted {len(note_ids)} notes from vector store")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", nargs="*", help="Note IDs to re-embed")
    parser.add_argument("--deleted", nargs="*", help="Note IDs to remove")
    parser.add_argument("--all",     action="store_true", help="Re-embed everything")
    args = parser.parse_args()

    import lancedb
    DB_DIR.mkdir(exist_ok=True)
    db = lancedb.connect(str(DB_DIR))

    # Determine what to embed
    if args.all or TABLE_NAME not in db.table_names():
        chunks_to_embed = load_chunks()
    elif args.changed:
        chunks_to_embed = load_chunks(note_ids=args.changed)
    else:
        chunks_to_embed = []

    if not chunks_to_embed and not args.deleted:
        print("Nothing to do.")
        return

    model = load_model()
    dim   = model.get_sentence_embedding_dimension()
    table = get_or_create_table(db, dim)

    if args.deleted:
        delete_notes(table, args.deleted)

    n = embed_and_upsert(table, chunks_to_embed, model)

    total = table.count_rows()
    print(f"\nEmbedded {n} chunks. Total in store: {total}")
    print(f"DB: {DB_DIR}")


if __name__ == "__main__":
    main()
