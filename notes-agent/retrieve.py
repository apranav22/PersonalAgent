#!/usr/bin/env python3
"""
Hybrid retrieval: BM25 (keyword) + semantic (vector) search, fused via RRF.

Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
  - k=60 is standard; dampens the effect of very high ranks
  - doesn't require normalizing scores across two different scales

Usage (standalone):
    python3 retrieve.py "what did I write about confidence?"
    python3 retrieve.py "project ideas" --top-k 5

Returns ranked list of chunks with note title, text, and source path.
"""

import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR = Path(__file__).parent
CHUNKS_FILE = AGENT_DIR / "chunks.jsonl"
GRAPH_FILE = AGENT_DIR / "graph.json"
DB_DIR = AGENT_DIR / "lancedb"
TABLE_NAME = "notes_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

GRAPH_HOP_DISCOUNT = 0.7  # neighbor scores scaled by this factor

_chunks_cache: list = []
_chunks_mtime: float = 0.0

_graph_cache: dict = {}
_graph_mtime: float = 0.0

_embed_model = None

_bm25: object = None
_bm25_mtime: float = 0.0


def load_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model


def load_graph() -> dict:
    global _graph_cache, _graph_mtime
    try:
        mtime = GRAPH_FILE.stat().st_mtime
    except FileNotFoundError:
        return {}
    if mtime == _graph_mtime:
        return _graph_cache
    _graph_cache = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    _graph_mtime = mtime
    return _graph_cache


def graph_expand(results: list, top_k: int, all_chunks: list) -> list:
    """Expand top-k results by one hop in the note graph.

    For each note in the current results, looks up its neighbors in graph.json,
    scores them as (direct_score * edge_weight * GRAPH_HOP_DISCOUNT), and
    merges the best neighbors into the result list.  At most top_k//2 neighbor
    notes are added so direct hits always dominate.
    """
    graph = load_graph()
    if not graph:
        return results

    # Index chunks by note_id for fast lookup
    chunks_by_note: dict[str, list] = {}
    for chunk in all_chunks:
        chunks_by_note.setdefault(chunk["note_id"], []).append(chunk)

    seen_notes = {chunk["note_id"] for chunk, _ in results}

    # Derive the best direct score per note
    best_direct: dict[str, float] = {}
    for chunk, score in results:
        nid = chunk["note_id"]
        best_direct[nid] = max(best_direct.get(nid, 0.0), score)

    # Accumulate candidate neighbor scores
    neighbor_scores: dict[str, float] = {}
    for nid, direct_score in best_direct.items():
        for edge in graph.get(nid, []):
            nnid = edge["note_id"]
            if nnid in seen_notes:
                continue
            candidate = direct_score * edge["weight"] * GRAPH_HOP_DISCOUNT
            neighbor_scores[nnid] = max(neighbor_scores.get(nnid, 0.0), candidate)

    # Take top top_k//2 neighbors
    n_expand = max(1, top_k // 2)
    top_neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[
        :n_expand
    ]

    expanded = list(results)
    for nnid, score in top_neighbors:
        note_chunks = chunks_by_note.get(nnid)
        if not note_chunks:
            continue
        expanded.append((note_chunks[0], score))

    expanded.sort(key=lambda x: x[1], reverse=True)
    return expanded[:top_k]


def load_all_chunks() -> list:
    global _chunks_cache, _chunks_mtime
    try:
        mtime = CHUNKS_FILE.stat().st_mtime
    except FileNotFoundError:
        return []
    if mtime == _chunks_mtime:
        return _chunks_cache
    chunks = []
    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    _chunks_cache = chunks
    _chunks_mtime = mtime
    return chunks


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _get_bm25(chunks: list):
    """Return a cached BM25Okapi index, rebuilding only when chunks.jsonl changes."""
    global _bm25, _bm25_mtime
    try:
        mtime = CHUNKS_FILE.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    if _bm25 is None or mtime != _bm25_mtime:
        from rank_bm25 import BM25Okapi
        _bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        _bm25_mtime = mtime
    return _bm25


def bm25_search(query: str, chunks: list, top_k: int) -> list:
    bm25 = _get_bm25(chunks)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i], scores[i]) for i in ranked[:top_k]]


def semantic_search(query: str, top_k: int) -> list:
    import lancedb

    try:
        db = lancedb.connect(str(DB_DIR))
        try:
            table = db.open_table(TABLE_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Vector table '{TABLE_NAME}' not found — run embed.py first. ({e})"
            )
        model = load_embed_model()
        vec = model.encode(query).tolist()
        results = table.search(vec).limit(top_k).to_list()
        return [(r, r.get("_distance", 0)) for r in results]
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  WARN: semantic_search failed: {e}")
        return []


def rrf_fuse(
    bm25_results: list,
    semantic_results: list,
    top_k: int,
) -> list:
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    def chunk_key(chunk: dict) -> str:
        return f"{chunk['note_id']}::{chunk['chunk_index']}"

    for rank, (chunk, _) in enumerate(bm25_results):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        meta[key] = chunk

    for rank, (chunk, _) in enumerate(semantic_results):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        meta[key] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(meta[k], score) for k, score in ranked[:top_k]]


def retrieve(query: str, top_k: int = 5, expand_graph: bool = True) -> list:
    chunks = load_all_chunks()
    if not chunks:
        return []

    # Run both searches with 2x candidates so fusion has room to work
    k = top_k * 2
    bm25_res = bm25_search(query, chunks, k)
    semantic_res = semantic_search(query, k)
    results = rrf_fuse(bm25_res, semantic_res, top_k)

    if expand_graph:
        results = graph_expand(results, top_k, chunks)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(args.query, top_k=args.top_k)

    if not results:
        print("No results.")
        return

    print(f'\nTop {len(results)} results for: "{args.query}"\n')
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(
            f"[{i}] {chunk['title']}  (chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']})"
        )
        print(f"    score={score:.4f}  ~{chunk['tokens_approx']} tokens")
        print(
            f"    {chunk['text'][:300].strip()}{'...' if len(chunk['text']) > 300 else ''}"
        )
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
