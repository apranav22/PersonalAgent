#!/usr/bin/env python3
"""
Step 6: Build a note-level similarity graph.

For each note, averages its chunk vectors → one embedding per note,
then computes pairwise cosine similarity and keeps edges above SIM_THRESHOLD.

Supports incremental updates (same --changed / --deleted pattern as chunk.py
and embed.py) so only affected nodes are recomputed on each ingest.

Output: notes-agent/graph.json
  { note_id: [{note_id, weight}, ...], ... }

Usage:
    python3 graph.py             # update only notes not yet in graph
    python3 graph.py --all       # full rebuild
    python3 graph.py --changed p1200 p3967 --deleted p999
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

AGENT_DIR = Path(__file__).parent
DB_DIR = AGENT_DIR / "lancedb"
TABLE_NAME = "notes_chunks"
GRAPH_FILE = AGENT_DIR / "graph.json"

SIM_THRESHOLD = 0.40  # minimum cosine similarity to create an edge
MAX_NEIGHBORS = 10  # max edges stored per note


# ── Embedding helpers ─────────────────────────────────────────────────────


def load_note_embeddings() -> dict[str, np.ndarray]:
    """Pull all chunk vectors from lancedb, average per note → one vector per note."""
    import lancedb

    db = lancedb.connect(str(DB_DIR))
    if TABLE_NAME not in db.table_names():
        raise RuntimeError(f"Table '{TABLE_NAME}' not found — run embed.py first.")

    tbl = db.open_table(TABLE_NAME)
    arrow = tbl.to_arrow()
    note_ids = arrow.column("note_id").to_pylist()
    vectors = arrow.column("vector").to_pylist()

    note_vecs: dict[str, list[np.ndarray]] = {}
    for nid, vec in zip(note_ids, vectors):
        note_vecs.setdefault(nid, []).append(np.array(vec, dtype=np.float32))

    return {nid: np.mean(vecs, axis=0) for nid, vecs in note_vecs.items()}


# ── Graph construction ────────────────────────────────────────────────────


def build_edges(
    target_ids: list[str],
    all_embeddings: dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    """
    Compute edges for target_ids against every note in all_embeddings.

    Uses matrix multiplication for efficiency: normalise all vectors once,
    then each target note is a single dot-product against the full matrix.

    Returns { note_id: [{note_id, weight}, ...] }
    """
    all_ids = list(all_embeddings.keys())
    all_vecs = np.stack([all_embeddings[nid] for nid in all_ids])  # (N, D)

    # Pre-normalise corpus for cosine via dot product
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vecs = all_vecs / norms  # (N, D)

    edges: dict[str, list[dict]] = {}

    for nid in target_ids:
        if nid not in all_embeddings:
            continue

        vec = all_embeddings[nid]
        norm = np.linalg.norm(vec)
        if norm == 0:
            edges[nid] = []
            continue

        sims = norm_vecs @ (vec / norm)  # (N,) — all cosine sims at once
        order = np.argsort(sims)[::-1]  # descending

        neighbors: list[dict] = []
        for idx in order:
            candidate = all_ids[idx]
            if candidate == nid:
                continue
            sim = float(sims[idx])
            if sim < SIM_THRESHOLD:
                break
            neighbors.append({"note_id": candidate, "weight": round(sim, 4)})
            if len(neighbors) >= MAX_NEIGHBORS:
                break

        edges[nid] = neighbors

    return edges


# ── I/O ──────────────────────────────────────────────────────────────────


def load_graph() -> dict[str, list[dict]]:
    if GRAPH_FILE.exists():
        return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    return {}


def save_graph(graph: dict[str, list[dict]]) -> None:
    GRAPH_FILE.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", nargs="*", help="Note IDs to recompute edges for")
    parser.add_argument("--deleted", nargs="*", help="Note IDs to remove from graph")
    parser.add_argument("--all", action="store_true", help="Full graph rebuild")
    args = parser.parse_args()

    print("Loading note embeddings from lancedb...")
    all_embeddings = load_note_embeddings()
    print(f"  {len(all_embeddings)} notes with embeddings")

    graph = load_graph()

    # Remove deleted nodes
    for nid in args.deleted or []:
        if graph.pop(nid, None) is not None:
            print(f"  Removed node: {nid}")

    # Determine which notes need (re)computation
    if args.all or not GRAPH_FILE.exists():
        target_ids = list(all_embeddings.keys())
    elif args.changed:
        target_ids = [nid for nid in args.changed if nid in all_embeddings]
    else:
        # Default: notes not yet in the graph
        target_ids = [nid for nid in all_embeddings if nid not in graph]

    if not target_ids:
        print("Nothing to update.")
        save_graph(graph)
        return

    print(f"Building edges for {len(target_ids)} notes...")
    graph.update(build_edges(target_ids, all_embeddings))

    save_graph(graph)

    edge_count = sum(len(v) for v in graph.values())
    print(f"Graph saved → {GRAPH_FILE}")
    print(f"  {len(graph)} nodes, {edge_count} edges")


if __name__ == "__main__":
    main()
