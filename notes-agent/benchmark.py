#!/usr/bin/env python3
"""
Latency profiler for the notes-agent retrieval pipeline.

Measures wall-clock time for each stage independently and reports
median + p95 across N runs per query.

Stages measured:
  chunk_load    loading chunks.jsonl into memory
  bm25          BM25 keyword search
  embed_query   encoding the query with sentence-transformers
  vector_search LanceDB ANN search
  rrf_fuse      Reciprocal Rank Fusion
  graph_expand  1-hop graph neighbour expansion
  full          end-to-end retrieve() call (warm cache)

Usage:
    python3 profile.py                      # default queries, 5 runs each
    python3 profile.py --runs 10            # more runs for tighter numbers
    python3 profile.py --cold               # include cold-start (first load)
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR = Path(__file__).parent

DEFAULT_QUERIES = [
    "confidence and self belief",
    "what are my project ideas?",
    "what do I write about research?",
    "stressors and anxiety",
    "what are my daily habits?",
]

COL_W = 10


# ── Timing helpers ────────────────────────────────────────────────────────


class Timer:
    def __init__(self):
        self._t = None
        self.elapsed_ms = None

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._t) * 1000


# ── Stage-level benchmarks ────────────────────────────────────────────────


def bench_chunk_load(runs: int) -> list[float]:
    """Cold chunk load (clears cache each run)."""
    import retrieve as r
    times = []
    for _ in range(runs):
        r._chunks_cache = []
        r._chunks_mtime = 0.0
        with Timer() as t:
            r.load_all_chunks()
        times.append(t.elapsed_ms)
    return times


def bench_bm25(query: str, runs: int) -> list[float]:
    from retrieve import load_all_chunks, bm25_search
    chunks = load_all_chunks()
    times = []
    for _ in range(runs):
        with Timer() as t:
            bm25_search(query, chunks, top_k=10)
        times.append(t.elapsed_ms)
    return times


def bench_embed_query(query: str, runs: int) -> list[float]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    times = []
    for _ in range(runs):
        with Timer() as t:
            model.encode(query)
        times.append(t.elapsed_ms)
    return times


def bench_vector_search(query: str, runs: int) -> list[float]:
    import lancedb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(query).tolist()
    db = lancedb.connect(str(AGENT_DIR / "lancedb"))
    table = db.open_table("notes_chunks")
    times = []
    for _ in range(runs):
        with Timer() as t:
            table.search(vec).limit(10).to_list()
        times.append(t.elapsed_ms)
    return times


def bench_rrf(query: str, runs: int) -> list[float]:
    from retrieve import load_all_chunks, bm25_search, semantic_search, rrf_fuse
    chunks = load_all_chunks()
    bm25_res = bm25_search(query, chunks, top_k=10)
    sem_res  = semantic_search(query, top_k=10)
    times = []
    for _ in range(runs):
        with Timer() as t:
            rrf_fuse(bm25_res, sem_res, top_k=5)
        times.append(t.elapsed_ms)
    return times


def bench_graph_expand(query: str, runs: int) -> list[float]:
    from retrieve import load_all_chunks, retrieve, graph_expand
    chunks = load_all_chunks()
    results = retrieve(query, top_k=5, expand_graph=False)
    times = []
    for _ in range(runs):
        with Timer() as t:
            graph_expand(results, top_k=5, all_chunks=chunks)
        times.append(t.elapsed_ms)
    return times


def bench_full_retrieve(query: str, runs: int) -> list[float]:
    """End-to-end retrieve() with warm cache."""
    from retrieve import retrieve
    retrieve(query, top_k=5)  # warm-up
    times = []
    for _ in range(runs):
        with Timer() as t:
            retrieve(query, top_k=5)
        times.append(t.elapsed_ms)
    return times


# ── Reporting ─────────────────────────────────────────────────────────────


def fmt(ms: float) -> str:
    return f"{ms:.1f}ms"


def summarise(samples: list[float]) -> tuple[float, float]:
    """Returns (median, p95)."""
    s = sorted(samples)
    med = statistics.median(s)
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    return med, p95


def print_row(label: str, samples: list[float], note: str = ""):
    med, p95 = summarise(samples)
    note_str = f"  {note}" if note else ""
    print(f"  {label:<18}  median={fmt(med):<10}  p95={fmt(p95)}{note_str}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=5,  help="Runs per query (default 5)")
    parser.add_argument("--cold",  action="store_true",  help="Include cold-start measurement")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    runs    = args.runs

    print(f"\nLatency profile — {len(queries)} queries × {runs} runs each")
    print("=" * 60)

    # ── Cold start ────────────────────────────────────────────────
    if args.cold:
        print("\nCold start (first chunk load):")
        cold_times = bench_chunk_load(runs=3)
        print_row("chunk_load", cold_times, note="includes disk read + parse")

    # ── Per-stage (warm cache) ────────────────────────────────────
    print("\nPer-stage latency (warm cache):")

    all_bm25, all_embed, all_vec, all_rrf, all_graph, all_full = [], [], [], [], [], []

    for q in queries:
        all_bm25  .extend(bench_bm25(q, runs))
        all_embed .extend(bench_embed_query(q, runs))
        all_vec   .extend(bench_vector_search(q, runs))
        all_rrf   .extend(bench_rrf(q, runs))
        all_graph .extend(bench_graph_expand(q, runs))
        all_full  .extend(bench_full_retrieve(q, runs))

    print_row("bm25",          all_bm25,  note="keyword search over chunks.jsonl")
    print_row("embed_query",   all_embed, note="encode query → 384-dim vector")
    print_row("vector_search", all_vec,   note="LanceDB ANN search")
    print_row("rrf_fuse",      all_rrf,   note="merge BM25 + semantic rankings")
    print_row("graph_expand",  all_graph, note="1-hop neighbour expansion")
    print_row("full retrieve", all_full,  note="end-to-end (BM25+vec+RRF+graph)")

    # ── Breakdown ─────────────────────────────────────────────────
    full_med = summarise(all_full)[0]
    vec_med  = summarise(all_vec)[0]
    emb_med  = summarise(all_embed)[0]
    bm25_med = summarise(all_bm25)[0]

    print(f"\nBottleneck breakdown (% of full retrieve median):")
    for label, t in [
        ("embed_query",   emb_med),
        ("vector_search", vec_med),
        ("bm25",          bm25_med),
    ]:
        pct = (t / full_med * 100) if full_med > 0 else 0
        print(f"  {label:<18}  {pct:.0f}%")

    # ── Save ──────────────────────────────────────────────────────
    results = {
        "runs_per_query": runs,
        "queries":        len(queries),
        "stages": {
            "bm25":          {"median_ms": round(summarise(all_bm25)[0], 2),  "p95_ms": round(summarise(all_bm25)[1], 2)},
            "embed_query":   {"median_ms": round(summarise(all_embed)[0], 2), "p95_ms": round(summarise(all_embed)[1], 2)},
            "vector_search": {"median_ms": round(summarise(all_vec)[0], 2),   "p95_ms": round(summarise(all_vec)[1], 2)},
            "rrf_fuse":      {"median_ms": round(summarise(all_rrf)[0], 2),   "p95_ms": round(summarise(all_rrf)[1], 2)},
            "graph_expand":  {"median_ms": round(summarise(all_graph)[0], 2), "p95_ms": round(summarise(all_graph)[1], 2)},
            "full_retrieve": {"median_ms": round(summarise(all_full)[0], 2),  "p95_ms": round(summarise(all_full)[1], 2)},
        }
    }
    out = AGENT_DIR / "profile_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
