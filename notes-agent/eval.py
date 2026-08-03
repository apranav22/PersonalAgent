#!/usr/bin/env python3
"""
Eval harness for the notes retrieval pipeline.

Compares three retrieval strategies:
  - bm25:     keyword search only
  - semantic: vector search only
  - hybrid:   BM25 + semantic (RRF fusion)  ← our approach

Metrics computed per query, then averaged:
  - Precision@k  fraction of top-k results that are relevant
  - Recall@k     fraction of relevant notes found in top-k
  - MRR          1/rank of first relevant result
  - nDCG@k       normalized discounted cumulative gain

Query set format (eval_queries.json):
[
  {
    "query": "what did I write about confidence?",
    "relevant_note_ids": ["note_9", "note_22"]   ← ground truth
  },
  ...
]

Usage:
    python3 eval.py                        # run all strategies, print table
    python3 eval.py --strategy hybrid      # one strategy only
    python3 eval.py --k 3                  # evaluate at k=3
    python3 eval.py --queries my_qs.json   # custom query file
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR    = Path(__file__).parent
QUERIES_FILE = AGENT_DIR / "eval_queries.json"


# ── Retrieval strategies ───────────────────────────────────────────────────

def retrieve_bm25(query: str, top_k: int) -> list:
    from retrieve import load_all_chunks, bm25_search
    chunks = load_all_chunks()
    results = bm25_search(query, chunks, top_k)
    return [chunk["note_id"] for chunk, _ in results]


def retrieve_semantic(query: str, top_k: int) -> list:
    from retrieve import semantic_search
    results = semantic_search(query, top_k)
    return [chunk["note_id"] for chunk, _ in results]


def retrieve_hybrid(query: str, top_k: int) -> list:
    from retrieve import retrieve
    results = retrieve(query, top_k=top_k, expand_graph=False)
    return [chunk["note_id"] for chunk, _ in results]


def retrieve_hybrid_graph(query: str, top_k: int) -> list:
    from retrieve import retrieve
    results = retrieve(query, top_k=top_k, expand_graph=True)
    return [chunk["note_id"] for chunk, _ in results]


STRATEGIES = {
    "bm25":         retrieve_bm25,
    "semantic":     retrieve_semantic,
    "hybrid":       retrieve_hybrid,
    "hybrid+graph": retrieve_hybrid_graph,
}


# ── Metrics ────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for nid in retrieved_k if nid in relevant)
    return hits / k if k else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for nid in retrieved_k if nid in relevant)
    return hits / len(relevant)


def mrr(retrieved: list, relevant: set) -> float:
    for rank, nid in enumerate(retrieved, 1):
        if nid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    def dcg(items):
        return sum(
            (1 if nid in relevant else 0) / math.log2(i + 2)
            for i, nid in enumerate(items[:k])
        )
    ideal_relevant = list(relevant)[:k]
    ideal_padding  = ["_"] * max(0, k - len(ideal_relevant))
    ideal = ideal_relevant + ideal_padding
    idcg = dcg(ideal)
    return dcg(retrieved) / idcg if idcg > 0 else 0.0


def score_query(retrieved: list, relevant: set, k: int) -> dict:
    return {
        f"P@{k}":   precision_at_k(retrieved, relevant, k),
        f"R@{k}":   recall_at_k(retrieved, relevant, k),
        "MRR":      mrr(retrieved, relevant),
        f"nDCG@{k}": ndcg_at_k(retrieved, relevant, k),
    }


# ── Main eval loop ─────────────────────────────────────────────────────────

def run_eval(queries: list, strategy_name: str, k: int) -> dict:
    fn = STRATEGIES[strategy_name]
    all_scores = []

    for q in queries:
        query    = q["query"]
        relevant = set(q["relevant_note_ids"])
        try:
            retrieved = fn(query, top_k=k)
        except Exception as e:
            print(f"  ERROR on '{query}': {e}")
            continue
        scores = score_query(retrieved, relevant, k)
        all_scores.append(scores)

    if not all_scores:
        return {}

    # Average across queries
    keys = all_scores[0].keys()
    return {key: sum(s[key] for s in all_scores) / len(all_scores) for key in keys}


def print_results(results: dict, k: int) -> None:
    strategies = list(results.keys())
    metrics    = list(next(iter(results.values())).keys()) if results else []

    col_w = 12
    header = f"{'Strategy':<12}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for strategy, scores in results.items():
        row = f"{strategy:<12}" + "".join(f"{scores[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print("─" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=None,
                        help="Run one strategy only (default: all)")
    parser.add_argument("--k",        type=int, default=5,
                        help="Evaluate at top-k (default: 5)")
    parser.add_argument("--queries",  type=str, default=str(QUERIES_FILE),
                        help="Path to eval_queries.json")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Query file not found: {queries_path}")
        print("Create eval_queries.json with format:")
        print('[{"query": "...", "relevant_note_ids": ["note_1", "note_2"]}, ...]')
        sys.exit(1)

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries  |  k={args.k}")

    strategies_to_run = [args.strategy] if args.strategy else list(STRATEGIES)
    results = {}

    for name in strategies_to_run:
        print(f"Running {name}...")
        results[name] = run_eval(queries, name, k=args.k)

    print_results(results, args.k)

    # Save results
    out_path = AGENT_DIR / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
