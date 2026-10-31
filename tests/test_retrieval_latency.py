"""End-to-end retrieval on a small fixture corpus with the real embedding model.

Builds the index the same way the ingest pipeline does (chunk_note ->
all-MiniLM-L6-v2 embeddings -> LanceDB, plus the note graph), then checks that
hybrid retrieval returns the right note and that warm retrieval latency stays
under a generous budget. Downloads the ~90MB model on first run (cached in CI).
"""

import json
import statistics
import time

import numpy as np
import pytest

import embed
import graph
from chunk import chunk_note

# Generous budget for a shared CI runner (README reports ~17ms median on a laptop).
WARM_MEDIAN_BUDGET_MS = 250

TOPICS = {
    "Sourdough baking": "Feed the starter at a 1:1:1 ratio. Bulk ferment six hours, "
    "shape, cold retard overnight, bake in a dutch oven at 250C.",
    "Marathon training": "Long run on Sunday, tempo on Wednesday, easy miles otherwise. "
    "Taper three weeks before race day and practise gels.",
    "Japanese study": "Review Anki kanji deck daily, shadow NHK easy news, "
    "learn the te-form conjugation and keigo for work emails.",
    "Home network": "Router firmware update, separate VLAN for IoT devices, "
    "Pi-hole for DNS blocking, WireGuard for remote access.",
    "Tax filing": "Collect W-2 and 1099 forms, track charitable donations, "
    "check the standard deduction, file before April 15.",
    "Guitar practice": "Metronome at 80bpm for alternate picking, learn the CAGED "
    "system, transcribe one solo per week.",
}


@pytest.fixture(scope="module")
def model():
    import retrieve

    return retrieve.load_embed_model()


@pytest.fixture
def index(tmp_path, fresh_retrieve, model):
    import lancedb

    r = fresh_retrieve
    chunks = []
    for i, (title, body) in enumerate(TOPICS.items()):
        for j in range(10):  # 60 notes, long enough to split into multiple chunks
            nid = f"n{i}_{j}"
            md = tmp_path / f"{nid}.md"
            paras = [f"Entry {j}.{k}: {body}" for k in range(15)]
            md.write_text(f"# {title}\n\n" + "\n\n".join(paras), encoding="utf-8")
            chunks.extend(chunk_note(nid, f"{title} {j}", str(md), modified_at=0))

    r.CHUNKS_FILE.write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    db = lancedb.connect(str(r.DB_DIR))
    table = embed.get_or_create_table(db, model.get_sentence_embedding_dimension())
    assert embed.embed_and_upsert(table, chunks, model) == len(chunks)

    arrow = table.to_arrow()
    per_note: dict[str, list] = {}
    for nid, vec in zip(arrow.column("note_id").to_pylist(), arrow.column("vector").to_pylist()):
        per_note.setdefault(nid, []).append(np.array(vec))
    note_vecs = {k: np.mean(v, axis=0) for k, v in per_note.items()}
    r.GRAPH_FILE.write_text(json.dumps(graph.build_edges(list(note_vecs), note_vecs)))
    return r, len(chunks)


def test_hybrid_retrieval_finds_relevant_note(index):
    r, _ = index
    for query, title in [
        ("how long should the dough proof overnight", "Sourdough baking"),
        ("taper before race day", "Marathon training"),
        ("DNS ad blocking on my wifi", "Home network"),
    ]:
        results = r.retrieve(query, top_k=5)
        assert results, query
        assert results[0][0]["title"].startswith(title), (query, results[0][0]["title"])


def test_warm_retrieval_latency_budget(index):
    r, n_chunks = index
    queries = [
        "dough fermentation schedule", "weekly long run plan", "kanji review habit",
        "remote access vpn setup", "charitable donation receipts", "alternate picking speed",
    ]
    r.retrieve(queries[0], top_k=5)  # warm-up: model + BM25 index + LanceDB handle

    samples = []
    for _ in range(5):
        for q in queries:
            t0 = time.perf_counter()
            r.retrieve(q, top_k=5)
            samples.append((time.perf_counter() - t0) * 1000)

    median = statistics.median(samples)
    p95 = sorted(samples)[int(0.95 * len(samples)) - 1]
    print(f"\nwarm retrieve over {n_chunks} chunks: median={median:.1f}ms "
          f"p95={p95:.1f}ms (n={len(samples)})")
    assert median < WARM_MEDIAN_BUDGET_MS
