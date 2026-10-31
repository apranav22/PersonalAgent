import json

import pytest

from conftest import make_chunk, write_jsonl

CORPUS = [
    make_chunk("bread", 0, "Feed the sourdough starter daily; bake at 230C with steam."),
    make_chunk("gym", 0, "Leg day: squats, lunges, deadlifts. Track progressive overload."),
    make_chunk("books", 0, "Reading list: Dune, Neuromancer, The Left Hand of Darkness."),
    make_chunk("travel", 0, "Pack passport, adapters, and the sourdough recipe for Lisbon."),
    make_chunk("work", 0, "Quarterly planning: hiring, roadmap review, budget sign-off."),
    make_chunk("music", 0, "Practice scales on guitar, learn the Blackbird fingerpicking."),
    make_chunk("code", 0, "Refactor the parser, add unit tests, profile the hot loop."),
    make_chunk("garden", 0, "Plant tomatoes and basil; water twice a week in summer."),
]


def test_tokenize_lowercases_and_drops_punctuation(fresh_retrieve):
    assert fresh_retrieve.tokenize("Hello, World! It's 2026.") == [
        "hello", "world", "it", "s", "2026",
    ]


def test_rrf_fuse_scores_match_formula(fresh_retrieve):
    r = fresh_retrieve
    a, b, c, d = (make_chunk(n, 0, n) for n in "abcd")
    bm25 = [(a, 9.0), (b, 5.0), (c, 1.0)]
    sem = [(b, 0.1), (d, 0.2)]

    fused = r.rrf_fuse(bm25, sem, top_k=10)
    scores = {ch["note_id"]: s for ch, s in fused}

    k = r.RRF_K
    assert scores["b"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert scores["a"] == pytest.approx(1 / (k + 1))
    assert scores["d"] == pytest.approx(1 / (k + 2))
    assert scores["c"] == pytest.approx(1 / (k + 3))
    # A doc found by both retrievers beats the single top hit of either.
    assert [ch["note_id"] for ch, _ in fused] == ["b", "a", "d", "c"]


def test_rrf_fuse_dedupes_by_note_and_chunk_and_truncates(fresh_retrieve):
    r = fresh_retrieve
    x0, x1 = make_chunk("x", 0, "first"), make_chunk("x", 1, "second")
    fused = r.rrf_fuse([(x0, 1), (x1, 1)], [(x1, 1), (x0, 1)], top_k=1)
    assert len(fused) == 1
    assert fused[0][1] == pytest.approx(1 / (r.RRF_K + 1) + 1 / (r.RRF_K + 2))


def test_bm25_search_ranks_keyword_match_first(fresh_retrieve):
    r = fresh_retrieve
    write_jsonl(r.CHUNKS_FILE, CORPUS)
    chunks = r.load_all_chunks()
    assert len(chunks) == len(CORPUS)

    top = r.bm25_search("squats deadlifts", chunks, top_k=2)
    assert top[0][0]["note_id"] == "gym"
    both = {c["note_id"] for c, _ in r.bm25_search("sourdough", chunks, top_k=2)}
    assert both == {"bread", "travel"}


def test_load_all_chunks_skips_bad_lines_and_caches(fresh_retrieve):
    r = fresh_retrieve
    r.CHUNKS_FILE.write_text(json.dumps(CORPUS[0]) + "\n{broken\n", encoding="utf-8")
    first = r.load_all_chunks()
    assert [c["note_id"] for c in first] == ["bread"]
    assert r.load_all_chunks() is first  # same mtime -> cached object


def test_graph_expand_adds_discounted_neighbours(fresh_retrieve):
    r = fresh_retrieve
    r.GRAPH_FILE.write_text(json.dumps({
        "bread": [{"note_id": "travel", "weight": 0.8}, {"note_id": "gym", "weight": 0.5}],
    }))
    bread, travel, gym = CORPUS[0], CORPUS[3], CORPUS[1]
    out = r.graph_expand([(bread, 0.5)], top_k=2, all_chunks=CORPUS)

    # top_k // 2 == 1 neighbour, the stronger edge, scored direct * weight * discount
    assert [(c["note_id"], s) for c, s in out] == [
        ("bread", 0.5),
        ("travel", pytest.approx(0.5 * 0.8 * r.GRAPH_HOP_DISCOUNT)),
    ]
    assert gym not in [c for c, _ in out]
    assert travel in [c for c, _ in out]


def test_graph_expand_skips_notes_already_in_results(fresh_retrieve):
    r = fresh_retrieve
    r.GRAPH_FILE.write_text(json.dumps({"bread": [{"note_id": "travel", "weight": 0.9}]}))
    results = [(CORPUS[0], 0.5), (CORPUS[3], 0.4)]
    assert r.graph_expand(results, top_k=4, all_chunks=CORPUS) == results


def test_graph_expand_without_graph_is_noop(fresh_retrieve):
    results = [(CORPUS[0], 0.5)]
    assert fresh_retrieve.graph_expand(results, 5, CORPUS) == results


def test_retrieve_fuses_bm25_with_semantic_results(fresh_retrieve, monkeypatch):
    r = fresh_retrieve
    write_jsonl(r.CHUNKS_FILE, CORPUS)
    calls = {}

    def fake_semantic(query, top_k):
        calls["top_k"] = top_k
        return [(CORPUS[2], 0.1), (CORPUS[0], 0.2)]  # books, bread

    monkeypatch.setattr(r, "semantic_search", fake_semantic)
    out = r.retrieve("sourdough starter", top_k=2, expand_graph=False)

    assert calls["top_k"] == 4  # 2x candidates for fusion
    assert out[0][0]["note_id"] == "bread"  # top of BM25 and #2 semantic
    assert len(out) == 2


def test_retrieve_with_no_chunks_returns_empty(fresh_retrieve):
    assert fresh_retrieve.retrieve("anything") == []
