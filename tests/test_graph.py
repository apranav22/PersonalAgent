import numpy as np

import graph


def test_build_edges_threshold_self_exclusion_and_order():
    emb = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9, 0.1]),
        "c": np.array([0.6, 0.8]),
        "d": np.array([0.0, 1.0]),  # orthogonal to a -> below SIM_THRESHOLD
    }
    edges = graph.build_edges(["a"], emb)
    ids = [e["note_id"] for e in edges["a"]]
    assert ids == ["b", "c"]
    assert "a" not in ids and "d" not in ids
    cos_ab = 0.9 / np.linalg.norm([0.9, 0.1])
    assert edges["a"][0]["weight"] == round(float(cos_ab), 4)
    assert edges["a"][1]["weight"] == 0.6


def test_build_edges_caps_neighbours_and_handles_edge_cases():
    emb = {f"n{i}": np.array([1.0, 0.01 * i]) for i in range(15)}
    emb["zero"] = np.zeros(2)
    edges = graph.build_edges(["n0", "zero", "missing"], emb)
    assert len(edges["n0"]) == graph.MAX_NEIGHBORS
    assert edges["zero"] == []
    assert "missing" not in edges


def test_save_and_load_graph_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "GRAPH_FILE", tmp_path / "graph.json")
    assert graph.load_graph() == {}
    g = {"a": [{"note_id": "b", "weight": 0.5}]}
    graph.save_graph(g)
    assert graph.load_graph() == g
