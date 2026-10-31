import math

import pytest

from eval import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall_at_k():
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, set(), 5) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / 1.0  # one relevant doc found at rank 2
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(expected)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
