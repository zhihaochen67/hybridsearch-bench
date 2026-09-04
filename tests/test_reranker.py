"""Deterministic tests for cross-encoder reranking (Phase 4).

The real cross-encoder is never loaded here: a fake scorer with
hand-picked scores is injected instead, so tests run offline, instantly,
and are fully deterministic.
"""

import sys

import pytest

from hybridsearch.ranking.reranker import CrossEncoderReranker
from hybridsearch.retrieval.hybrid import HybridRetriever


class FakeScorer:
    """Deterministic stand-in for ``CrossEncoder`` that records calls."""

    def __init__(self, scores):
        self.scores = {tuple(pair): float(score) for pair, score in scores.items()}
        self.calls = []

    def predict(self, pairs):
        self.calls.append([tuple(pair) for pair in pairs])
        return [self.scores[tuple(pair)] for pair in pairs]


QUERY = "car repair"

# Shaped like hybrid-retrieval output: retrieval diagnostics included.
CANDIDATES = [
    {
        "id": "C1",
        "text": "alpha",
        "score": 0.9,
        "bm25_score": 9.0,
        "dense_score": 0.4,
        "normalized_bm25_score": 1.0,
        "normalized_dense_score": 0.2,
    },
    {
        "id": "C2",
        "text": "beta",
        "score": 0.8,
        "bm25_score": 6.0,
        "dense_score": 0.6,
        "normalized_bm25_score": 0.6,
        "normalized_dense_score": 0.6,
    },
    {
        "id": "C3",
        "text": "gamma",
        "score": 0.7,
        "bm25_score": 0.0,
        "dense_score": 0.9,
        "normalized_bm25_score": 0.0,
        "normalized_dense_score": 1.0,
    },
    {
        "id": "C4",
        "text": "delta",
        "score": 0.1,
        "bm25_score": 1.0,
        "dense_score": 0.2,
        "normalized_bm25_score": 0.1,
        "normalized_dense_score": 0.0,
    },
]

# C3 > C1 == C2 (tie, id order) > C4.
FAKE_SCORES = {
    (QUERY, "alpha"): 2.1,
    (QUERY, "beta"): 2.1,
    (QUERY, "gamma"): 4.2,
    (QUERY, "delta"): 0.5,
}


@pytest.fixture
def scorer():
    return FakeScorer(FAKE_SCORES)


@pytest.fixture
def reranker(scorer):
    return CrossEncoderReranker(scorer=scorer)


# --- scoring mechanics ---


def test_scorer_receives_query_document_pairs(scorer, reranker):
    reranker.rerank(QUERY, CANDIDATES, top_k=4)

    assert scorer.calls == [
        [
            (QUERY, "alpha"),
            (QUERY, "beta"),
            (QUERY, "gamma"),
            (QUERY, "delta"),
        ]
    ]


def test_all_candidates_scored_in_one_batch(scorer, reranker):
    reranker.rerank(QUERY, CANDIDATES, top_k=4)

    # One predict call for the whole candidate list, never one-by-one.
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == len(CANDIDATES)


def test_scorer_reused_not_recreated(scorer, reranker):
    assert reranker._ensure_scorer() is scorer

    reranker.rerank(QUERY, CANDIDATES, top_k=4)
    reranker.rerank(QUERY, CANDIDATES, top_k=4)

    # The same injected scorer serves every call.
    assert len(scorer.calls) == 2


# --- ranking behavior ---


def test_expected_reranking_order(reranker):
    results = reranker.rerank(QUERY, CANDIDATES, top_k=4)

    assert [result["id"] for result in results] == ["C3", "C1", "C2", "C4"]


def test_reranker_scores_preserved(reranker):
    results = reranker.rerank(QUERY, CANDIDATES, top_k=4)
    by_id = {result["id"]: result for result in results}

    assert by_id["C3"]["reranker_score"] == pytest.approx(4.2)
    assert by_id["C1"]["reranker_score"] == pytest.approx(2.1)
    assert by_id["C2"]["reranker_score"] == pytest.approx(2.1)
    assert by_id["C4"]["reranker_score"] == pytest.approx(0.5)


def test_existing_candidate_fields_preserved(reranker):
    result = reranker.rerank(QUERY, CANDIDATES, top_k=4)[0]

    # "C3" originally carried all hybrid diagnostics.
    assert set(result) == {
        "id",
        "text",
        "score",
        "bm25_score",
        "dense_score",
        "normalized_bm25_score",
        "normalized_dense_score",
        "reranker_score",
    }
    assert result["text"] == "gamma"
    assert result["score"] == pytest.approx(0.7)
    assert result["bm25_score"] == 0.0
    assert result["normalized_dense_score"] == pytest.approx(1.0)


def test_ties_broken_by_document_id(reranker):
    results = reranker.rerank(QUERY, CANDIDATES, top_k=4)

    # C1 and C2 share score 2.1; ascending id puts C1 first.
    assert results[1]["id"] == "C1"
    assert results[2]["id"] == "C2"
    assert results[1]["reranker_score"] == pytest.approx(
        results[2]["reranker_score"]
    )


def test_top_k(reranker):
    results = reranker.rerank(QUERY, CANDIDATES, top_k=2)

    assert [result["id"] for result in results] == ["C3", "C1"]


def test_top_k_beyond_candidate_count_returns_all(reranker):
    results = reranker.rerank(QUERY, CANDIDATES, top_k=100)

    assert len(results) == len(CANDIDATES)


# --- validation ---


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_non_positive_top_k_raises(reranker, top_k):
    with pytest.raises(ValueError):
        reranker.rerank(QUERY, CANDIDATES, top_k=top_k)


def test_empty_candidates_raise(reranker):
    with pytest.raises(ValueError):
        reranker.rerank(QUERY, [])


@pytest.mark.parametrize("query", ["", "   ", "\t\n", None])
def test_invalid_query_raises(reranker, query):
    with pytest.raises(ValueError):
        reranker.rerank(query, CANDIDATES)


def test_candidate_without_id_or_text_raises(reranker):
    with pytest.raises(ValueError):
        reranker.rerank(QUERY, [{"id": "only-id"}])

    with pytest.raises(ValueError):
        reranker.rerank(QUERY, [{"text": "only-text"}])


def test_scorer_returning_wrong_number_of_scores_raises():
    class BrokenScorer:
        def predict(self, pairs):
            return [1.0]

    reranker = CrossEncoderReranker(scorer=BrokenScorer())

    with pytest.raises(ValueError):
        reranker.rerank(QUERY, CANDIDATES)


# --- determinism, model loading, and configuration ---


def test_repeated_calls_are_deterministic(reranker):
    first = reranker.rerank(QUERY, CANDIDATES, top_k=4)
    second = reranker.rerank(QUERY, CANDIDATES, top_k=4)

    assert first == second

    rebuilt = CrossEncoderReranker(scorer=FakeScorer(FAKE_SCORES))
    assert first == rebuilt.rerank(QUERY, CANDIDATES, top_k=4)


def test_injected_scorer_prevents_real_model_loading(reranker):
    reranker.rerank(QUERY, CANDIDATES, top_k=4)

    assert "sentence_transformers" not in sys.modules


def test_default_model_name(scorer):
    assert (
        CrossEncoderReranker(scorer=scorer).model_name
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def test_configurable_model_name(scorer):
    reranker = CrossEncoderReranker(model_name="custom/reranker", scorer=scorer)

    assert reranker.model_name == "custom/reranker"


# --- integration with hybrid retrieval ---


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=10):
        return self.results[query][:top_k]


def test_integration_with_hybrid_retriever_candidates():
    sparse = StubRetriever(
        {
            "q": [
                {"id": "A", "text": "alpha", "score": 9.0},
                {"id": "B", "text": "beta", "score": 6.0},
                {"id": "C", "text": "gamma", "score": 0.0},
            ],
        }
    )
    dense = StubRetriever(
        {
            "q": [
                {"id": "C", "text": "gamma", "score": 0.9},
                {"id": "B", "text": "beta", "score": 0.7},
                {"id": "A", "text": "alpha", "score": 0.1},
            ],
        }
    )
    hybrid = HybridRetriever(sparse, dense, method="weighted", alpha=0.5)
    reranker = CrossEncoderReranker(
        scorer=FakeScorer(
            {
                ("q", "alpha"): 1.0,
                ("q", "beta"): 3.0,
                ("q", "gamma"): 5.0,
            }
        )
    )

    # Composition: retrieval stage and ranking stage stay separate.
    candidates = hybrid.search("q", top_k=4)
    results = reranker.rerank("q", candidates, top_k=3)

    assert [result["id"] for result in results] == ["C", "B", "A"]
    assert results[0]["reranker_score"] == pytest.approx(5.0)

    # Hybrid diagnostics flow through the reranker untouched.
    assert set(results[0]) == {
        "id",
        "text",
        "score",
        "bm25_score",
        "dense_score",
        "normalized_bm25_score",
        "normalized_dense_score",
        "reranker_score",
    }
    assert results[0]["score"] == pytest.approx(0.5)
