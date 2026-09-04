"""Deterministic tests for the hybrid retriever (Phase 3).

Stub retrievers inject fixed, hand-computed retrieval outputs, so no model
is ever loaded and nothing touches the network. One integration test pairs
a real ``BM25`` with a real ``DenseRetriever`` driven by an injected fake
encoder, to prove the dense corpus is never re-encoded.
"""

import sys

import numpy as np
import pytest

from hybridsearch.data.toy import TOY_CORPUS
from hybridsearch.retrieval.bm25 import BM25
from hybridsearch.retrieval.dense import DenseRetriever
from hybridsearch.retrieval.hybrid import HybridRetriever


# --- stub scenario ---------------------------------------------------------
#
# Union of "q" candidates: A B C D E.  F is never retrieved.
# Sparse order: A B C D          (scores 9.0, 6.0, 3.0, 0.5)
# Dense order:  D C B E          (scores 0.99, 0.85, 0.80, 0.40)
# Weighted, alpha=0.5 -> B, C, A, D (tie, id), E.
# RRF, k=60          -> D, B, C (tie, id), A, E.

CORPUS = [
    {"id": "A", "text": "alpha text"},
    {"id": "B", "text": "beta text"},
    {"id": "C", "text": "gamma text"},
    {"id": "D", "text": "delta text"},
    {"id": "E", "text": "epsilon text"},
    {"id": "F", "text": "zeta text"},
]

TEXTS = {document["id"]: document["text"] for document in CORPUS}

SPARSE_RESULTS = {
    "q": [
        {"id": "A", "text": TEXTS["A"], "score": 9.0},
        {"id": "B", "text": TEXTS["B"], "score": 6.0},
        {"id": "C", "text": TEXTS["C"], "score": 3.0},
        {"id": "D", "text": TEXTS["D"], "score": 0.5},
    ],
}

DENSE_RESULTS = {
    "q": [
        {"id": "D", "text": TEXTS["D"], "score": 0.99},
        {"id": "C", "text": TEXTS["C"], "score": 0.85},
        {"id": "B", "text": TEXTS["B"], "score": 0.80},
        {"id": "E", "text": TEXTS["E"], "score": 0.40},
    ],
}


class StubRetriever:
    """Deterministic stand-in for a retriever that records its calls."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


@pytest.fixture
def sparse():
    return StubRetriever(SPARSE_RESULTS)


@pytest.fixture
def dense():
    return StubRetriever(DENSE_RESULTS)


@pytest.fixture
def hybrid(sparse, dense):
    return HybridRetriever(sparse, dense, method="weighted", alpha=0.5)


@pytest.fixture
def rrf(sparse, dense):
    return HybridRetriever(sparse, dense, method="rrf", k=60)


# --- weighted fusion through the hybrid retriever ---


def test_weighted_hybrid_ordering(hybrid):
    results = hybrid.search("q", top_k=5)

    assert [result["id"] for result in results] == ["B", "C", "A", "D", "E"]


def test_weighted_hybrid_scores(hybrid):
    results = hybrid.search("q", top_k=5)
    by_id = {result["id"]: result for result in results}

    # norm sparse: A=1, B=5.5/8.5, C=2.5/8.5, D=0
    # norm dense:  D=1, C=0.45/0.59, B=0.40/0.59, E=0
    assert by_id["B"]["score"] == pytest.approx(
        0.5 * (5.5 / 8.5) + 0.5 * (0.40 / 0.59)
    )
    assert by_id["C"]["score"] == pytest.approx(
        0.5 * (2.5 / 8.5) + 0.5 * (0.45 / 0.59)
    )
    assert by_id["A"]["score"] == pytest.approx(0.5)
    assert by_id["D"]["score"] == pytest.approx(0.5)
    assert by_id["E"]["score"] == pytest.approx(0.0)


def test_weighted_hybrid_diagnostics_and_missing_contributions(hybrid):
    results = hybrid.search("q", top_k=5)
    by_id = {result["id"]: result for result in results}

    # "A" comes only from the sparse retriever.
    assert by_id["A"]["bm25_score"] == 9.0
    assert by_id["A"]["dense_score"] == 0.0
    assert by_id["A"]["normalized_bm25_score"] == pytest.approx(1.0)
    assert by_id["A"]["normalized_dense_score"] == 0.0

    # "E" comes only from the dense retriever.
    assert by_id["E"]["bm25_score"] == 0.0
    assert by_id["E"]["dense_score"] == pytest.approx(0.40)
    assert by_id["E"]["normalized_bm25_score"] == 0.0
    assert by_id["E"]["normalized_dense_score"] == 0.0

    assert by_id["B"]["text"] == "beta text"
    assert by_id["E"]["text"] == "epsilon text"


def test_weighted_result_fields(hybrid):
    result = hybrid.search("q", top_k=1)[0]

    assert set(result) == {
        "id",
        "text",
        "score",
        "bm25_score",
        "dense_score",
        "normalized_bm25_score",
        "normalized_dense_score",
    }


def test_alpha_one_is_sparse_only_ordering(sparse, dense):
    retriever = HybridRetriever(sparse, dense, method="weighted", alpha=1.0)

    results = retriever.search("q", top_k=5)

    assert [result["id"] for result in results] == ["A", "B", "C", "D", "E"]
    by_id = {result["id"]: result for result in results}
    assert by_id["A"]["score"] == pytest.approx(1.0)
    assert by_id["B"]["score"] == pytest.approx(5.5 / 8.5)
    assert by_id["E"]["score"] == pytest.approx(0.0)


def test_alpha_zero_is_dense_only_ordering(sparse, dense):
    retriever = HybridRetriever(sparse, dense, method="weighted", alpha=0.0)

    results = retriever.search("q", top_k=5)

    assert [result["id"] for result in results] == ["D", "C", "B", "A", "E"]
    by_id = {result["id"]: result for result in results}
    assert by_id["D"]["score"] == pytest.approx(1.0)
    assert by_id["C"]["score"] == pytest.approx(0.45 / 0.59)
    assert by_id["A"]["score"] == pytest.approx(0.0)


# --- RRF through the hybrid retriever ---


def test_rrf_hybrid_ordering_and_ranks(rrf):
    results = rrf.search("q", top_k=5)
    by_id = {result["id"]: result for result in results}

    # D: 1/64 + 1/61 > B: 1/62 + 1/63 == C: 1/63 + 1/62 (tie, id) > A: 1/61 > E: 1/64.
    assert [result["id"] for result in results] == ["D", "B", "C", "A", "E"]

    assert by_id["D"]["score"] == pytest.approx(1 / 64 + 1 / 61)
    assert by_id["D"]["sparse_rank"] == 4
    assert by_id["D"]["dense_rank"] == 1
    assert by_id["B"]["score"] == pytest.approx(1 / 62 + 1 / 63)
    assert by_id["B"]["sparse_rank"] == 2
    assert by_id["B"]["dense_rank"] == 3
    assert by_id["A"]["score"] == pytest.approx(1 / 61)
    assert by_id["A"]["sparse_rank"] == 1
    assert by_id["A"]["dense_rank"] is None
    assert by_id["E"]["sparse_rank"] is None
    assert by_id["E"]["dense_rank"] == 4


def test_rrf_result_fields(rrf):
    result = rrf.search("q", top_k=1)[0]

    assert set(result) == {"id", "text", "score", "sparse_rank", "dense_rank"}


def test_rrf_uses_ranks_not_raw_scores(sparse):
    dense = StubRetriever(
        {
            "q": [
                {"id": "D", "text": TEXTS["D"], "score": 999.0},
                {"id": "C", "text": TEXTS["C"], "score": 500.0},
                {"id": "B", "text": TEXTS["B"], "score": 0.001},
                {"id": "E", "text": TEXTS["E"], "score": -50.0},
            ],
        }
    )
    rrf = HybridRetriever(sparse, dense, method="rrf", k=60)

    baseline = HybridRetriever(
        StubRetriever(SPARSE_RESULTS), StubRetriever(DENSE_RESULTS), method="rrf"
    ).search("q", top_k=5)

    # Same rank order in both retrievers => identical RRF output despite
    # completely different raw score scales.
    assert rrf.search("q", top_k=5) == baseline


def test_rrf_custom_k_changes_scores(sparse, dense):
    default_k = HybridRetriever(sparse, dense, method="rrf").search("q", top_k=5)
    small_k = HybridRetriever(sparse, dense, method="rrf", k=1).search("q", top_k=5)

    assert [r["id"] for r in default_k] == [r["id"] for r in small_k]
    assert default_k[0]["score"] != small_k[0]["score"]


def test_all_zero_sparse_list_contributes_no_rrf_rank_signal():
    sparse = StubRetriever(
        {
            "q": [
                {"id": "A", "text": TEXTS["A"], "score": 0.0},
                {"id": "B", "text": TEXTS["B"], "score": 0.0},
                {"id": "C", "text": TEXTS["C"], "score": 0.0},
                {"id": "D", "text": TEXTS["D"], "score": 0.0},
            ],
        }
    )
    dense = StubRetriever(DENSE_RESULTS)
    rrf = HybridRetriever(sparse, dense, method="rrf")

    results = rrf.search("q", top_k=4)

    # Pure dense ranking; the all-zero sparse list adds no rank signal.
    assert [result["id"] for result in results] == ["D", "C", "B", "E"]
    for result in results:
        assert result["sparse_rank"] is None
        assert result["score"] == pytest.approx(1 / (60 + result["dense_rank"]))


def test_car_repair_style_bm25_nonmatches_do_not_beat_dense_match():
    # Real BM25 over the toy corpus: "car repair" matches no term, so every
    # candidate scores exactly 0 and the list is ordered by document id
    # (D1 before D3). That tie-break order must not leak into RRF.
    sparse = BM25(TOY_CORPUS)
    dense = StubRetriever(
        {
            "car repair": [
                {"id": "D3", "text": "automobile maintenance", "score": 0.73},
                {"id": "D1", "text": "machine learning algorithms", "score": 0.24},
                {"id": "D2", "text": "neural network deep learning", "score": 0.18},
            ],
        }
    )
    rrf = HybridRetriever(sparse, dense, method="rrf")

    results = rrf.search("car repair", top_k=3)

    # The genuine dense match wins instead of BM25's id tie-break order.
    assert [result["id"] for result in results] == ["D3", "D1", "D2"]
    assert results[0]["dense_rank"] == 1
    assert results[0]["sparse_rank"] is None
    assert results[1]["sparse_rank"] is None
    assert results[2]["sparse_rank"] is None


# --- top_k, candidate pool, and validation ---


def test_top_k(hybrid):
    assert len(hybrid.search("q", top_k=2)) == 2
    assert [result["id"] for result in hybrid.search("q", top_k=2)] == ["B", "C"]

    # top_k beyond the union returns every candidate.
    assert len(hybrid.search("q", top_k=100)) == 5


def test_candidate_pool_limits_fusion():
    small_sparse = StubRetriever(SPARSE_RESULTS)
    small_dense = StubRetriever(DENSE_RESULTS)
    small_pool = HybridRetriever(
        small_sparse, small_dense, alpha=0.5, candidate_k=2
    )

    # Candidates: sparse [A, B] (norm 1, 0), dense [D, C] (norm 1, 0).
    # A and D tie at 0.5 and lead; B and C sit at 0.0.
    results = small_pool.search("q", top_k=2)

    assert [result["id"] for result in results] == ["A", "D"]
    assert small_sparse.calls == [("q", 2)]
    assert small_dense.calls == [("q", 2)]
    assert results[0]["score"] == pytest.approx(0.5)


def test_full_candidate_pool_changes_ranking():
    full_sparse = StubRetriever(SPARSE_RESULTS)
    full_dense = StubRetriever(DENSE_RESULTS)
    full_pool = HybridRetriever(
        full_sparse, full_dense, alpha=0.5, candidate_k=50
    )

    # With the full pool, B and C are normalized against all candidates and
    # overtake the single-retriever leaders A and D.
    results = full_pool.search("q", top_k=2)

    assert [result["id"] for result in results] == ["B", "C"]
    assert full_sparse.calls == [("q", 50)]
    assert full_dense.calls == [("q", 50)]


def test_candidate_k_smaller_than_top_k_is_raised(sparse, dense):
    hybrid = HybridRetriever(sparse, dense, method="weighted", candidate_k=1)

    results = hybrid.search("q", top_k=10)

    # Fetch size is raised to top_k, so the full union is still available.
    assert sparse.calls == [("q", 10)]
    assert dense.calls == [("q", 10)]
    assert len(results) == 5


def test_candidate_k_larger_than_top_k(sparse, dense):
    hybrid = HybridRetriever(sparse, dense, method="weighted", candidate_k=50)

    hybrid.search("q", top_k=3)

    assert sparse.calls == [("q", 50)]
    assert dense.calls == [("q", 50)]


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_non_positive_top_k_raises(hybrid, top_k):
    with pytest.raises(ValueError):
        hybrid.search("q", top_k=top_k)


@pytest.mark.parametrize("query", ["", "   ", "\t\n", None])
def test_empty_query_raises(hybrid, query):
    with pytest.raises(ValueError):
        hybrid.search(query)


@pytest.mark.parametrize("method", ["RRF", "weighted_score", "rrf ", "hybrid", None])
def test_invalid_method_raises(sparse, dense, method):
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense, method=method)


@pytest.mark.parametrize("alpha", [-0.1, 1.5, "0.5", None])
def test_invalid_alpha_raises(sparse, dense, alpha):
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense, method="weighted", alpha=alpha)


@pytest.mark.parametrize("k", [0, -60, "60", None])
def test_invalid_k_raises(sparse, dense, k):
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense, method="rrf", k=k)


@pytest.mark.parametrize("candidate_k", [0, -5])
def test_invalid_candidate_k_raises(sparse, dense, candidate_k):
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense, candidate_k=candidate_k)


# --- determinism, defaults, and dense corpus reuse ---


def test_deterministic_repeated_calls(hybrid, rrf):
    assert hybrid.search("q", top_k=5) == hybrid.search("q", top_k=5)
    assert rrf.search("q", top_k=5) == rrf.search("q", top_k=5)

    rebuilt = HybridRetriever(
        StubRetriever(SPARSE_RESULTS), StubRetriever(DENSE_RESULTS), method="weighted"
    )
    assert hybrid.search("q", top_k=5) == rebuilt.search("q", top_k=5)


def test_defaults(sparse, dense):
    retriever = HybridRetriever(sparse, dense)

    assert retriever.method == "weighted"
    assert retriever.alpha == 0.5
    assert retriever.k == 60.0
    assert retriever.candidate_k == 50


def test_weighted_and_rrf_both_work_over_same_candidates(sparse, dense):
    weighted = HybridRetriever(sparse, dense, method="weighted").search("q", top_k=5)
    rrf = HybridRetriever(sparse, dense, method="rrf").search("q", top_k=5)

    assert {result["id"] for result in weighted} == {result["id"] for result in rrf}


# --- real BM25 + real DenseRetriever (fake encoder) integration ------------


class FakeEncoder:
    """Deterministic stand-in for ``SentenceTransformer`` that records calls."""

    def __init__(self, vectors):
        self.vectors = {
            text: np.asarray(vector, dtype=np.float64)
            for text, vector in vectors.items()
        }
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=np.float64)


ML_CORPUS = [
    {"id": "T1", "text": "machine learning algorithms"},
    {"id": "T2", "text": "neural network deep learning"},
    {"id": "T3", "text": "automobile maintenance"},
    {"id": "T4", "text": "machine learning for image classification"},
]

ML_VECTORS = {
    "machine learning algorithms": [1.0, 0.0, 0.0, 0.0],
    "neural network deep learning": [0.0, 1.0, 0.0, 0.0],
    "automobile maintenance": [0.0, 0.0, 1.0, 0.0],
    "machine learning for image classification": [0.0, 0.0, 0.0, 1.0],
    "car repair": [0.0, 0.0, 1.0, 0.0],
}


def test_dense_corpus_is_encoded_once_and_never_reencoded():
    encoder = FakeEncoder(ML_VECTORS)
    sparse = BM25(ML_CORPUS)
    dense = DenseRetriever(ML_CORPUS, encoder=encoder)
    hybrid = HybridRetriever(sparse, dense, method="weighted", alpha=0.5)

    hybrid.search("car repair", top_k=10)
    hybrid.search("car repair", top_k=10)

    # BM25 gives every document 0 for "car repair" (no lexical match), so
    # all sparse contributions are 0 and the dense ranking decides: T3 first.
    results = hybrid.search("car repair", top_k=10)
    assert [result["id"] for result in results] == ["T3", "T1", "T2", "T4"]
    assert results[0]["score"] == pytest.approx(0.5)

    # Exactly one corpus-length encode (at build time) plus one per query.
    corpus_calls = [call for call in encoder.calls if len(call) == len(ML_CORPUS)]
    assert corpus_calls == [[document["text"] for document in ML_CORPUS]]
    assert encoder.calls == [
        [document["text"] for document in ML_CORPUS],
        ["car repair"],
        ["car repair"],
        ["car repair"],
    ]


def test_no_sentence_transformers_imported_by_hybrid_path():
    assert "sentence_transformers" not in sys.modules
