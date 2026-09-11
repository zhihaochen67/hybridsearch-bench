"""Deterministic tests for dense retrieval (Phase 2).

The real sentence-transformers model is never loaded here: a small fake
encoder with hand-picked synthetic vectors is injected instead, so tests run
offline, instantly, and are fully deterministic. The FAISS index itself is
real: only the embeddings are stubbed.
"""

import sys

import numpy as np
import pytest

from hybridsearch.retrieval.dense import DenseRetriever, l2_normalize


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


# Unnormalized on purpose, so the tests also exercise L2 normalization.
VECTORS = {
    "alpha": [2, 0, 0],      # -> [1, 0, 0]
    "beta": [0, 4, 0],       # -> [0, 1, 0]
    "gamma": [0, 0, 6],      # -> [0, 0, 1]
    "delta": [2, 0, 0],      # -> [1, 0, 0] (ties with "alpha")
    "q-x": [2, 0, 0],        # -> [1, 0, 0]
    "q-y": [0, 4, 0],        # -> [0, 1, 0]
    "q-mixed": [3, 4, 0],    # -> [0.6, 0.8, 0]
}

CORPUS = [
    {"id": "A", "text": "alpha"},
    {"id": "B", "text": "beta"},
    {"id": "C", "text": "gamma"},
    {"id": "D", "text": "delta"},
]


@pytest.fixture
def encoder():
    return FakeEncoder(VECTORS)


@pytest.fixture
def retriever(encoder):
    return DenseRetriever(CORPUS, encoder=encoder)


# --- normalization ---


def test_l2_normalize_returns_unit_vectors():
    normalized = l2_normalize(np.array([[3.0, 4.0]]))

    assert normalized == pytest.approx(np.array([[0.6, 0.8]]))
    assert np.linalg.norm(normalized, axis=1) == pytest.approx([1.0])


def test_l2_normalize_multiple_rows():
    normalized = l2_normalize(np.array([[2.0, 0.0], [0.0, 4.0]]))

    assert normalized == pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_l2_normalize_zero_vector_stays_zero():
    normalized = l2_normalize(np.array([[0.0, 0.0, 0.0]]))

    assert normalized == pytest.approx(np.array([[0.0, 0.0, 0.0]]))
    assert not np.isnan(normalized).any()


# --- index building and no re-encoding ---


def test_corpus_encoded_once_at_build_time(encoder, retriever):
    assert encoder.calls == [["alpha", "beta", "gamma", "delta"]]

    retriever.search("q-x")
    retriever.search("q-y")
    retriever.search("q-mixed")

    # One corpus encode plus exactly one single-text encode per query.
    corpus_calls = [call for call in encoder.calls if len(call) == len(CORPUS)]
    assert corpus_calls == [["alpha", "beta", "gamma", "delta"]]
    assert encoder.calls == [
        ["alpha", "beta", "gamma", "delta"],
        ["q-x"],
        ["q-y"],
        ["q-mixed"],
    ]


def test_index_properties(retriever):
    assert retriever.index.ntotal == 4
    assert retriever.index.d == 3
    assert retriever.document_embeddings.shape == (4, 3)

    norms = np.linalg.norm(retriever.document_embeddings, axis=1)
    assert norms == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_no_model_loaded_when_encoder_injected(encoder):
    DenseRetriever(CORPUS, encoder=encoder)

    assert "sentence_transformers" not in sys.modules


# --- retrieval behavior ---


def test_expected_ranking_with_synthetic_embeddings(retriever):
    results = retriever.search("q-x", top_k=4)

    assert [result["id"] for result in results] == ["A", "D", "B", "C"]
    assert [result["score"] for result in results] == pytest.approx(
        [1.0, 1.0, 0.0, 0.0]
    )


def test_manual_cosine_similarity(retriever):
    # q-mixed normalizes to [0.6, 0.8, 0]:
    # beta ([0, 1, 0]) -> 0.8, alpha/delta ([1, 0, 0]) -> 0.6, gamma -> 0.0.
    results = retriever.search("q-mixed", top_k=4)

    assert [result["id"] for result in results] == ["B", "A", "D", "C"]
    assert [result["score"] for result in results] == pytest.approx(
        [0.8, 0.6, 0.6, 0.0]
    )


def test_ties_broken_by_document_id(retriever):
    results = retriever.search("q-x", top_k=2)

    # "A" and "D" have identical embeddings; the id tie-break puts A first.
    assert [result["id"] for result in results] == ["A", "D"]
    assert results[0]["score"] == results[1]["score"] == pytest.approx(1.0)


def test_flat_tie_breaking_is_global_across_top_k_boundary():
    corpus = [
        {"id": "Z", "text": "doc-z"},
        {"id": "Y", "text": "doc-y"},
        {"id": "A", "text": "doc-a"},
        {"id": "B", "text": "doc-b"},
        {"id": "C", "text": "doc-c"},
    ]
    vectors = {document["text"]: [1.0, 0.0] for document in corpus}
    vectors["query"] = [1.0, 0.0]
    retriever = DenseRetriever(corpus, encoder=FakeEncoder(vectors), index_type="flat")

    results = retriever.search("query", top_k=2)

    assert [result["id"] for result in results] == ["A", "B"]


def test_result_contains_id_text_score(retriever):
    result = retriever.search("q-y", top_k=1)[0]

    assert set(result) == {"id", "text", "score"}
    assert result["id"] == "B"
    assert result["text"] == "beta"
    assert result["score"] == pytest.approx(1.0)


def test_top_k(retriever):
    assert len(retriever.search("q-x", top_k=1)) == 1
    assert [result["id"] for result in retriever.search("q-x", top_k=2)] == ["A", "D"]

    # top_k beyond the corpus size returns every document.
    assert len(retriever.search("q-x", top_k=100)) == len(CORPUS)


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_non_positive_top_k_raises(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.search("q-x", top_k=top_k)


@pytest.mark.parametrize("query", ["", "   ", "\t\n", None])
def test_empty_query_raises(retriever, query):
    with pytest.raises(ValueError):
        retriever.search(query)


def test_empty_corpus_raises(encoder):
    with pytest.raises(ValueError):
        DenseRetriever([], encoder=encoder)


# --- determinism and configuration ---


def test_deterministic_results(retriever):
    first = retriever.search("q-mixed", top_k=4)
    second = retriever.search("q-mixed", top_k=4)

    assert first == second

    rebuilt = DenseRetriever(CORPUS, encoder=FakeEncoder(VECTORS)).search(
        "q-mixed", top_k=4
    )
    assert first == rebuilt


def test_default_model_name(encoder):
    retriever = DenseRetriever(CORPUS, encoder=encoder)

    assert retriever.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_configurable_model_name(encoder):
    retriever = DenseRetriever(CORPUS, encoder=encoder, model_name="custom/model")

    assert retriever.model_name == "custom/model"
