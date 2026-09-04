import math

import pytest

from hybridsearch.data.toy import TOY_CORPUS
from hybridsearch.retrieval.bm25 import BM25


@pytest.fixture
def bm25():
    return BM25(TOY_CORPUS)


def test_document_frequencies(bm25):
    assert bm25.document_frequency("machine") == 2
    assert bm25.document_frequency("learning") == 3
    assert bm25.document_frequency("automobile") == 1
    assert bm25.document_frequency("not-in-corpus") == 0
    assert bm25.document_frequencies["neural"] == 1


def test_term_frequencies(bm25):
    assert bm25.term_frequency("machine", "D1") == 1
    assert bm25.term_frequency("machine", "D2") == 0
    assert bm25.term_frequency("not-in-corpus", "D1") == 0


def test_document_lengths_and_average(bm25):
    assert bm25.doc_lengths == {"D1": 3, "D2": 4, "D3": 2, "D4": 5}
    assert bm25.average_document_length == pytest.approx(3.5)


def test_idf(bm25):
    assert bm25.idf("machine") == pytest.approx(math.log(2.0))
    assert bm25.idf("learning") == pytest.approx(math.log(1.0 + 1.5 / 3.5))
    assert bm25.idf("automobile") == pytest.approx(math.log(1.0 + 3.5 / 1.5))
    assert bm25.idf("not-in-corpus") == pytest.approx(math.log(10.0))


def test_score_matches_reference_formula(bm25):
    # Hand-computed BM25 components for D1 with k1=1.5, b=0.75,
    # N=4, avgdl=3.5, query "machine learning".
    idf_machine = math.log(2.0)
    idf_learning = math.log(1.0 + 1.5 / 3.5)
    tf_component = 2.5 / (1.0 + 1.5 * (0.25 + 0.75 * (3.0 / 3.5)))

    expected = (idf_machine + idf_learning) * tf_component

    assert bm25.score("machine learning", "D1") == pytest.approx(expected)


def test_scores_non_negative(bm25):
    for document in TOY_CORPUS:
        assert bm25.score("machine learning", document["id"]) >= 0

    # A document with no matching term scores exactly zero.
    assert bm25.score("machine learning", "D3") == 0.0


def test_expected_ranking(bm25):
    results = bm25.search("machine learning", top_k=4)

    assert [result["id"] for result in results] == ["D1", "D4", "D2", "D3"]

    scores = [result["score"] for result in results]
    assert scores[0] > scores[1] > scores[2] > scores[3]

    assert results[0]["text"] == "machine learning algorithms"


def test_single_term_query(bm25):
    results = bm25.search("automobile", top_k=4)

    assert results[0]["id"] == "D3"
    assert results[0]["score"] > 0


def test_top_k(bm25):
    results = bm25.search("machine learning", top_k=2)

    assert [result["id"] for result in results] == ["D1", "D4"]

    # top_k beyond the corpus size returns every document.
    assert len(bm25.search("machine learning", top_k=10)) == len(TOY_CORPUS)

    assert len(bm25.search("machine learning", top_k=1)) == 1


def test_deterministic(bm25):
    first = bm25.search("machine learning", top_k=4)
    second = bm25.search("machine learning", top_k=4)

    assert first == second

    # Rebuilding the index from the same corpus gives identical results.
    rebuilt = BM25(TOY_CORPUS).search("machine learning", top_k=4)
    assert first == rebuilt


def test_configurable_parameters():
    default = BM25(TOY_CORPUS)
    other = BM25(TOY_CORPUS, k1=0.5, b=1.0)

    assert default.k1 == 1.5
    assert default.b == 0.75
    assert other.k1 == 0.5
    assert other.b == 1.0

    assert default.score("machine learning", "D1") != other.score("machine learning", "D1")


def test_empty_query(bm25):
    results = bm25.search("", top_k=4)

    assert [result["id"] for result in results] == ["D1", "D2", "D3", "D4"]
    assert all(result["score"] == 0.0 for result in results)