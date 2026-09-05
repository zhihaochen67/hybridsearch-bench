"""Offline tests for FiQA normalization (Phase 13).

Only synthetic raw records are used: the Hugging Face loader is never
called, so pytest needs no network and no ``datasets`` import. The shared
normalization helpers live in ``hybridsearch.data.common`` and are
exercised here with FiQA-style records (empty titles, binary qrel
grades, the ``require_text=False`` opt-out).
"""

import pytest

from hybridsearch.data.common import (
    normalize_corpus,
    normalize_qrels,
    normalize_queries,
)
from hybridsearch.data.fiqa import FIQA_HF_PATH, FiQADataset


RAW_CORPUS = [
    {"_id": "3", "title": "", "text": "First answer body"},
    {"_id": 42, "title": "A title", "text": "Body with a title"},
    {"_id": "7", "text": "No title field"},
]


# --- corpus normalization / title-text handling ---


def test_corpus_normalization_handles_empty_titles():
    corpus = normalize_corpus(RAW_CORPUS)

    assert corpus == [
        {"id": "3", "text": "First answer body"},
        {"id": "42", "text": "A title Body with a title"},
        {"id": "7", "text": "No title field"},
    ]


def test_corpus_ids_preserved_as_strings():
    corpus = normalize_corpus(RAW_CORPUS)

    assert all(isinstance(document["id"], str) for document in corpus)
    assert corpus[1]["id"] == "42"


def test_corpus_duplicate_ids_raise():
    records = [
        {"_id": "1", "title": "", "text": "a"},
        {"_id": "1", "title": "", "text": "b"},
    ]

    with pytest.raises(ValueError):
        normalize_corpus(records)


def test_corpus_missing_id_raises():
    with pytest.raises(ValueError):
        normalize_corpus([{"title": "", "text": "body"}])


def test_empty_text_kept_when_require_text_is_false():
    # FiQA's published corpus really contains empty-text posts; the FiQA
    # loader keeps them as ``{"id": ..., "text": ""}`` instead of failing.
    corpus = normalize_corpus(
        [{"_id": "9", "title": "", "text": "   "}], require_text=False
    )

    assert corpus == [{"id": "9", "text": ""}]


def test_empty_text_raises_by_default():
    with pytest.raises(ValueError):
        normalize_corpus([{"_id": "9", "title": "", "text": "   "}])


# --- queries ---


def test_query_normalization_and_string_ids():
    queries = normalize_queries(
        [
            {"_id": "0", "text": "What is a business expense?"},
            {"_id": 4, "text": "Car insurance deductible"},
        ]
    )

    assert queries == {
        "0": "What is a business expense?",
        "4": "Car insurance deductible",
    }
    assert all(isinstance(query_id, str) for query_id in queries)


def test_query_duplicate_ids_raise():
    with pytest.raises(ValueError):
        normalize_queries(
            [
                {"_id": "q1", "text": "a"},
                {"_id": "q1", "text": "b"},
            ]
        )


# --- qrels ---


def test_qrel_construction_with_binary_grades():
    qrels = normalize_qrels(
        [
            {"query-id": "8", "corpus-id": "566392", "score": 1},
            {"query-id": "8", "corpus-id": 65404, "score": 1},
            {"query-id": "15", "corpus-id": "325273", "score": 1},
        ]
    )

    assert qrels == {
        "8": {"566392": 1, "65404": 1},
        "15": {"325273": 1},
    }
    assert all(
        isinstance(query_id, str) and isinstance(doc_id, str)
        for query_id, entries in qrels.items()
        for doc_id in entries
    )


def test_qrel_unknown_document_id_raises():
    with pytest.raises(ValueError, match="document"):
        normalize_qrels(
            [{"query-id": "8", "corpus-id": "ghost", "score": 1}],
            corpus_ids={"566392"},
            query_ids={"8"},
        )


def test_qrel_unknown_query_id_raises():
    with pytest.raises(ValueError, match="query"):
        normalize_qrels(
            [{"query-id": "ghost", "corpus-id": "566392", "score": 1}],
            corpus_ids={"566392"},
            query_ids={"8"},
        )


def test_qrel_missing_score_column_raises():
    with pytest.raises(ValueError):
        normalize_qrels([{"query-id": "8", "corpus-id": "566392"}])


# --- dataset container ---


def test_fiqa_dataset_defaults_and_helpers():
    dataset = FiQADataset(
        corpus=[{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
        queries={"q1": "x"},
        qrels={"q1": {"1": 1}},
    )

    assert dataset.name == "fiqa"
    assert dataset.source == FIQA_HF_PATH
    assert dataset.corpus_ids() == {"1", "2"}
    assert dataset.query_ids() == {"q1"}