"""Offline tests for SciFact normalization (Phase 6).

Only synthetic raw records are used: the Hugging Face loader is never
called, so pytest needs no network and no ``datasets`` import.
"""

import pytest

from hybridsearch.data.scifact import (
    SciFactDataset,
    normalize_corpus,
    normalize_qrels,
    normalize_queries,
)


RAW_CORPUS = [
    {"_id": "1", "title": "First title", "text": "First body"},
    {"_id": 42, "title": "", "text": "Only a body"},
    {"_id": "3", "text": "No title field"},
]


def test_corpus_normalization_joins_title_and_text():
    corpus = normalize_corpus(RAW_CORPUS)

    assert corpus == [
        {"id": "1", "text": "First title First body"},
        {"id": "42", "text": "Only a body"},
        {"id": "3", "text": "No title field"},
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


def test_corpus_empty_text_raises():
    with pytest.raises(ValueError):
        normalize_corpus([{"_id": "1", "title": "", "text": "   "}])


def test_query_normalization_and_string_ids():
    queries = normalize_queries(
        [
            {"_id": "q1", "text": "question one"},
            {"_id": 7, "text": "question two"},
        ]
    )

    assert queries == {"q1": "question one", "7": "question two"}
    assert all(isinstance(query_id, str) for query_id in queries)


def test_query_duplicate_ids_raise():
    with pytest.raises(ValueError):
        normalize_queries(
            [
                {"_id": "q1", "text": "a"},
                {"_id": "q1", "text": "b"},
            ]
        )


def test_query_missing_id_raises():
    with pytest.raises(ValueError):
        normalize_queries([{"text": "no id"}])


def test_query_blank_text_raises():
    with pytest.raises(ValueError):
        normalize_queries([{"_id": "q1", "text": "   "}])


def test_qrel_normalization_structure_and_string_ids():
    qrels = normalize_qrels(
        [
            {"query-id": "q1", "corpus-id": "c1", "score": 2},
            {"query-id": "q1", "corpus-id": 9, "score": 1},
            {"query-id": "q2", "corpus-id": "c2", "score": 0},
        ]
    )

    assert qrels == {"q1": {"c1": 2, "9": 1}, "q2": {"c2": 0}}
    assert all(
        isinstance(query_id, str) and isinstance(doc_id, str)
        for query_id, entries in qrels.items()
        for doc_id in entries
    )


def test_qrel_unknown_document_id_raises():
    with pytest.raises(ValueError, match="document"):
        normalize_qrels(
            [{"query-id": "q1", "corpus-id": "ghost", "score": 1}],
            corpus_ids={"c1"},
            query_ids={"q1"},
        )


def test_qrel_unknown_query_id_raises():
    with pytest.raises(ValueError, match="query"):
        normalize_qrels(
            [{"query-id": "ghost", "corpus-id": "c1", "score": 1}],
            corpus_ids={"c1"},
            query_ids={"q1"},
        )


def test_qrel_missing_score_column_raises():
    with pytest.raises(ValueError):
        normalize_qrels([{"query-id": "q1", "corpus-id": "c1"}])


def test_dataset_helpers():
    dataset = SciFactDataset(
        corpus=[{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
        queries={"q1": "x"},
        qrels={"q1": {"1": 1}},
    )

    assert dataset.corpus_ids() == {"1", "2"}
    assert dataset.query_ids() == {"q1"}
