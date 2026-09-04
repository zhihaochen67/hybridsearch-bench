"""Offline tests for the SciFact benchmark harness (Phase 6).

The real SciFact loader and models are never touched: a synthetic
``SciFactDataset`` and fake retrievers are injected, so pytest needs no
network access and runs instantly.
"""

import json

import pytest

from experiments.benchmark import (
    evaluate_retriever,
    format_table,
    run_benchmark,
    save_results,
    select_query_ids,
)
from hybridsearch.data.scifact import SciFactDataset
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics


DATASET = SciFactDataset(
    corpus=[
        {"id": "C1", "text": "alpha doc"},
        {"id": "C2", "text": "beta doc"},
        {"id": "C3", "text": "gamma doc"},
    ],
    queries={
        "q1": "query one",
        "q2": "query two",
        "q3": "query three",  # no qrels -> excluded from evaluation
    },
    qrels={
        "q1": {"C1": 2, "C2": 1},
        "q2": {"C3": 1},
    },
)


class FakeRetriever:
    """Deterministic stand-in for a retriever keyed by query text."""

    def __init__(self, results_by_query):
        self.results = results_by_query
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


FAKE_RESULTS_A = {
    "query one": [
        {"id": "C2", "text": "beta doc", "score": 9.0},
        {"id": "C1", "text": "alpha doc", "score": 5.0},
        {"id": "C3", "text": "gamma doc", "score": 1.0},
    ],
    "query two": [
        {"id": "C3", "text": "gamma doc", "score": 4.0},
        {"id": "C1", "text": "alpha doc", "score": 2.0},
    ],
}

FAKE_RESULTS_B = {
    "query one": [
        {"id": "C1", "text": "alpha doc", "score": 7.0},
        {"id": "C2", "text": "beta doc", "score": 6.0},
    ],
    "query two": [
        {"id": "C1", "text": "alpha doc", "score": 3.0},
        {"id": "C3", "text": "gamma doc", "score": 1.0},
    ],
}


def fake_factory():
    return {
        "Fake A": FakeRetriever(FAKE_RESULTS_A),
        "Fake B": FakeRetriever(FAKE_RESULTS_B),
    }


def test_select_query_ids_skips_queries_without_qrels():
    assert select_query_ids(DATASET.queries, DATASET.qrels) == ["q1", "q2"]


def test_select_query_ids_limit():
    assert select_query_ids(DATASET.queries, DATASET.qrels, max_queries=1) == [
        "q1"
    ]


def test_select_query_ids_empty_raises():
    with pytest.raises(ValueError):
        select_query_ids(DATASET.queries, {}, max_queries=1)

    with pytest.raises(ValueError):
        select_query_ids(DATASET.queries, DATASET.qrels, max_queries=0)


def test_evaluate_retriever_matches_manual_metrics():
    retriever = FakeRetriever(FAKE_RESULTS_A)

    means = evaluate_retriever(retriever, DATASET, ["q1", "q2"], k=2)
    expected = mean_metrics(
        [
            evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C3"], DATASET.qrels["q2"], k=2),
        ]
    )

    assert means == pytest.approx(expected)


def test_run_benchmark_aggregates_per_method():
    payload = run_benchmark(DATASET, k=2, retriever_factory=fake_factory)

    assert set(payload["results"]) == {"Fake A", "Fake B"}
    for means in payload["results"].values():
        assert set(means) == {"precision@2", "recall@2", "mrr@2", "ndcg@2"}

    assert payload["results"]["Fake A"] == pytest.approx(
        mean_metrics(
            [
                evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2),
                evaluate_query(["C3"], DATASET.qrels["q2"], k=2),
            ]
        )
    )
    assert payload["results"]["Fake B"] == pytest.approx(
        mean_metrics(
            [
                evaluate_query(["C1", "C2"], DATASET.qrels["q1"], k=2),
                evaluate_query(["C1", "C3"], DATASET.qrels["q2"], k=2),
            ]
        )
    )


def test_run_benchmark_metadata():
    payload = run_benchmark(
        DATASET,
        k=10,
        candidate_k=20,
        alpha=0.3,
        rrf_k=70.0,
        dense_model="fake/model",
        retriever_factory=fake_factory,
    )
    metadata = payload["metadata"]

    assert metadata["dataset"] == "scifact"
    assert metadata["k"] == 10
    assert metadata["candidate_k"] == 20
    assert metadata["alpha"] == 0.3
    assert metadata["rrf_k"] == 70.0
    assert metadata["dense_model"] == "fake/model"
    assert metadata["num_documents"] == 3
    assert metadata["num_queries"] == 2
    assert metadata["max_queries"] is None
    assert metadata["seed"] is None


def test_query_limiting_reduces_evaluation():
    payload = run_benchmark(
        DATASET, k=2, max_queries=1, retriever_factory=fake_factory
    )

    assert payload["metadata"]["num_queries"] == 1
    assert payload["metadata"]["max_queries"] == 1
    assert payload["results"]["Fake A"] == pytest.approx(
        mean_metrics([evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2)])
    )


def test_save_results_shape(tmp_path):
    payload = run_benchmark(DATASET, k=2, retriever_factory=fake_factory)
    path = tmp_path / "nested" / "benchmark.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert set(loaded["metadata"]) == {
        "dataset",
        "source",
        "dense_model",
        "k",
        "candidate_k",
        "alpha",
        "rrf_k",
        "num_documents",
        "num_queries",
        "max_queries",
        "seed",
    }
    assert loaded["metadata"]["num_queries"] == 2
    assert set(loaded["results"]) == {"Fake A", "Fake B"}
    assert set(loaded["results"]["Fake A"]) == {
        "precision@2",
        "recall@2",
        "mrr@2",
        "ndcg@2",
    }
    assert all(
        isinstance(value, float) for value in loaded["results"]["Fake A"].values()
    )


def test_format_table_contains_methods_and_metrics():
    payload = run_benchmark(DATASET, k=2, retriever_factory=fake_factory)

    table = format_table(payload)

    assert "Method" in table
    for name in ("Fake A", "Fake B"):
        assert name in table
    for column in ("Precision@2", "Recall@2", "MRR@2", "nDCG@2"):
        assert column in table


def test_benchmark_is_deterministic():
    first = run_benchmark(DATASET, k=2, retriever_factory=fake_factory)
    second = run_benchmark(DATASET, k=2, retriever_factory=fake_factory)

    assert first == second
