"""Offline tests for the reranker benchmark harness (Phase 8).

No SciFact download, no sentence-transformers, and no cross-encoder model
are touched: a synthetic ``SciFactDataset``, fake retrievers, and a fake
reranker drive every test deterministically.
"""

import json

import pytest

from experiments.benchmark import save_results
from experiments.reranker_benchmark import (
    evaluate_method,
    output_path_for,
    rerank_query,
    run_reranker_benchmark,
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
    """Deterministic stand-in keyed by query text."""

    def __init__(self, results_by_query):
        self.results = results_by_query
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


RETRIEVER_RESULTS = {
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

# Reranker order for "query one": C1 first (not the retrieval order C2 first).
RERANKER_SCORES = {"C1": 5.0, "C2": 1.0, "C3": 0.5}


class FakeReranker:
    """Deterministic cross-encoder stand-in that records every call."""

    def __init__(self, scores_by_doc):
        self.scores = scores_by_doc
        self.calls = []

    def rerank(self, query, candidates, top_k=10):
        self.calls.append((query, [candidate["id"] for candidate in candidates], top_k))
        scored = [
            {**candidate, "reranker_score": self.scores[candidate["id"]]}
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item["reranker_score"], item["id"]))
        return scored[:top_k]


def make_methods(k, candidate_k, retriever, reranker):
    return {
        "Base": lambda q: [r["id"] for r in retriever.search(q, top_k=k)],
        "Base + Reranker": lambda q: rerank_query(q, retriever, reranker, candidate_k, k),
    }


def build_payload(k=2, candidate_k=3, max_queries=None, factory_calls=None):
    def method_factory():
        if factory_calls is not None:
            factory_calls.append(1)
        return make_methods(k, candidate_k, FakeRetriever(RETRIEVER_RESULTS), FakeReranker(RERANKER_SCORES))

    return run_reranker_benchmark(
        DATASET,
        k=k,
        candidate_k=candidate_k,
        dense_model="fake/dense",
        reranker_model="fake/reranker",
        max_queries=max_queries,
        method_factory=method_factory,
    )


# --- candidate pool semantics ---


def test_candidate_k_greater_than_k_feeds_full_pool_to_reranker():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    reranker = FakeReranker(RERANKER_SCORES)
    method_factory = lambda: make_methods(2, 3, retriever, reranker)

    run_reranker_benchmark(
        DATASET, k=2, candidate_k=3, method_factory=method_factory
    )

    assert reranker.calls == [
        ("query one", ["C2", "C1", "C3"], 2),
        ("query two", ["C3", "C1"], 2),
    ]
    # The retrieval stage really fetched candidate_k=3, not the final k=2.
    assert ("query one", 3) in retriever.calls


def test_candidate_k_equals_k():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    reranker = FakeReranker(RERANKER_SCORES)
    method_factory = lambda: make_methods(2, 2, retriever, reranker)

    payload = run_reranker_benchmark(
        DATASET, k=2, candidate_k=2, method_factory=method_factory
    )

    assert reranker.calls[0] == ("query one", ["C2", "C1"], 2)
    assert payload["metadata"]["candidate_k"] == 2


def test_candidate_k_smaller_than_k_raises_before_building():
    factory_calls = []

    with pytest.raises(ValueError, match="candidate_k"):
        build_payload(k=3, candidate_k=2, factory_calls=factory_calls)

    # Validation happens before any method is constructed.
    assert factory_calls == []


# --- evaluation semantics ---


def test_reranker_receives_expected_candidate_set():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    reranker = FakeReranker(RERANKER_SCORES)

    rerank_query("query one", retriever, reranker, candidate_k=3, k=2)

    assert reranker.calls == [("query one", ["C2", "C1", "C3"], 2)]


def test_evaluation_uses_reranked_top_k_ids():
    payload = build_payload(k=2, candidate_k=3)

    # Base: q1 -> [C2, C1] (mrr 1.0), q2 -> [C3, C1] (mrr 1.0).
    # Reranked: q1 -> [C1, C2] (mrr 1.0), q2 -> [C1, C3] (mrr 0.5).
    # If the pre-rerank order were evaluated, both would give mrr 1.0.
    expected_reranked = mean_metrics(
        [
            evaluate_query(["C1", "C2"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C1", "C3"], DATASET.qrels["q2"], k=2),
        ]
    )
    assert payload["results"]["Base + Reranker"] == pytest.approx(expected_reranked)
    assert payload["results"]["Base + Reranker"]["mrr@2"] == pytest.approx(0.75)
    assert payload["results"]["Base"]["mrr@2"] == pytest.approx(1.0)


def test_fake_reranker_changes_metrics_both_directions():
    payload = build_payload(k=2, candidate_k=3)
    base = payload["results"]["Base"]
    reranked = payload["results"]["Base + Reranker"]

    base_q1 = evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2)
    reranked_q1 = evaluate_query(["C1", "C2"], DATASET.qrels["q1"], k=2)
    base_q2 = evaluate_query(["C3", "C1"], DATASET.qrels["q2"], k=2)
    reranked_q2 = evaluate_query(["C1", "C3"], DATASET.qrels["q2"], k=2)

    # q1: promoting the higher-graded C1 improves nDCG.
    assert reranked_q1["ndcg@2"] > base_q1["ndcg@2"]
    # q2: promoting the non-relevant C1 hurts both MRR and nDCG.
    assert reranked_q2["mrr@2"] < base_q2["mrr@2"]
    assert reranked_q2["ndcg@2"] < base_q2["ndcg@2"]
    # The means reflect both directions honestly: they can also drop.
    assert reranked["ndcg@2"] < base["ndcg@2"]
    assert reranked["mrr@2"] < base["mrr@2"]


def test_baseline_methods_evaluate_correctly():
    payload = build_payload(k=2, candidate_k=3)

    expected_base = mean_metrics(
        [
            evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C3", "C1"], DATASET.qrels["q2"], k=2),
        ]
    )
    assert payload["results"]["Base"] == pytest.approx(expected_base)


# --- construction reuse, limits, metadata, output paths ---


def test_reranker_is_constructed_and_reused_once():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    reranker = FakeReranker(RERANKER_SCORES)
    factory_calls = []

    def method_factory():
        factory_calls.append(1)
        return make_methods(2, 3, retriever, reranker)

    run_reranker_benchmark(DATASET, k=2, candidate_k=3, method_factory=method_factory)

    # One factory call -> one reranker instance serving both queries.
    assert factory_calls == [1]
    assert len(reranker.calls) == 2


def test_query_limit_behavior():
    payload = build_payload(k=2, candidate_k=3, max_queries=1)

    assert payload["metadata"]["num_queries"] == 1
    assert payload["metadata"]["max_queries"] == 1


def test_metadata_schema():
    payload = build_payload(k=2, candidate_k=3)

    metadata = payload["metadata"]
    assert metadata["dataset"] == "scifact"
    assert metadata["dense_model"] == "fake/dense"
    assert metadata["reranker_model"] == "fake/reranker"
    assert metadata["k"] == 2
    assert metadata["candidate_k"] == 3
    assert metadata["alpha"] == 0.5
    assert metadata["rrf_k"] == 60.0
    assert metadata["num_queries"] == 2
    assert metadata["max_queries"] is None
    assert metadata["methods"] == ["Base", "Base + Reranker"]
    assert set(payload["results"]) == {"Base", "Base + Reranker"}


def test_saved_payload_shape(tmp_path):
    payload = build_payload(k=2, candidate_k=3)
    path = tmp_path / "reranker_benchmark.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert loaded["metadata"]["methods"] == ["Base", "Base + Reranker"]
    assert set(loaded["results"]["Base + Reranker"]) == {
        "precision@2",
        "recall@2",
        "mrr@2",
        "ndcg@2",
    }


def test_smoke_and_full_output_paths_do_not_collide():
    smoke = output_path_for(20)
    full = output_path_for(None)

    assert smoke == "outputs/scifact_reranker_smoke20.json"
    assert full == "outputs/scifact_reranker_benchmark.json"
    assert smoke != full

    # Neither collides with the Phase 6/7 outputs.
    assert smoke not in ("outputs/scifact_benchmark.json", "outputs/scifact_latency.json")
    assert full not in ("outputs/scifact_benchmark.json", "outputs/scifact_latency.json")


def test_evaluate_method_matches_manual_metrics():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    search_fn = lambda q: [r["id"] for r in retriever.search(q, top_k=2)]

    means = evaluate_method(search_fn, DATASET, ["q1", "q2"], k=2)
    expected = mean_metrics(
        [
            evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C3", "C1"], DATASET.qrels["q2"], k=2),
        ]
    )

    assert means == pytest.approx(expected)


# --- determinism ---


def test_reranker_benchmark_is_deterministic():
    first = build_payload(k=2, candidate_k=3)
    second = build_payload(k=2, candidate_k=3)

    assert first == second
