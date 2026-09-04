"""Offline tests for the quality-vs-latency candidate-size study (Phase 10).

No SciFact download and no real models: a synthetic ``SciFactDataset``,
fake retrievers/reranker, and an injected fake timer keep every test
deterministic and network-free. Plot tests render with matplotlib's Agg
backend to temporary files.
"""

import json

import pytest

from experiments.benchmark import save_results
from experiments.quality_latency import (
    CANDIDATE_SIZES_DEFAULT,
    output_path_for,
    parse_candidate_sizes,
    plot_all,
    rerank_points,
    run_method,
    run_quality_latency,
    validate_candidate_sizes,
)
from hybridsearch.data.scifact import SciFactDataset
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics


DATASET = SciFactDataset(
    corpus=[
        {"id": "C1", "text": "alpha doc"},
        {"id": "C2", "text": "beta doc"},
        {"id": "C3", "text": "gamma doc"},
        {"id": "C4", "text": "delta doc"},
    ],
    queries={
        "q1": "query one",
        "q2": "query two",
        "q3": "query three",  # no qrels -> excluded
    },
    qrels={
        "q1": {"C1": 2, "C2": 1},
        "q2": {"C3": 1},
    },
)

RETRIEVER_RESULTS = {
    "query one": [
        {"id": "C4", "text": "delta doc", "score": 9.0},
        {"id": "C3", "text": "gamma doc", "score": 6.0},
        {"id": "C2", "text": "beta doc", "score": 3.0},
        {"id": "C1", "text": "alpha doc", "score": 1.0},
    ],
    "query two": [
        {"id": "C1", "text": "alpha doc", "score": 7.0},
        {"id": "C2", "text": "beta doc", "score": 5.0},
        {"id": "C3", "text": "gamma doc", "score": 2.0},
    ],
}

# Reranker order per document: C1 > C2 > C3 > C4.
RERANKER_SCORES = {"C1": 5.0, "C2": 4.0, "C3": 1.0, "C4": 0.0}


class FakeRetriever:
    def __init__(self, results_by_query):
        self.results = results_by_query
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


class FakeReranker:
    def __init__(self, scores_by_doc):
        self.scores = scores_by_doc
        self.calls = []

    def rerank(self, query, candidates, top_k=10):
        self.calls.append((query, [c["id"] for c in candidates], top_k))
        scored = [
            {**c, "reranker_score": self.scores[c["id"]]} for c in candidates
        ]
        scored.sort(key=lambda item: (-item["reranker_score"], item["id"]))
        return scored[:top_k]


class FakeTimer:
    """Advances by 5 ms per call: every timed query costs exactly 5 ms."""

    def __init__(self):
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        current = self.now
        self.now += 0.005
        return current


def make_methods(k, sizes, retriever, reranker):
    methods = {
        "Base": lambda q: [r["id"] for r in retriever.search(q, top_k=k)],
    }
    for size in sizes:
        def make(size):
            return lambda q: [
                r["id"]
                for r in reranker.rerank(
                    q, retriever.search(q, top_k=size), top_k=k
                )
            ]

        methods[f"Base + rerank@{size}"] = make(size)
    return methods


def run_study(k=2, sizes=(2, 3, 4), warmup=1, created=None):
    def method_factory():
        retriever = FakeRetriever(RETRIEVER_RESULTS)
        reranker = FakeReranker(RERANKER_SCORES)
        if created is not None:
            created.append((retriever, reranker))
        return make_methods(k, sizes, retriever, reranker)

    return run_quality_latency(
        DATASET,
        k=k,
        candidate_sizes=list(sizes),
        alpha=0.5,
        dense_model="fake/dense",
        reranker_model="fake/reranker",
        hybrid_candidate_k=4,
        warmup_queries=warmup,
        method_factory=method_factory,
        timer=FakeTimer(),
    )


# --- candidate size parsing / validation ---


def test_default_candidate_sizes():
    assert CANDIDATE_SIZES_DEFAULT == [20, 50, 100]


def test_parse_candidate_sizes():
    assert parse_candidate_sizes("20,50,100", k=10) == [20, 50, 100]
    assert parse_candidate_sizes(" 20 , 50 ", k=10) == [20, 50]


@pytest.mark.parametrize("text", ["", "   ", "abc", "0,50", "20.5"])
def test_parse_candidate_sizes_invalid_raises(text):
    with pytest.raises(ValueError):
        parse_candidate_sizes(text, k=10)


def test_candidate_size_must_be_at_least_k():
    with pytest.raises(ValueError, match="candidate size"):
        validate_candidate_sizes([10, 50], k=20)
    # size == k is allowed.
    validate_candidate_sizes([10, 10], k=10)


def test_validate_candidate_sizes_rejects_empty_and_non_positive():
    with pytest.raises(ValueError):
        validate_candidate_sizes([], k=10)
    with pytest.raises(ValueError):
        validate_candidate_sizes([-1], k=10)


# --- candidate fetching semantics ---


def test_rerank_methods_fetch_exactly_their_candidate_size():
    created = []
    run_study(sizes=(2, 3, 4), warmup=0, created=created)
    reranker = created[0][1]

    by_size = {}
    for query, candidate_ids, top_k in reranker.calls:
        assert top_k == 2  # final k
        by_size.setdefault(len(candidate_ids), (query, candidate_ids))

    # "query one" has 4 documents, so sizes 2, 3, and 4 fetch 2/3/4 docs.
    assert by_size[2] == ("query one", ["C4", "C3"])
    assert by_size[3] == ("query one", ["C4", "C3", "C2"])
    assert by_size[4] == ("query one", ["C4", "C3", "C2", "C1"])


def test_final_evaluation_uses_reranked_top_k_ids():
    payload = run_study(sizes=(2, 3, 4), warmup=0)

    # rerank@3 for q1: candidates [C4, C3, C2] -> reranked [C2, C3, C4]
    # -> final top-2 [C2, C3]; q2 -> [C1, C2].
    expected = mean_metrics(
        [
            evaluate_query(["C2", "C3"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C1", "C2"], DATASET.qrels["q2"], k=2),
        ]
    )
    assert payload["results"]["Base + rerank@3"]["quality"] == pytest.approx(
        expected
    )


def test_quality_improves_as_candidate_size_grows():
    payload = run_study(sizes=(2, 3, 4), warmup=0)

    ndcg2 = payload["results"]["Base + rerank@2"]["quality"]["ndcg@2"]
    ndcg3 = payload["results"]["Base + rerank@3"]["quality"]["ndcg@2"]
    ndcg4 = payload["results"]["Base + rerank@4"]["quality"]["ndcg@2"]

    # Larger pools let the reranker reach better documents: 0 -> 0.138 -> 0.5.
    assert ndcg2 == pytest.approx(0.0)
    assert ndcg3 > ndcg2
    assert ndcg4 > ndcg3
    assert payload["results"]["Base"]["quality"]["mrr@2"] == pytest.approx(0.0)


# --- reuse and timing semantics ---


def test_same_retriever_and_reranker_reused_across_sizes():
    created = []
    run_study(sizes=(2, 3, 4), warmup=0, created=created)

    assert len(created) == 1  # one method construction
    retriever, reranker = created[0]

    # All five methods (Base + rerank@2/3/4) shared the single instances.
    assert len(retriever.calls) == 8  # 1 base + 3 sizes, over 2 queries
    assert len(reranker.calls) == 6  # 3 sizes over 2 queries


def test_latency_covers_full_reranking_path():
    created = []
    payload = run_study(sizes=(4,), warmup=1, created=created)
    retriever, reranker = created[0]

    entry = payload["results"]["Base + rerank@4"]
    assert entry["latency"]["samples"] == 2
    assert entry["latency"]["mean_ms"] == pytest.approx(5.0)
    assert entry["latency"]["median_ms"] == pytest.approx(5.0)
    assert entry["latency"]["p95_ms"] == pytest.approx(5.0)

    # Each timed query ran BOTH stages (warmup + timed): retrieval and
    # reranking happen inside the measured call.
    assert ("query one", 4) in retriever.calls
    assert ("query one", [r["id"] for r in RETRIEVER_RESULTS["query one"][:4]], 2) in reranker.calls


def test_construction_is_excluded_from_timing():
    payload = run_study(sizes=(2, 3, 4), warmup=0)

    # Every timed query costs exactly two timer calls (5 ms); method
    # construction ran before any timing, so nothing disturbs the 5 ms.
    for entry in payload["results"].values():
        assert entry["latency"]["mean_ms"] == pytest.approx(5.0)
        assert entry["latency"]["min_ms"] == pytest.approx(5.0)
        assert entry["latency"]["max_ms"] == pytest.approx(5.0)


def test_warmup_calls_are_excluded_from_latency_samples():
    payload = run_study(sizes=(4,), warmup=1)

    entry = payload["results"]["Base + rerank@4"]
    assert entry["latency"]["samples"] == 2  # two queries, not three


def test_run_method_returns_quality_and_latency():
    retriever = FakeRetriever(RETRIEVER_RESULTS)
    reranker = FakeReranker(RERANKER_SCORES)

    def search_fn(query):
        candidates = retriever.search(query, top_k=3)
        return [r["id"] for r in reranker.rerank(query, candidates, top_k=2)]

    entry = run_method(
        search_fn,
        ["q1", "q2"],
        ["query one", "query two"],
        DATASET.qrels,
        k=2,
        warmup=1,
        timer=FakeTimer(),
    )

    assert set(entry) == {"quality", "latency"}
    assert set(entry["quality"]) == {
        "precision@2", "recall@2", "mrr@2", "ndcg@2",
    }
    assert set(entry["latency"]) == {
        "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "samples",
    }
    assert entry["latency"]["samples"] == 2


# --- output schema, filenames, plotting ---


def test_output_json_schema(tmp_path):
    payload = run_study(sizes=(2, 3, 4), warmup=1)
    path = tmp_path / "quality_latency.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert set(loaded["metadata"]) == {
        "dataset", "source", "dense_model", "reranker_model", "k",
        "candidate_sizes", "alpha", "hybrid_candidate_k", "num_queries",
        "max_queries", "warmup_queries", "timer", "timing_passes",
        "methodology",
    }
    assert loaded["metadata"]["candidate_sizes"] == [2, 3, 4]
    assert loaded["metadata"]["num_queries"] == 2
    assert loaded["metadata"]["timing_passes"] == 1
    assert isinstance(loaded["metadata"]["methodology"], list)

    assert set(loaded["results"]) == {
        "Base", "Base + rerank@2", "Base + rerank@3", "Base + rerank@4",
    }
    for entry in loaded["results"].values():
        assert set(entry) == {"quality", "latency"}


def test_smoke_and_full_output_paths_do_not_collide():
    smoke = output_path_for(20)
    full = output_path_for(None)

    assert smoke == "outputs/scifact_quality_latency_smoke20.json"
    assert full == "outputs/scifact_quality_latency.json"
    assert smoke != full


def test_rerank_points_ordered_by_candidate_size():
    payload = run_study(sizes=(4, 2, 3), warmup=0)

    sizes, entries = rerank_points(payload)

    assert sizes == [2, 3, 4]
    assert all(set(entry) == {"quality", "latency"} for entry in entries)


def test_plot_generation_reads_payload(tmp_path):
    payload = run_study(sizes=(2, 3, 4), warmup=0)

    paths = plot_all(payload, output_dir=tmp_path)

    assert {path.name for path in paths} == {
        "quality_latency_ndcg.png",
        "reranker_candidate_tradeoff.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


# --- determinism ---


def test_quality_latency_study_is_deterministic():
    first = run_study(sizes=(2, 3, 4), warmup=1)
    second = run_study(sizes=(2, 3, 4), warmup=1)

    assert first == second
