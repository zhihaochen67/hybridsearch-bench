"""Offline tests for the latency benchmark harness (Phase 7).

No models or datasets are loaded: a synthetic ``SciFactDataset``, fake
retrievers, and an injected fake timer keep everything deterministic and
network-free.
"""

import json

import pytest

from experiments.benchmark import save_results
from experiments.latency_benchmark import (
    format_latency_table,
    measure_method,
    run_latency_benchmark,
    summarize_latencies,
    time_query,
)
from hybridsearch.data.scifact import SciFactDataset


DATASET = SciFactDataset(
    corpus=[
        {"id": "C1", "text": "alpha doc"},
        {"id": "C2", "text": "beta doc"},
    ],
    queries={
        "q1": "query one",
        "q2": "query two",
        "q3": "query three",  # no qrels -> excluded from measurement
    },
    qrels={
        "q1": {"C1": 1},
        "q2": {"C2": 1},
    },
)


class FakeRetriever:
    """Deterministic stand-in that records every search call."""

    def __init__(self):
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return []


class FakeTimer:
    """Advances by 5 ms on every call, so each timed search costs 5 ms."""

    def __init__(self):
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        current = self.now
        self.now += 0.005
        return current


def fake_factory():
    return {"M1": FakeRetriever(), "M2": FakeRetriever()}


# --- latency aggregation ---


def test_summarize_latencies_mean_median_p95():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary["mean_ms"] == pytest.approx(22.0)
    assert summary["median_ms"] == pytest.approx(3.0)
    # Nearest-rank p95 of 5 samples: ceil(0.95 * 5) = 5 -> the largest value.
    assert summary["p95_ms"] == pytest.approx(100.0)
    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["max_ms"] == pytest.approx(100.0)
    assert summary["samples"] == 5


def test_summarize_latencies_p95_nearest_rank():
    # 20 samples 1..20: ceil(0.95 * 20) = 19 -> the 19th value.
    summary = summarize_latencies(list(range(1, 21)))

    assert summary["median_ms"] == pytest.approx(10.5)
    assert summary["p95_ms"] == pytest.approx(19.0)


def test_summarize_latencies_single_sample():
    summary = summarize_latencies([5.0])

    assert summary["mean_ms"] == pytest.approx(5.0)
    assert summary["median_ms"] == pytest.approx(5.0)
    assert summary["p95_ms"] == pytest.approx(5.0)
    assert summary["samples"] == 1


def test_summarize_latencies_empty_raises():
    with pytest.raises(ValueError):
        summarize_latencies([])


# --- timing mechanics ---


def test_time_query_measures_search_wall_time():
    retriever = FakeRetriever()
    timer = FakeTimer()

    latency_ms = time_query(retriever, "query one", top_k=10, timer=timer)

    assert latency_ms == pytest.approx(5.0)
    assert retriever.calls == [("query one", 10)]
    assert timer.calls == 2


def test_measure_method_warmup_is_untimed():
    retriever = FakeRetriever()
    timer = FakeTimer()

    latencies = measure_method(
        retriever,
        ["query one", "query two", "query three"],
        top_k=10,
        warmup=2,
        timer=timer,
    )

    # Two warmup searches (untimed) + three timed searches at 5 ms each.
    assert retriever.calls == [
        ("query one", 10),
        ("query two", 10),
        ("query one", 10),
        ("query two", 10),
        ("query three", 10),
    ]
    assert latencies == pytest.approx([5.0, 5.0, 5.0])
    # Only the three timed searches touched the timer (2 calls each).
    assert timer.calls == 6


# --- harness behavior ---


def test_run_latency_benchmark_measures_only_qrel_queries():
    retriever = FakeRetriever()
    payload = run_latency_benchmark(
        DATASET,
        k=10,
        warmup_queries=1,
        retriever_factory=lambda: {"Only": retriever},
    )

    assert payload["metadata"]["num_queries"] == 2
    summary = payload["results"]["Only"]
    assert summary["samples"] == 2
    assert len(summary["latencies_ms"]) == 2

    searched = [query for query, _ in retriever.calls]
    assert "query three" not in searched


def test_run_latency_benchmark_metadata_and_shape():
    payload = run_latency_benchmark(
        DATASET,
        k=10,
        candidate_k=20,
        alpha=0.3,
        rrf_k=70.0,
        dense_model="fake/model",
        warmup_queries=2,
        retriever_factory=fake_factory,
        timer=FakeTimer(),
    )
    metadata = payload["metadata"]

    assert metadata["dataset"] == "scifact"
    assert metadata["k"] == 10
    assert metadata["candidate_k"] == 20
    assert metadata["alpha"] == 0.3
    assert metadata["rrf_k"] == 70.0
    assert metadata["dense_model"] == "fake/model"
    assert metadata["num_queries"] == 2
    assert metadata["warmup_queries"] == 2
    assert metadata["timer"] == "FakeTimer"

    assert set(payload["results"]) == {"M1", "M2"}
    for summary in payload["results"].values():
        assert set(summary) == {
            "mean_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "samples",
            "latencies_ms",
        }
        assert summary["mean_ms"] == pytest.approx(5.0)


def test_run_latency_benchmark_query_limit():
    payload = run_latency_benchmark(
        DATASET,
        k=10,
        max_queries=1,
        warmup_queries=0,
        retriever_factory=fake_factory,
    )

    assert payload["metadata"]["num_queries"] == 1
    assert payload["results"]["M1"]["samples"] == 1


def test_run_latency_benchmark_empty_query_set_raises():
    empty = SciFactDataset(corpus=[], queries={"q1": "x"}, qrels={})

    with pytest.raises(ValueError):
        run_latency_benchmark(empty, retriever_factory=fake_factory)


def test_saved_payload_shape(tmp_path):
    payload = run_latency_benchmark(
        DATASET,
        k=10,
        warmup_queries=1,
        retriever_factory=fake_factory,
        timer=FakeTimer(),
    )
    path = tmp_path / "latency.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert loaded["metadata"]["num_queries"] == 2
    assert loaded["results"]["M1"]["samples"] == 2
    assert loaded["results"]["M1"]["latencies_ms"] == pytest.approx([5.0, 5.0])


def test_format_latency_table_contains_methods_and_columns():
    payload = run_latency_benchmark(
        DATASET,
        k=10,
        warmup_queries=0,
        retriever_factory=fake_factory,
    )

    table = format_latency_table(payload)

    for column in ("Method", "Mean ms", "Median ms", "P95 ms"):
        assert column in table
    for name in ("M1", "M2"):
        assert name in table


def test_latency_benchmark_is_deterministic():
    first = run_latency_benchmark(
        DATASET,
        k=10,
        warmup_queries=1,
        retriever_factory=fake_factory,
        timer=FakeTimer(),
    )
    second = run_latency_benchmark(
        DATASET,
        k=10,
        warmup_queries=1,
        retriever_factory=fake_factory,
        timer=FakeTimer(),
    )

    assert first == second
