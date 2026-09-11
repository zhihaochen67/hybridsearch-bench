"""Offline tests for multi-dataset experiment dispatch (Phase 13).

Every experiment harness is driven with a synthetic ``FiQADataset`` and
fake retrievers/rerankers/timers: no network, no real models, no
``datasets`` import. Together with ``test_dataset_registry.py`` these
cover the ``--dataset scifact|fiqa`` CLI surface and dataset-scoped
output naming.
"""

import pytest

from experiments.benchmark import (
    output_path_for as benchmark_output_path_for,
    parse_args as benchmark_parse_args,
    run_benchmark,
)
from experiments.fusion_ablation import (
    output_path_for as fusion_output_path_for,
    parse_args as fusion_parse_args,
    run_fusion_ablation,
)
from experiments.latency_benchmark import (
    output_path_for as latency_output_path_for,
    parse_args as latency_parse_args,
    run_latency_benchmark,
)
from experiments.quality_latency import (
    output_path_for as quality_output_path_for,
    parse_args as quality_parse_args,
    plot_all as quality_plot_all,
    run_quality_latency,
)
from experiments.reranker_benchmark import (
    output_path_for as reranker_output_path_for,
    parse_args as reranker_parse_args,
    run_reranker_benchmark,
)
from hybridsearch.data.fiqa import FiQADataset

FIQA = FiQADataset(
    corpus=[
        {"id": "C1", "text": "alpha doc"},
        {"id": "C2", "text": "beta doc"},
        {"id": "C3", "text": "gamma doc"},
    ],
    queries={
        "q1": "query one",
        "q2": "query two",
        "q3": "query three",  # no qrels -> excluded
    },
    qrels={
        "q1": {"C1": 1, "C2": 1},
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


class FakeTimer:
    """Advances by 5 ms per call."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        current = self.now
        self.now += 0.005
        return current


def fake_benchmark_factory():
    return {
        "Fake A": FakeRetriever(FAKE_RESULTS_A),
        "Fake B": FakeRetriever(FAKE_RESULTS_A),
    }


def fake_alpha_factory(alphas):
    return {
        f"alpha={alpha:.2f}": FakeRetriever(FAKE_RESULTS_A)
        for alpha in alphas
    }


# --- benchmark harness ---


def test_benchmark_supports_fiqa_dataset():
    payload = run_benchmark(FIQA, k=2, retriever_factory=fake_benchmark_factory)

    assert payload["metadata"]["dataset"] == "fiqa"
    assert payload["metadata"]["source"] == "BeIR/fiqa"
    assert payload["metadata"]["num_documents"] == 3
    assert payload["metadata"]["num_queries"] == 2
    assert set(payload["results"]) == {"Fake A", "Fake B"}


def test_benchmark_output_paths_are_dataset_scoped():
    assert benchmark_output_path_for("fiqa") == "outputs/fiqa_benchmark.json"
    assert (
        benchmark_output_path_for("fiqa", 20)
        == "outputs/fiqa_benchmark_smoke20.json"
    )
    assert benchmark_output_path_for("scifact") == "outputs/scifact_benchmark.json"
    assert (
        benchmark_output_path_for("scifact", 20)
        == "outputs/scifact_benchmark_smoke20.json"
    )
    assert benchmark_output_path_for("fiqa") != benchmark_output_path_for("scifact")


def test_benchmark_cli_accepts_both_datasets():
    assert benchmark_parse_args([]).dataset == "scifact"
    assert benchmark_parse_args(["--dataset", "scifact"]).dataset == "scifact"
    assert benchmark_parse_args(["--dataset", "fiqa"]).dataset == "fiqa"


def test_benchmark_cli_output_default_is_none():
    assert benchmark_parse_args([]).output is None


# --- fusion ablation harness ---


def test_fusion_ablation_supports_fiqa_dataset():
    alphas = [0.0, 0.5, 1.0]
    payload = run_fusion_ablation(
        FIQA,
        k=2,
        alpha_values=alphas,
        retriever_factory=lambda: fake_alpha_factory(alphas),
    )

    assert payload["metadata"]["dataset"] == "fiqa"
    assert payload["metadata"]["source"] == "BeIR/fiqa"
    assert payload["metadata"]["alpha_values"] == [0.0, 0.5, 1.0]
    assert set(payload["results"]) == {"alpha=0.00", "alpha=0.50", "alpha=1.00"}


def test_fusion_output_paths_are_dataset_scoped():
    assert (
        fusion_output_path_for(20, "fiqa")
        == "outputs/fiqa_fusion_ablation_smoke20.json"
    )
    assert (
        fusion_output_path_for(None, "fiqa")
        == "outputs/fiqa_fusion_ablation.json"
    )
    # The historical single-argument SciFact API is unchanged.
    assert (
        fusion_output_path_for(20)
        == "outputs/scifact_fusion_ablation_smoke20.json"
    )
    assert fusion_output_path_for(None) == "outputs/scifact_fusion_ablation.json"


def test_fusion_cli_accepts_fiqa():
    assert fusion_parse_args(["--dataset", "fiqa"]).dataset == "fiqa"


# --- reranker harness ---


def test_reranker_benchmark_supports_fiqa_dataset():
    def method_factory():
        retriever = FakeRetriever(FAKE_RESULTS_A)
        return {
            "Base": lambda q: [r["id"] for r in retriever.search(q, top_k=2)]
        }

    payload = run_reranker_benchmark(
        FIQA, k=2, candidate_k=3, method_factory=method_factory
    )

    assert payload["metadata"]["dataset"] == "fiqa"
    assert payload["metadata"]["source"] == "BeIR/fiqa"
    assert payload["metadata"]["methods"] == ["Base"]


def test_reranker_output_paths_are_dataset_scoped():
    assert (
        reranker_output_path_for(20, "fiqa")
        == "outputs/fiqa_reranker_smoke20.json"
    )
    assert (
        reranker_output_path_for(None, "fiqa")
        == "outputs/fiqa_reranker_benchmark.json"
    )
    assert (
        reranker_output_path_for(20)
        == "outputs/scifact_reranker_smoke20.json"
    )


def test_reranker_cli_accepts_fiqa():
    assert reranker_parse_args(["--dataset", "fiqa"]).dataset == "fiqa"


# --- latency harness ---


def test_latency_benchmark_supports_fiqa_dataset():
    payload = run_latency_benchmark(
        FIQA,
        k=2,
        warmup_queries=0,
        retriever_factory=lambda: {"Only": FakeRetriever(FAKE_RESULTS_A)},
        timer=FakeTimer(),
    )

    assert payload["metadata"]["dataset"] == "fiqa"
    assert payload["metadata"]["source"] == "BeIR/fiqa"
    assert payload["metadata"]["num_queries"] == 2


def test_latency_output_path_is_dataset_scoped():
    assert latency_output_path_for("fiqa") == "outputs/fiqa_latency.json"
    assert latency_output_path_for("scifact") == "outputs/scifact_latency.json"
    assert (
        latency_output_path_for("fiqa", 20)
        == "outputs/fiqa_latency_smoke20.json"
    )
    assert (
        latency_output_path_for("scifact", 20)
        == "outputs/scifact_latency_smoke20.json"
    )


def test_latency_cli_accepts_fiqa():
    assert latency_parse_args(["--dataset", "fiqa"]).dataset == "fiqa"
    assert latency_parse_args([]).output is None


# --- quality / latency harness ---


def test_quality_latency_supports_fiqa_dataset():
    def method_factory():
        retriever = FakeRetriever(FAKE_RESULTS_A)
        return {
            "Base": lambda q: [r["id"] for r in retriever.search(q, top_k=2)]
        }

    payload = run_quality_latency(
        FIQA,
        k=2,
        candidate_sizes=[2],
        warmup_queries=0,
        method_factory=method_factory,
        timer=FakeTimer(),
    )

    assert payload["metadata"]["dataset"] == "fiqa"
    assert payload["metadata"]["source"] == "BeIR/fiqa"
    assert payload["metadata"]["candidate_sizes"] == [2]


def test_quality_output_paths_are_dataset_scoped():
    assert (
        quality_output_path_for(20, "fiqa")
        == "outputs/fiqa_quality_latency_smoke20.json"
    )
    assert (
        quality_output_path_for(None, "fiqa")
        == "outputs/fiqa_quality_latency.json"
    )
    assert (
        quality_output_path_for(20)
        == "outputs/scifact_quality_latency_smoke20.json"
    )


def test_quality_fiqa_plot_filenames_do_not_collide(tmp_path):
    def method_factory():
        retriever = FakeRetriever(FAKE_RESULTS_A)
        return {
            "Base": lambda q: [r["id"] for r in retriever.search(q, top_k=2)]
        }

    payload = run_quality_latency(
        FIQA,
        k=2,
        candidate_sizes=[2],
        warmup_queries=0,
        method_factory=method_factory,
        timer=FakeTimer(),
    )

    paths = quality_plot_all(payload, output_dir=tmp_path)

    assert {path.name for path in paths} == {
        "fiqa_quality_latency_ndcg.png",
        "fiqa_reranker_candidate_tradeoff.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_quality_cli_accepts_fiqa():
    assert quality_parse_args(["--dataset", "fiqa"]).dataset == "fiqa"


# --- CLI validation + cross-cutting guarantees ---


@pytest.mark.parametrize(
    "parse",
    [
        benchmark_parse_args,
        fusion_parse_args,
        latency_parse_args,
        quality_parse_args,
        reranker_parse_args,
    ],
)
def test_cli_rejects_unknown_dataset(parse):
    with pytest.raises(SystemExit):
        parse(["--dataset", "nope"])


def test_fiqa_output_filenames_never_collide_with_scifact():
    scifact_paths = {
        benchmark_output_path_for("scifact"),
        benchmark_output_path_for("scifact", 20),
        fusion_output_path_for(None, "scifact"),
        fusion_output_path_for(20, "scifact"),
        latency_output_path_for("scifact"),
        latency_output_path_for("scifact", 20),
        reranker_output_path_for(None, "scifact"),
        reranker_output_path_for(20, "scifact"),
        quality_output_path_for(None, "scifact"),
        quality_output_path_for(20, "scifact"),
    }
    fiqa_paths = {
        benchmark_output_path_for("fiqa"),
        benchmark_output_path_for("fiqa", 20),
        fusion_output_path_for(None, "fiqa"),
        fusion_output_path_for(20, "fiqa"),
        latency_output_path_for("fiqa"),
        latency_output_path_for("fiqa", 20),
        reranker_output_path_for(None, "fiqa"),
        reranker_output_path_for(20, "fiqa"),
        quality_output_path_for(None, "fiqa"),
        quality_output_path_for(20, "fiqa"),
    }

    assert len(scifact_paths) == 10
    assert len(fiqa_paths) == 10
    assert not (scifact_paths & fiqa_paths)


def test_experiment_outputs_contain_dataset_metadata():
    payloads = [
        run_benchmark(FIQA, k=2, retriever_factory=fake_benchmark_factory),
        run_fusion_ablation(
            FIQA, k=2, alpha_values=[0.0, 1.0],
            retriever_factory=lambda: fake_alpha_factory([0.0, 1.0]),
        ),
        run_reranker_benchmark(
            FIQA, k=2, candidate_k=3,
            method_factory=lambda: {
                "Base": lambda q: [r["id"] for r in FakeRetriever(FAKE_RESULTS_A).search(q, top_k=2)]
            },
        ),
        run_latency_benchmark(
            FIQA, k=2, warmup_queries=0,
            retriever_factory=lambda: {"Only": FakeRetriever(FAKE_RESULTS_A)},
            timer=FakeTimer(),
        ),
        run_quality_latency(
            FIQA, k=2, candidate_sizes=[2], warmup_queries=0,
            method_factory=lambda: {
                "Base": lambda q: [r["id"] for r in FakeRetriever(FAKE_RESULTS_A).search(q, top_k=2)]
            },
            timer=FakeTimer(),
        ),
    ]

    for payload in payloads:
        assert "dataset" in payload["metadata"]
        assert payload["metadata"]["dataset"] == "fiqa"
        assert "source" in payload["metadata"]


# --- determinism ---


def test_fiqa_benchmark_is_deterministic():
    first = run_benchmark(FIQA, k=2, retriever_factory=fake_benchmark_factory)
    second = run_benchmark(FIQA, k=2, retriever_factory=fake_benchmark_factory)

    assert first == second


def test_fiqa_fusion_ablation_is_deterministic():
    alphas = [0.0, 0.5, 1.0]
    first = run_fusion_ablation(
        FIQA, k=2, alpha_values=alphas,
        retriever_factory=lambda: fake_alpha_factory(alphas),
    )
    second = run_fusion_ablation(
        FIQA, k=2, alpha_values=alphas,
        retriever_factory=lambda: fake_alpha_factory(alphas),
    )

    assert first == second
