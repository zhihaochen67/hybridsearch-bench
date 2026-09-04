"""Offline tests for the fusion alpha ablation (Phase 9).

No SciFact download and no real models: a synthetic ``SciFactDataset``,
fake retrievers, and an injected fake encoder drive every test. Plot tests
use matplotlib's Agg backend (rendering to files only, fully offline).
"""

import json

import numpy as np
import pytest

from experiments.benchmark import save_results
from experiments.fusion_ablation import (
    ALPHA_GRID,
    build_alpha_retrievers,
    extract_curve,
    format_ablation_table,
    output_path_for,
    parse_alpha_values,
    plot_ablation,
    run_fusion_ablation,
    validate_alpha_values,
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
        "q3": "query three",  # no qrels -> excluded
    },
    qrels={
        "q1": {"C1": 2, "C2": 1},
        "q2": {"C3": 1},
    },
)


class FakeRetriever:
    """Deterministic stand-in that records every search call."""

    def __init__(self, results_by_query):
        self.results = results_by_query
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


FAKE_RESULTS = {
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


def fake_factory(alphas):
    return {
        f"alpha={alpha:.2f}": FakeRetriever(FAKE_RESULTS) for alpha in alphas
    }


def run_ablation(k=2, candidate_k=5, alphas=None, max_queries=None):
    return run_fusion_ablation(
        DATASET,
        k=k,
        candidate_k=candidate_k,
        alpha_values=alphas,
        dense_model="fake/dense",
        max_queries=max_queries,
        retriever_factory=lambda: fake_factory(alphas or ALPHA_GRID),
    )


# --- alpha parsing / validation ---


def test_default_alpha_grid():
    assert ALPHA_GRID == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_parse_alpha_values():
    assert parse_alpha_values("0.0,0.25,0.5,0.75,1.0") == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert parse_alpha_values("0, 0.5 ,1") == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("text", ["", "   ", "1.5", "abc", "0.5,-0.1"])
def test_parse_alpha_values_invalid_raises(text):
    with pytest.raises(ValueError):
        parse_alpha_values(text)


def test_validate_alpha_values_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        validate_alpha_values([])
    with pytest.raises(ValueError):
        validate_alpha_values([0.5, 1.1])
    with pytest.raises(ValueError):
        validate_alpha_values([-0.1])
    with pytest.raises(ValueError):
        validate_alpha_values(["0.5"])


def test_run_fusion_ablation_rejects_invalid_alphas():
    with pytest.raises(ValueError):
        run_ablation(alphas=[1.2])


# --- shared-index reuse ---


class FakeEncoder:
    """Deterministic encoder for the offline DenseRetriever."""

    def __init__(self, vectors):
        self.vectors = {
            text: np.asarray(vector, dtype=np.float64)
            for text, vector in vectors.items()
        }
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=np.float64)


def test_same_sparse_and_dense_reused_across_alphas():
    corpus = DATASET.corpus
    vectors = {doc["text"]: [1.0, 0.0] for doc in corpus}
    vectors["query one"] = [1.0, 0.0]
    encoder = FakeEncoder(vectors)

    retrievers = build_alpha_retrievers(corpus, [0.0, 0.5, 1.0], candidate_k=5, encoder=encoder)

    sparse_objects = {id(retriever.sparse_retriever) for retriever in retrievers.values()}
    dense_objects = {id(retriever.dense_retriever) for retriever in retrievers.values()}

    assert len(sparse_objects) == 1  # one BM25 index for every alpha
    assert len(dense_objects) == 1  # one DenseRetriever for every alpha

    # The dense corpus is encoded exactly once at build time.
    corpus_calls = [call for call in encoder.calls if len(call) == len(corpus)]
    assert corpus_calls == [[doc["text"] for doc in corpus]]


def test_alpha_values_wired_into_hybrid_retrievers():
    corpus = DATASET.corpus
    vectors = {doc["text"]: [1.0, 0.0] for doc in corpus}
    retrievers = build_alpha_retrievers(
        corpus, [0.0, 0.25, 1.0], candidate_k=5, encoder=FakeEncoder(vectors)
    )

    assert retrievers["alpha=0.00"].alpha == 0.0
    assert retrievers["alpha=0.25"].alpha == 0.25
    assert retrievers["alpha=1.00"].alpha == 1.0
    assert all(retriever.candidate_k == 5 for retriever in retrievers.values())


# --- ablation semantics ---


def test_same_query_set_and_order_for_all_alphas():
    retrievers = fake_factory([0.0, 0.5, 1.0])
    payload = run_fusion_ablation(
        DATASET,
        k=2,
        candidate_k=5,
        alpha_values=[0.0, 0.5, 1.0],
        retriever_factory=lambda: retrievers,
    )

    assert payload["metadata"]["num_queries"] == 2
    first_calls = retrievers["alpha=0.00"].calls
    for label in ("alpha=0.50", "alpha=1.00"):
        assert retrievers[label].calls == first_calls


def test_aggregation_matches_manual_metrics():
    payload = run_ablation(alphas=[0.0, 0.5, 1.0])

    expected = mean_metrics(
        [
            evaluate_query(["C2", "C1"], DATASET.qrels["q1"], k=2),
            evaluate_query(["C3"], DATASET.qrels["q2"], k=2),
        ]
    )
    for label in ("alpha=0.00", "alpha=0.50", "alpha=1.00"):
        assert payload["results"][label] == pytest.approx(expected)


def test_output_json_schema(tmp_path):
    payload = run_ablation(alphas=[0.0, 1.0])
    path = tmp_path / "ablation.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert set(loaded["metadata"]) == {
        "dataset",
        "source",
        "dense_model",
        "k",
        "candidate_k",
        "alpha_values",
        "num_queries",
        "max_queries",
    }
    assert loaded["metadata"]["alpha_values"] == [0.0, 1.0]
    assert loaded["metadata"]["num_queries"] == 2
    assert set(loaded["results"]) == {"alpha=0.00", "alpha=1.00"}
    assert set(loaded["results"]["alpha=0.00"]) == {
        "precision@2",
        "recall@2",
        "mrr@2",
        "ndcg@2",
    }


def test_smoke_and_full_output_paths_do_not_collide():
    smoke = output_path_for(20)
    full = output_path_for(None)

    assert smoke == "outputs/scifact_fusion_ablation_smoke20.json"
    assert full == "outputs/scifact_fusion_ablation.json"
    assert smoke != full
    for path in (smoke, full):
        assert path not in (
            "outputs/scifact_benchmark.json",
            "outputs/scifact_latency.json",
            "outputs/scifact_reranker_benchmark.json",
        )


def test_format_ablation_table_shows_alpha_column():
    payload = run_ablation(alphas=[0.0, 0.5, 1.0])

    table = format_ablation_table(payload)

    for column in ("Alpha", "Precision@2", "Recall@2", "MRR@2", "nDCG@2"):
        assert column in table
    for alpha in ("0.00", "0.50", "1.00"):
        assert alpha in table


# --- plotting ---


def test_extract_curve_uses_payload_values_not_hardcoded():
    first = run_ablation(alphas=[0.0, 0.5, 1.0])
    alphas, values = extract_curve(first, "ndcg@2")

    assert alphas == [0.0, 0.5, 1.0]
    assert values == [
        first["results"]["alpha=0.00"]["ndcg@2"],
        first["results"]["alpha=0.50"]["ndcg@2"],
        first["results"]["alpha=1.00"]["ndcg@2"],
    ]

    # A payload with different results produces a different curve.
    modified = dict(first)
    modified["results"] = {
        "alpha=0.00": {"ndcg@2": 0.111},
        "alpha=0.50": {"ndcg@2": 0.222},
        "alpha=1.00": {"ndcg@2": 0.333},
    }
    _, modified_values = extract_curve(modified, "ndcg@2")
    assert modified_values == [0.111, 0.222, 0.333]
    assert modified_values != values


def test_plot_generation_writes_pngs_from_payload(tmp_path):
    payload = run_ablation(alphas=[0.0, 0.25, 0.5, 0.75, 1.0])

    paths = plot_ablation(payload, output_dir=tmp_path)

    names = [path.name for path in paths]
    assert set(names) == {
        "fusion_alpha_precision.png",
        "fusion_alpha_recall.png",
        "fusion_alpha_mrr.png",
        "fusion_alpha_ndcg.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


# --- determinism ---


def test_ablation_is_deterministic():
    first = run_ablation(alphas=[0.0, 0.5, 1.0])
    second = run_ablation(alphas=[0.0, 0.5, 1.0])

    assert first == second
