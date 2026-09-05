"""Offline tests for the FAISS index benchmark harness (Phase 14A).

A synthetic ``FiQADataset``, a fake encoder, and a fake timer drive every
test: no network, no real models, deterministic results. The FAISS
library itself is exercised (it is a local dependency, not a download).
"""

import json

import numpy as np
import pytest

from experiments.benchmark import save_results
from experiments.faiss_index_benchmark import (
    output_path_for,
    parse_args,
    resolve_indexes,
    run_faiss_index_benchmark,
)
from hybridsearch.data.fiqa import FiQADataset

DATASET = FiQADataset(
    corpus=[
        {"id": "C1", "text": "alpha"},
        {"id": "C2", "text": "beta"},
        {"id": "C3", "text": "gamma"},
        {"id": "C4", "text": "delta"},
    ],
    queries={
        "q1": "query one",
        "q2": "query two",
        "q3": "query three",  # no qrels -> excluded
    },
    qrels={
        "q1": {"C1": 1},
        "q2": {"C2": 1},
    },
)

VECTORS = {
    "alpha": [2, 0, 0],
    "beta": [0, 4, 0],
    "gamma": [0, 0, 6],
    "delta": [1, 1, 0],
    "query one": [2, 0, 0],
    "query two": [0, 4, 0],
    "query three": [3, 4, 0],
}


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=np.float64)


class FakeTimer:
    """Advances by 5 ms per call: every timed step costs exactly 5 ms."""

    def __init__(self):
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        current = self.now
        self.now += 0.005
        return current


def run_harness(index_specs=None, k=2, max_queries=None, warmup=0, timer=None):
    encoder = FakeEncoder(VECTORS)
    timer = timer or FakeTimer()
    payload = run_faiss_index_benchmark(
        DATASET,
        k=k,
        max_queries=max_queries,
        dense_model="fake/dense",
        index_specs=index_specs,
        warmup_queries=warmup,
        encoder=encoder,
        timer=timer,
    )
    return payload, encoder, timer


DEFAULT_SPECS = [
    ("flat", {}),
    ("hnsw", {"M": 16}),
    ("ivf", {"nlist": 4, "nprobe": 4}),
]


# --- CLI parsing ---


def test_resolve_indexes_parses_and_deduplicates():
    assert resolve_indexes("flat,hnsw,ivf") == ["flat", "hnsw", "ivf"]
    assert resolve_indexes(" flat , hnsw,flat ") == ["flat", "hnsw"]


@pytest.mark.parametrize("text", ["", "   ", "pizza", "flat,pizza", ",,,"])
def test_resolve_indexes_invalid_raises(text):
    with pytest.raises(ValueError):
        resolve_indexes(text)


def test_parse_args_defaults_and_overrides():
    args = parse_args(["--dataset", "fiqa", "--indexes", "flat,ivf", "--k", "5"])
    assert args.dataset == "fiqa"
    assert args.indexes == "flat,ivf"
    assert args.k == 5

    defaults = parse_args([])
    assert defaults.indexes == "flat,hnsw,ivf"
    assert defaults.hnsw_m == 32
    assert defaults.ivf_nlist == 512
    assert defaults.ivf_nprobe == 32


def test_parse_args_rejects_unknown_dataset_and_index():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "nope"])
    # The index list is validated later (after arg parsing) for a clear error.
    with pytest.raises(ValueError):
        resolve_indexes(parse_args(["--indexes", "pizza"]).indexes)


# --- payload schema ---


def test_run_payload_schema():
    payload, _, _ = run_harness(DEFAULT_SPECS)

    metadata = payload["metadata"]
    assert metadata["dataset"] == "fiqa"
    assert metadata["source"] == "BeIR/fiqa"
    assert metadata["dense_model"] == "fake/dense"
    assert metadata["k"] == 2
    assert metadata["num_queries"] == 2
    assert metadata["max_queries"] is None
    assert metadata["indexes"] == [
        {"index_type": "flat", "params": {}},
        {"index_type": "hnsw", "params": {"M": 16}},
        {"index_type": "ivf", "params": {"nlist": 4, "nprobe": 4}},
    ]
    assert isinstance(metadata["timing_methodology"], list)

    assert set(payload["results"]) == {"flat", "hnsw", "ivf"}
    for entry in payload["results"].values():
        assert set(entry) == {
            "index_type", "params", "dimension", "n_vectors",
            "requires_training", "trained", "build_time_s",
            "serialized_size_bytes", "ann_recall_at_k", "latency",
        }
        assert entry["dimension"] == 3
        assert entry["n_vectors"] == 4
        assert entry["serialized_size_bytes"] > 0
        assert set(entry["latency"]) == {
            "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "samples",
        }
        assert entry["latency"]["samples"] == 2


def test_training_flags_per_index_type():
    payload, _, _ = run_harness(DEFAULT_SPECS)

    assert payload["results"]["flat"]["requires_training"] is False
    assert payload["results"]["hnsw"]["requires_training"] is False
    assert payload["results"]["ivf"]["requires_training"] is True
    assert all(entry["trained"] for entry in payload["results"].values())


def test_ann_recall_against_flat_reference():
    payload, _, _ = run_harness(DEFAULT_SPECS)

    # flat vs flat = 1.0 by construction; the tiny full-probe indexes here
    # are exact too, but only flat's value is asserted as a guarantee.
    assert payload["results"]["flat"]["ann_recall_at_k"] == pytest.approx(1.0)
    assert 0.0 <= payload["results"]["hnsw"]["ann_recall_at_k"] <= 1.0
    assert 0.0 <= payload["results"]["ivf"]["ann_recall_at_k"] <= 1.0


def test_flat_reference_built_when_not_requested():
    payload, _, _ = run_harness([("ivf", {"nlist": 4, "nprobe": 4})])

    assert set(payload["results"]) == {"ivf"}
    # Recall is still measured against a hidden exact Flat reference.
    assert payload["results"]["ivf"]["ann_recall_at_k"] == pytest.approx(1.0)


# --- timing semantics ---


def test_build_time_and_latency_use_fake_timer():
    payload, _, timer = run_harness(DEFAULT_SPECS)

    # One build per index: each build = exactly two timer calls (5 ms).
    for entry in payload["results"].values():
        assert entry["build_time_s"] == pytest.approx(0.005)
        # Two timed queries at 5 ms each.
        assert entry["latency"]["mean_ms"] == pytest.approx(5.0)
        assert entry["latency"]["p95_ms"] == pytest.approx(5.0)

    # 3 builds * 2 calls + 3 indexes * 2 queries * 2 calls = 18 timer calls.
    assert timer.calls == 18


def test_encoding_is_excluded_from_timing():
    payload, encoder, _ = run_harness(DEFAULT_SPECS)

    # Encoder ran exactly twice: corpus once, all queries once — before the
    # first timer call, so its work never lands inside a timed region.
    assert encoder.calls == [
        ["alpha", "beta", "gamma", "delta"],
        ["query one", "query two"],
    ]
    for entry in payload["results"].values():
        assert entry["build_time_s"] == pytest.approx(0.005)


def test_warmup_queries_are_untimed():
    payload, _, timer = run_harness(DEFAULT_SPECS, warmup=1)

    # Warmup searches add no timer calls: same 18 calls as without warmup.
    assert timer.calls == 18
    assert payload["metadata"]["warmup_queries"] == 1


# --- output paths + JSON round trip ---


def test_output_paths_dataset_scoped():
    assert output_path_for("fiqa") == "outputs/fiqa_faiss_index_benchmark.json"
    assert (
        output_path_for("fiqa", 20)
        == "outputs/fiqa_faiss_index_benchmark_smoke20.json"
    )
    assert (
        output_path_for("scifact", 20)
        == "outputs/scifact_faiss_index_benchmark_smoke20.json"
    )
    assert output_path_for("fiqa") != output_path_for("scifact")
    assert output_path_for("fiqa", 20) != output_path_for("fiqa")


def test_output_json_schema(tmp_path):
    payload, _, _ = run_harness(DEFAULT_SPECS)
    path = tmp_path / "faiss_index_benchmark.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert set(loaded["results"]) == {"flat", "hnsw", "ivf"}
    assert loaded["metadata"]["num_queries"] == 2


# --- determinism ---


def test_harness_is_deterministic():
    first, _, _ = run_harness(DEFAULT_SPECS)
    second, _, _ = run_harness(DEFAULT_SPECS)

    assert first == second