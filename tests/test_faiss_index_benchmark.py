"""Offline tests for the Phase 14B FAISS index study harness.

A synthetic ``FiQADataset``, a fake encoder, and a fake timer drive every
test: no network, no real models, deterministic results. The FAISS
library itself is exercised (it is a local dependency, not a download).
"""

import json

import numpy as np
import pytest

from experiments.benchmark import save_results
from experiments.faiss_index_benchmark import (
    build_grid,
    default_grid,
    output_path_for,
    parse_args,
    parse_int_list,
    plot_all,
    prepare_hnsw_curve,
    prepare_ivf_curves,
    prepare_size_data,
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
    "delta": [1, 1, 0],  # tie-free with respect to every query
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


TEST_GRID = [
    ("flat", "flat", {}),
    ("hnsw_efS8", "hnsw", {"M": 16, "efConstruction": 50, "efSearch": 8}),
    ("hnsw_efS16", "hnsw", {"M": 16, "efConstruction": 50, "efSearch": 16}),
    ("ivf_n4_p2", "ivf", {"nlist": 4, "nprobe": 2}),
    ("ivf_n4_p4", "ivf", {"nlist": 4, "nprobe": 4}),
]


def run_harness(grid=None, k=2, max_queries=None, warmup=1, timed=2,
                relevance=True, timer=None):
    encoder = FakeEncoder(VECTORS)
    timer = timer or FakeTimer()
    payload = run_faiss_index_benchmark(
        DATASET,
        k=k,
        max_queries=max_queries,
        dense_model="fake/dense",
        grid=grid if grid is not None else TEST_GRID,
        warmup_passes=warmup,
        timed_passes=timed,
        relevance_check=relevance,
        encoder=encoder,
        timer=timer,
    )
    return payload, encoder, timer


# --- grid construction ---


def test_default_grid_shape_and_order():
    points, excluded = default_grid()

    labels = [label for label, _, _ in points]
    assert labels[0] == "flat"
    assert labels[1:5] == ["hnsw_efS16", "hnsw_efS32", "hnsw_efS64", "hnsw_efS128"]
    ivf_labels = labels[5:]
    assert len(ivf_labels) == 12
    assert ivf_labels[0] == "ivf_n256_p8"
    assert ivf_labels[-1] == "ivf_n1024_p64"
    assert len(points) == 1 + 4 + 12
    assert excluded == []


def test_build_grid_excludes_nprobe_above_nlist():
    points, excluded = build_grid(
        hnsw_m=32,
        ef_construction=200,
        ef_search_values=[64],
        ivf_nlist_values=[256],
        ivf_nprobe_values=[8, 300],
    )

    labels = [label for label, _, _ in points]
    assert "ivf_n256_p8" in labels
    assert "ivf_n256_p300" not in labels
    assert excluded == [
        {"nlist": 256, "nprobe": 300, "reason": "nprobe > nlist"}
    ]


def test_build_grid_rejects_invalid_hnsw_values():
    with pytest.raises(ValueError):
        build_grid(hnsw_m=0, ef_construction=200, ef_search_values=[64])


def test_parse_int_list():
    assert parse_int_list("16,32,64", "efSearch") == [16, 32, 64]
    assert parse_int_list(" 8 , 16, 8 ", "nprobe") == [8, 16]


@pytest.mark.parametrize("text", ["", "   ", "abc", "0,16", "16,-2", ",,,"])
def test_parse_int_list_invalid_raises(text):
    with pytest.raises(ValueError):
        parse_int_list(text, "values")


# --- payload schema and semantics ---


def test_run_payload_schema():
    payload, _, _ = run_harness()

    metadata = payload["metadata"]
    assert metadata["dataset"] == "fiqa"
    assert metadata["source"] == "BeIR/fiqa"
    assert metadata["corpus_size"] == 4
    assert metadata["num_queries"] == 2
    assert metadata["embedding_dimension"] == 3
    assert metadata["k"] == 2
    assert metadata["warmup_passes"] == 1
    assert metadata["timed_passes"] == 2
    assert metadata["samples_per_query"] == 2
    assert metadata["relevance_check"] is True
    assert metadata["grid"] == [
        {"label": label, "index_type": index_type, "params": params}
        for label, index_type, params in TEST_GRID
    ]
    assert isinstance(metadata["timing_methodology"], list)

    assert set(payload["results"]) == {label for label, _, _ in TEST_GRID}
    for entry in payload["results"].values():
        assert set(entry) == {
            "label", "index_type", "params", "effective_params", "dimension",
            "n_vectors", "requires_training", "trained", "build_time_s",
            "serialized_size_bytes", "ann_recall", "latency",
            "qrel_ndcg_at_k",
        }
        assert entry["dimension"] == 3
        assert entry["n_vectors"] == 4
        assert entry["serialized_size_bytes"] > 0
        assert set(entry["ann_recall"]) == {
            "mean", "min", "queries_with_recall_1_0", "queries_with_recall_ge_0_9",
        }
        assert set(entry["latency"]) == {
            "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "samples",
        }
        assert entry["latency"]["samples"] == 2  # per-query aggregates


def test_effective_params_match_requested():
    payload, _, _ = run_harness()

    assert payload["results"]["flat"]["effective_params"] == {}
    assert payload["results"]["hnsw_efS8"]["effective_params"]["efSearch"] == 8
    assert payload["results"]["hnsw_efS16"]["effective_params"]["efSearch"] == 16
    assert payload["results"]["ivf_n4_p2"]["effective_params"]["nprobe"] == 2
    assert payload["results"]["ivf_n4_p4"]["effective_params"]["nprobe"] == 4
    assert payload["results"]["ivf_n4_p2"]["effective_params"]["nlist"] == 4


def test_flat_recall_is_exactly_one():
    payload, _, _ = run_harness()

    flat = payload["results"]["flat"]
    assert flat["ann_recall"]["mean"] == pytest.approx(1.0)
    assert flat["ann_recall"]["min"] == pytest.approx(1.0)
    assert flat["ann_recall"]["queries_with_recall_1_0"] == 2
    assert flat["ann_recall"]["queries_with_recall_ge_0_9"] == 2


def test_ann_recall_distribution_stats():
    payload, _, _ = run_harness()

    # Full-probe IVF and the tiny exact HNSW graphs are exact here.
    assert payload["results"]["ivf_n4_p4"]["ann_recall"]["mean"] == pytest.approx(1.0)
    assert payload["results"]["hnsw_efS8"]["ann_recall"]["mean"] == pytest.approx(1.0)
    # Partial probe stays within [0, 1] and reports distribution stats.
    partial = payload["results"]["ivf_n4_p2"]["ann_recall"]
    assert 0.0 <= partial["mean"] <= 1.0
    assert 0.0 <= partial["min"] <= partial["mean"]
    assert partial["queries_with_recall_1_0"] <= 2
    assert partial["queries_with_recall_ge_0_9"] <= 2


def test_builds_are_grouped_by_build_configuration():
    payload, _, timer = run_harness()

    # Same index for both efSearch values, and for both nprobe values.
    assert payload["results"]["hnsw_efS8"]["build_time_s"] == pytest.approx(
        payload["results"]["hnsw_efS16"]["build_time_s"]
    )
    assert payload["results"]["ivf_n4_p2"]["build_time_s"] == pytest.approx(
        payload["results"]["ivf_n4_p4"]["build_time_s"]
    )
    assert payload["results"]["ivf_n4_p2"]["serialized_size_bytes"] == (
        payload["results"]["ivf_n4_p4"]["serialized_size_bytes"]
    )

    # 3 distinct builds (flat, hnsw, ivf) x 2 timer calls each = 6; plus
    # 5 configs x 2 queries x 2 timed passes x 2 calls = 40 -> 46 total.
    assert timer.calls == 46


def test_encoding_is_excluded_from_all_timing():
    payload, encoder, _ = run_harness()

    assert encoder.calls == [
        ["alpha", "beta", "gamma", "delta"],
        ["query one", "query two"],
    ]
    for entry in payload["results"].values():
        assert entry["build_time_s"] == pytest.approx(0.005)
        assert entry["latency"]["mean_ms"] == pytest.approx(5.0)


def test_per_query_latency_averages_timed_passes():
    payload, _, timer = run_harness(timed=3)

    # 3 timed passes per query: per-query mean = 5 ms (three 5 ms samples).
    for entry in payload["results"].values():
        assert entry["latency"]["mean_ms"] == pytest.approx(5.0)
        assert entry["latency"]["samples"] == 2  # per-query aggregates

    # 3 builds * 2 + 5 configs * 2 queries * 3 passes * 2 = 66 timer calls.
    assert timer.calls == 66


def test_warmup_passes_do_not_touch_timer():
    payload, _, timer = run_harness(warmup=2, timed=1)

    assert payload["metadata"]["warmup_passes"] == 2
    # 3 builds * 2 + 5 configs * 2 queries * 1 pass * 2 = 26 timer calls.
    assert timer.calls == 26


def test_relevance_diagnostic_optional():
    with_relevance, _, _ = run_harness(relevance=True)
    without_relevance, _, _ = run_harness(relevance=False)

    assert with_relevance["metadata"]["relevance_check"] is True
    assert with_relevance["results"]["flat"]["qrel_ndcg_at_k"] == pytest.approx(1.0)
    assert without_relevance["results"]["flat"]["qrel_ndcg_at_k"] is None


def test_grid_preflight_rejects_ivf_nlist_above_corpus():
    bad_grid = [("ivf_big", "ivf", {"nlist": 16, "nprobe": 4})]

    with pytest.raises(ValueError, match="training"):
        run_harness(grid=bad_grid)


def test_harness_rejects_invalid_pass_counts():
    with pytest.raises(ValueError):
        run_harness(timed=0)
    with pytest.raises(ValueError):
        run_harness(warmup=-1)


# --- figure data preparation ---


def test_prepare_hnsw_curve_reads_payload():
    payload, _, _ = run_harness()

    ef_searches, recalls, latencies = prepare_hnsw_curve(payload)

    assert ef_searches == [8, 16]
    assert recalls == [
        payload["results"]["hnsw_efS8"]["ann_recall"]["mean"],
        payload["results"]["hnsw_efS16"]["ann_recall"]["mean"],
    ]
    assert latencies == pytest.approx([5.0, 5.0])

    # A changed payload produces a changed curve (nothing hard-coded).
    modified = json.loads(json.dumps(payload))
    modified["results"]["hnsw_efS8"]["ann_recall"]["mean"] = 0.5
    _, modified_recalls, _ = prepare_hnsw_curve(modified)
    assert modified_recalls[0] == pytest.approx(0.5)
    assert modified_recalls != recalls


def test_prepare_ivf_curves_groups_by_nlist():
    grid = [
        ("flat", "flat", {}),
        ("ivf_n2_p2", "ivf", {"nlist": 2, "nprobe": 2}),
        ("ivf_n4_p2", "ivf", {"nlist": 4, "nprobe": 2}),
        ("ivf_n4_p4", "ivf", {"nlist": 4, "nprobe": 4}),
    ]
    payload, _, _ = run_harness(grid=grid)

    curves = prepare_ivf_curves(payload)

    assert set(curves) == {2, 4}
    assert curves[2][0] == [2]
    assert curves[4][0] == [2, 4]
    for nprobes, recalls, latencies in curves.values():
        assert len(nprobes) == len(recalls) == len(latencies)


def test_prepare_size_data_reads_payload():
    payload, _, _ = run_harness()

    labels, sizes = prepare_size_data(payload)

    assert labels == [label for label, _, _ in TEST_GRID]
    assert all(size > 0 for size in sizes)
    assert sizes[0] == pytest.approx(
        payload["results"]["flat"]["serialized_size_bytes"] / 1e6
    )


def test_plot_all_writes_expected_pngs(tmp_path):
    payload, _, _ = run_harness()

    paths = plot_all(payload, output_dir=tmp_path)

    assert {path.name for path in paths} == {
        "faiss_recall_latency.png",
        "faiss_index_size.png",
        "faiss_hnsw_efsearch.png",
        "faiss_ivf_nprobe.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


# --- output paths + JSON round trip ---


def test_output_paths_dataset_scoped():
    assert output_path_for("fiqa") == "outputs/fiqa_faiss_index_benchmark.json"
    assert (
        output_path_for("fiqa", 20)
        == "outputs/fiqa_faiss_index_benchmark_smoke20.json"
    )
    assert output_path_for("fiqa") != output_path_for("scifact")
    assert output_path_for("fiqa", 20) != output_path_for("fiqa")


def test_output_json_schema(tmp_path):
    payload, _, _ = run_harness()
    path = tmp_path / "faiss_index_benchmark.json"

    save_results(payload, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"metadata", "results"}
    assert set(loaded["results"]) == {"flat", "hnsw_efS8", "hnsw_efS16", "ivf_n4_p2", "ivf_n4_p4"}
    assert loaded["metadata"]["num_queries"] == 2


# --- CLI ---


def test_parse_args_defaults_and_overrides():
    defaults = parse_args([])
    assert defaults.hnsw_ef_search == "16,32,64,128"
    assert defaults.ivf_nlist == "256,512,1024"
    assert defaults.ivf_nprobe == "8,16,32,64"
    assert defaults.warmup_passes == 3
    assert defaults.timed_passes == 5

    args = parse_args(
        ["--dataset", "fiqa", "--timed-passes", "2", "--skip-relevance"]
    )
    assert args.dataset == "fiqa"
    assert args.timed_passes == 2
    assert args.skip_relevance is True


def test_parse_args_rejects_unknown_dataset():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "nope"])


# --- determinism ---


def test_harness_is_deterministic():
    first, _, _ = run_harness()
    second, _, _ = run_harness()

    assert first == second