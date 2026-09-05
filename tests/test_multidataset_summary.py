"""Offline tests for the multi-dataset summary script (Phase 13).

Synthetic payloads drive every test; the script is exercised exactly as
the CLI would exercise it (JSON round-trips + figure rendering with the
Agg backend), but nothing is downloaded and no model is loaded.
"""

import json

import pytest

from experiments.multidataset_summary import (
    BENCHMARK_METHOD_ORDER,
    best_alpha,
    build_summary,
    format_summary_table,
    run_summary,
)


def make_benchmark_payload(dataset, scale=1.0, k=10):
    results = {}
    for index, method in enumerate(BENCHMARK_METHOD_ORDER):
        value = 0.4 + index * 0.1 * scale
        results[method] = {
            f"precision@{k}": value,
            f"recall@{k}": value + 0.01,
            f"mrr@{k}": value + 0.02,
            f"ndcg@{k}": value + 0.03,
        }
    return {
        "metadata": {
            "dataset": dataset,
            "source": f"fake/{dataset}",
            "k": k,
            "num_queries": 3,
            "max_queries": None,
        },
        "results": results,
    }


def make_ablation_payload(dataset, best, k=10):
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    for alpha in alphas:
        value = 0.6 - abs(alpha - best) / 4.0
        results[f"alpha={alpha:.2f}"] = {
            f"precision@{k}": value,
            f"recall@{k}": value,
            f"mrr@{k}": value,
            f"ndcg@{k}": value,
        }
    return {
        "metadata": {
            "dataset": dataset,
            "source": f"fake/{dataset}",
            "k": k,
            "alpha_values": alphas,
            "num_queries": 3,
            "max_queries": None,
        },
        "results": results,
    }


# --- summary construction ---


def test_cross_dataset_comparison_structure():
    summary = build_summary(
        [make_benchmark_payload("scifact"), make_benchmark_payload("fiqa", 2.0)],
        [
            make_ablation_payload("scifact", 0.25),
            make_ablation_payload("fiqa", 0.75),
        ],
    )

    assert set(summary["comparison"]) == {"scifact", "fiqa"}
    for methods in summary["comparison"].values():
        assert list(methods) == BENCHMARK_METHOD_ORDER

    # Values come straight from the input payloads (never hard-coded).
    fiqa_payload = make_benchmark_payload("fiqa", 2.0)
    assert summary["comparison"]["fiqa"]["BM25"]["ndcg@10"] == pytest.approx(
        fiqa_payload["results"]["BM25"]["ndcg@10"]
    )


def test_comparison_reflects_payload_changes():
    first = build_summary(
        [make_benchmark_payload("fiqa", 1.0)], []
    )["comparison"]["fiqa"]["Dense"]["ndcg@10"]
    second = build_summary(
        [make_benchmark_payload("fiqa", 3.0)], []
    )["comparison"]["fiqa"]["Dense"]["ndcg@10"]

    assert second != first
    assert second == pytest.approx(make_benchmark_payload("fiqa", 3.0)["results"]["Dense"]["ndcg@10"])


def test_best_alpha_comes_from_payloads():
    assert best_alpha(make_ablation_payload("scifact", 0.25)) == 0.25
    assert best_alpha(make_ablation_payload("fiqa", 0.75)) == 0.75

    summary = build_summary(
        [make_benchmark_payload("fiqa")],
        [make_ablation_payload("fiqa", 0.75)],
    )
    assert summary["best_alpha"] == {"fiqa": 0.75}


def test_best_alpha_is_none_without_ablations():
    summary = build_summary([make_benchmark_payload("fiqa")], [])
    assert summary["best_alpha"] == {}
    assert summary["alpha_curves"] == {}


# --- JSON round-trip (the CLI path) ---


def test_summary_reads_actual_json_payloads(tmp_path):
    benchmark_dir = tmp_path / "benchmarks"
    ablation_dir = tmp_path / "ablations"
    benchmark_dir.mkdir()
    ablation_dir.mkdir()

    benchmark_paths = []
    for dataset, scale in [("scifact", 1.0), ("fiqa", 2.0)]:
        path = benchmark_dir / f"{dataset}.json"
        path.write_text(
            json.dumps(make_benchmark_payload(dataset, scale)), encoding="utf-8"
        )
        benchmark_paths.append(str(path))

    ablation_paths = []
    for dataset, best in [("scifact", 0.25), ("fiqa", 0.75)]:
        path = ablation_dir / f"{dataset}.json"
        path.write_text(
            json.dumps(make_ablation_payload(dataset, best)), encoding="utf-8"
        )
        ablation_paths.append(str(path))

    output = tmp_path / "summary.json"
    figures = tmp_path / "figures"
    summary, figure_paths = run_summary(
        benchmark_paths,
        ablation_paths=ablation_paths,
        output_json=str(output),
        figures_dir=str(figures),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == summary
    assert set(loaded["comparison"]) == {"scifact", "fiqa"}
    assert loaded["best_alpha"] == {"scifact": 0.25, "fiqa": 0.75}

    sources = loaded["metadata"]["benchmark_sources"]
    assert {entry["dataset"] for entry in sources} == {"scifact", "fiqa"}
    for entry in sources:
        assert entry["num_queries"] == 3
        assert entry["max_queries"] is None

    assert {path.name for path in figure_paths} == {
        "multidataset_ndcg.png",
        "multidataset_best_alpha.png",
    }
    for path in figure_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_summary_without_ablations_skips_best_alpha_figure(tmp_path):
    path = tmp_path / "fiqa.json"
    path.write_text(json.dumps(make_benchmark_payload("fiqa")), encoding="utf-8")

    summary, figure_paths = run_summary(
        [str(path)],
        output_json=str(tmp_path / "summary.json"),
        figures_dir=str(tmp_path / "figures"),
    )

    assert summary["best_alpha"] == {}
    assert [p.name for p in figure_paths] == ["multidataset_ndcg.png"]


def test_format_table_contains_datasets_and_methods():
    summary = build_summary(
        [make_benchmark_payload("scifact"), make_benchmark_payload("fiqa")],
        [make_ablation_payload("scifact", 0.25)],
    )

    table = format_summary_table(summary)

    for token in ("Dataset", "Method", "scifact", "fiqa", "BM25", "Dense",
                  "Hybrid Weighted", "Hybrid RRF", "ndcg@10", "Best alpha"):
        assert token in table


# --- determinism ---


def test_summary_is_deterministic():
    first = build_summary(
        [make_benchmark_payload("scifact"), make_benchmark_payload("fiqa")],
        [make_ablation_payload("scifact", 0.25)],
    )
    second = build_summary(
        [make_benchmark_payload("scifact"), make_benchmark_payload("fiqa")],
        [make_ablation_payload("scifact", 0.25)],
    )

    assert first == second