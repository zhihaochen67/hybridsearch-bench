"""Cross-dataset summary of saved benchmark / ablation JSON payloads.

    python experiments/multidataset_summary.py \
        --benchmarks outputs/scifact_benchmark.json outputs/fiqa_benchmark.json \
        --ablations outputs/scifact_fusion_ablation.json \
                    outputs/fiqa_fusion_ablation.json

Consumes previously saved experiment payloads — it never reruns
retrieval or loads a dataset/model. It prints a concise cross-dataset
table, writes ``outputs/multidataset_summary.json``, and generates:

- ``assets/figures/multidataset_ndcg.png`` — grouped bars of nDCG@k for
  BM25 / Dense / Hybrid Weighted / Hybrid RRF, grouped by dataset;
- ``assets/figures/multidataset_best_alpha.png`` — alpha vs nDCG@k for
  every dataset on the same axes, with each best alpha marked.

All numbers and curves are read from the input payloads; nothing is
hard-coded.

Scope note: only the four retrieval methods are compared — reranker rows
in benchmark payloads are ignored. A fair grouped comparison assumes the
payloads share the same hybrid fusion candidate pool; each payload's
metadata records its own ``candidate_k`` / ``hybrid_candidate_k``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/multidataset_summary.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BENCHMARK_METHOD_ORDER = ["BM25", "Dense", "Hybrid Weighted", "Hybrid RRF"]
METRIC_PREFIX_ORDER = ("precision", "recall", "mrr", "ndcg")


def load_payload(path) -> dict:
    """Read one saved experiment JSON payload."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def best_alpha(payload: dict) -> float | None:
    """Alpha with the highest nDCG@k in an ablation payload (or ``None``).

    Ties resolve to the first alpha in the saved grid, which keeps the
    result deterministic.
    """
    alphas = payload["metadata"]["alpha_values"]
    k = payload["metadata"]["k"]
    values = [
        payload["results"][f"alpha={alpha:.2f}"][f"ndcg@{k}"] for alpha in alphas
    ]
    if not alphas:
        return None
    return alphas[max(range(len(values)), key=values.__getitem__)]


def alpha_curve(payload: dict) -> tuple[list[float], list[float]]:
    """(alphas, nDCG@k values) straight from an ablation payload."""
    alphas = list(payload["metadata"]["alpha_values"])
    k = payload["metadata"]["k"]
    values = [
        payload["results"][f"alpha={alpha:.2f}"][f"ndcg@{k}"] for alpha in alphas
    ]
    return alphas, values


def _source_entry(path, payload: dict) -> dict:
    """Small provenance block for one input payload."""
    metadata = payload["metadata"]
    return {
        "path": str(path) if path is not None else None,
        "dataset": metadata.get("dataset"),
        "source": metadata.get("source"),
        "k": metadata.get("k"),
        "num_queries": metadata.get("num_queries"),
        "max_queries": metadata.get("max_queries"),
    }


def _normalize_items(items):
    """Yield ``(path, payload)`` pairs from payloads or ``(path, payload)``."""
    for item in items:
        if isinstance(item, tuple):
            path, payload = item
        else:
            path, payload = None, item
        yield path, payload


def build_summary(benchmark_payloads: list[dict], ablation_payloads: list[dict] | None = None) -> dict:
    """Build the cross-dataset summary payload from loaded experiment JSONs.

    Inputs may be plain payload dicts or ``(path, payload)`` pairs.
    ``comparison`` maps dataset -> method -> metric means for the four
    retrieval methods; ``alpha_curves`` and ``best_alpha`` come from the
    fusion-ablation payloads (best alpha is ``None`` when no ablation is
    available for a dataset). Nothing here is hard-coded.
    """
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    benchmark_sources = []
    for path, payload in _normalize_items(benchmark_payloads):
        dataset = payload["metadata"]["dataset"]
        methods = {}
        for method in BENCHMARK_METHOD_ORDER:
            if method in payload["results"]:
                methods[method] = payload["results"][method]
        comparison[dataset] = methods
        benchmark_sources.append(_source_entry(path, payload))

    alpha_curves: dict[str, dict[str, list]] = {}
    best_alphas: dict[str, float | None] = {}
    ablation_sources = []
    for path, payload in _normalize_items(ablation_payloads or []):
        dataset = payload["metadata"]["dataset"]
        alphas, values = alpha_curve(payload)
        alpha_curves[dataset] = {
            "alphas": alphas,
            f"ndcg@{payload['metadata']['k']}": values,
        }
        best_alphas[dataset] = best_alpha(payload)
        ablation_sources.append(_source_entry(path, payload))

    return {
        "metadata": {
            "benchmark_sources": benchmark_sources,
            "ablation_sources": ablation_sources,
        },
        "comparison": comparison,
        "alpha_curves": alpha_curves,
        "best_alpha": best_alphas,
    }


def format_summary_table(summary: dict) -> str:
    """Render the cross-dataset summary as a fixed-width text table.

    The metric columns are the union of the metrics present in the
    comparison, in the stable order precision / recall / mrr / ndcg.
    """
    metric_names: list[str] = []
    for methods in summary["comparison"].values():
        for means in methods.values():
            for name in means:
                if name not in metric_names:
                    metric_names.append(name)
    order = {prefix: index for index, prefix in enumerate(METRIC_PREFIX_ORDER)}
    metric_names.sort(
        key=lambda name: order.get(name.rsplit("@", 1)[0], len(order))
    )

    header = ["Dataset", "Method", *metric_names]
    rows = []
    for dataset in summary["comparison"]:
        for method, means in summary["comparison"][dataset].items():
            rows.append(
                [dataset, method]
                + [
                    f"{means.get(metric, float('nan')):.4f}"
                    for metric in metric_names
                ]
            )

    widths = [
        max(len(str(cell)) for cell in column_cells)
        for column_cells in zip(header, *rows)
    ]
    lines = [
        "  ".join(str(cell).ljust(width) for cell, width in zip(header, widths)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths))
        )

    if summary["best_alpha"]:
        lines.append("")
        lines.append("Best alpha (by nDCG):")
        for dataset, alpha in summary["best_alpha"].items():
            curve = summary["alpha_curves"].get(dataset)
            detail = ""
            if curve is not None and alpha is not None:
                ndcg_key = next(
                    name for name in curve if name.startswith("ndcg@")
                )
                index = curve["alphas"].index(alpha)
                detail = f"  (nDCG@{ndcg_key.removeprefix('ndcg@')} = {curve[ndcg_key][index]:.4f})"
            lines.append(f"  {dataset}: alpha = {alpha}{detail}")

    return "\n".join(lines)


def plot_multidataset_ndcg(summary: dict, output_path) -> Path:
    """Grouped bars of nDCG@k per method, grouped by dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(summary["comparison"])
    methods = [
        method
        for method in BENCHMARK_METHOD_ORDER
        if any(method in summary["comparison"][dataset] for dataset in datasets)
    ]

    ks = {
        source["dataset"]: source["k"]
        for source in summary["metadata"]["benchmark_sources"]
        if source["dataset"]
    }
    metric_label = (
        f"nDCG@{ks[datasets[0]]}"
        if datasets and len({ks.get(d) for d in datasets}) == 1
        else "nDCG@k"
    )

    x = list(range(len(methods)))
    width = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for offset, dataset in enumerate(datasets):
        values = []
        for method in methods:
            entry = summary["comparison"].get(dataset, {}).get(method)
            if entry is None:
                values.append(0.0)
                continue
            key = next((name for name in entry if name.startswith("ndcg@")), None)
            values.append(entry.get(key, 0.0) if key else 0.0)
        ax.bar(
            [position + offset * width for position in x],
            values,
            width=width,
            label=dataset,
        )

    ax.set_xticks([position + width * (len(datasets) - 1) / 2 for position in x])
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title("Multi-dataset retrieval quality (nDCG)")
    ax.set_ylim(0, None)
    ax.legend(title="Dataset")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_multidataset_best_alpha(summary: dict, output_path) -> Path:
    """Alpha vs nDCG@k for every dataset, with each best alpha marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ndcg_suffix = "@k"
    for dataset, curve in summary["alpha_curves"].items():
        alphas = curve["alphas"]
        ndcg_key = next(name for name in curve if name.startswith("ndcg@"))
        ndcg_suffix = f"@{ndcg_key.removeprefix('ndcg@')}"
        values = curve[ndcg_key]
        line, = ax.plot(alphas, values, marker="o", linestyle="-", label=dataset)

        best = summary["best_alpha"].get(dataset)
        if best is not None and best in alphas:
            index = alphas.index(best)
            ax.plot(
                [best], [values[index]], marker="x", markersize=10,
                markeredgewidth=2, color=line.get_color(),
            )

    ax.set_xlabel("alpha")
    ax.set_ylabel(f"nDCG{ndcg_suffix}")
    ax.set_title("Best alpha by dataset")
    ax.set_xticks(
        sorted(
            {
                alpha
                for curve in summary["alpha_curves"].values()
                for alpha in curve["alphas"]
            }
        )
    )
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(title="Dataset (x = best alpha)")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_summary(
    benchmark_paths,
    ablation_paths=None,
    output_json="outputs/multidataset_summary.json",
    figures_dir="assets/figures",
):
    """Load payloads, build the summary, write JSON, and generate figures.

    Returns ``(summary, figure_paths)``. ``benchmark_paths`` /
    ``ablation_paths`` are iterables of plain paths (loaded from disk) or
    ``(display_path, payload)`` pairs.
    """
    summary = build_summary(
        [load_payload(path) for path in benchmark_paths],
        [load_payload(path) for path in (ablation_paths or [])],
    )

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure_paths = [
        plot_multidataset_ndcg(
            summary, Path(figures_dir) / "multidataset_ndcg.png"
        )
    ]
    if summary["alpha_curves"]:
        figure_paths.append(
            plot_multidataset_best_alpha(
                summary, Path(figures_dir) / "multidataset_best_alpha.png"
            )
        )

    return summary, figure_paths


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="saved benchmark JSON payloads, one per dataset",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=[],
        help="saved fusion-ablation JSON payloads (for the best-alpha figure)",
    )
    parser.add_argument(
        "--output",
        default="outputs/multidataset_summary.json",
        help="path of the summary JSON file",
    )
    parser.add_argument(
        "--figures-dir",
        default="assets/figures",
        help="directory for the cross-dataset figures",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    summary, figure_paths = run_summary(
        args.benchmarks,
        ablation_paths=args.ablations,
        output_json=args.output,
        figures_dir=args.figures_dir,
    )

    print()
    print(format_summary_table(summary))

    print(f"\nSaved summary to {args.output}")
    for path in figure_paths:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()