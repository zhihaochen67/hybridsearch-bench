"""Fusion weight (alpha) ablation, per dataset.

    python experiments/fusion_ablation.py --dataset scifact \
        --max-queries 20 --k 10 --candidate-k 50
    python experiments/fusion_ablation.py --dataset fiqa \
        --k 10 --candidate-k 50

Evaluates Weighted Hybrid Retrieval over a grid of alpha values
(``hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_dense``)
using the existing ``HybridRetriever``. One BM25 index, one DenseRetriever
(one dense corpus encoding, one FAISS index) are built per run and reused
for every alpha value; only alpha changes between conditions.

Smoke runs save to ``outputs/<dataset>_fusion_ablation_smokeN.json``; the
full run saves to ``outputs/<dataset>_fusion_ablation.json`` and also
generates metric-vs-alpha plots under ``assets/figures/``. SciFact keeps
its historical figure names; other datasets get a dataset prefix so
figures never overwrite each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/fusion_ablation.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import (
    evaluate_retriever,
    save_results,
    select_query_ids,
)
from hybridsearch.data.common import Dataset
from hybridsearch.data.registry import DATASET_NAMES, load_dataset_by_name
from hybridsearch.retrieval.bm25 import BM25
from hybridsearch.retrieval.dense import DenseRetriever
from hybridsearch.retrieval.hybrid import HybridRetriever

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_alpha_values(text: str | None) -> list[float]:
    """Parse a comma-separated alpha string into a validated list."""
    if not text or not text.strip():
        raise ValueError("alpha list must not be empty")
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    validate_alpha_values(values)
    return values


def validate_alpha_values(alphas) -> None:
    """Reject empty lists and alpha values outside ``[0.0, 1.0]``."""
    if not alphas:
        raise ValueError("at least one alpha value is required")
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha!r}")


def build_alpha_retrievers(corpus, alpha_values, candidate_k: int,
                           dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
                           encoder=None) -> dict:
    """Build one sparse + one dense retriever and one hybrid per alpha.

    The same ``sparse`` and ``dense`` objects back every hybrid retriever,
    so the BM25 index, the dense corpus embeddings, and the FAISS index are
    each built exactly once. ``encoder`` exists for offline tests.
    """
    sparse = BM25(corpus)
    dense = DenseRetriever(corpus, model_name=dense_model, encoder=encoder)

    return {
        f"alpha={alpha:.2f}": HybridRetriever(
            sparse, dense, method="weighted", alpha=alpha, candidate_k=candidate_k
        )
        for alpha in alpha_values
    }


def run_fusion_ablation(
    dataset: Dataset,
    k: int = 10,
    candidate_k: int = 50,
    alpha_values: list | None = None,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    max_queries: int | None = None,
    retriever_factory=None,
) -> dict:
    """Evaluate weighted hybrid retrieval for every alpha and return a payload.

    The query set, query order, ``k``, and ``candidate_k`` are identical
    across alpha values. ``retriever_factory`` is an optional zero-argument
    callable returning a ``{label: retriever}`` mapping for offline tests.
    """
    if alpha_values is None:
        alpha_values = list(ALPHA_GRID)
    validate_alpha_values(alpha_values)

    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)

    if retriever_factory is None:
        retrievers = build_alpha_retrievers(
            dataset.corpus, alpha_values, candidate_k, dense_model=dense_model
        )
    else:
        retrievers = retriever_factory()

    results = {
        label: evaluate_retriever(retriever, dataset, query_ids, k)
        for label, retriever in retrievers.items()
    }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "k": k,
            "candidate_k": candidate_k,
            "alpha_values": [float(alpha) for alpha in alpha_values],
            "num_queries": len(query_ids),
            "max_queries": max_queries,
        },
        "results": results,
    }


def output_path_for(max_queries: int | None, dataset_name: str = "scifact") -> str:
    """Dataset-specific JSON path that keeps smoke and full runs separate."""
    if max_queries is not None:
        return f"outputs/{dataset_name}_fusion_ablation_smoke{max_queries}.json"
    return f"outputs/{dataset_name}_fusion_ablation.json"


def format_ablation_table(payload: dict) -> str:
    """Render the alpha ablation payload as a fixed-width text table."""
    k = payload["metadata"]["k"]
    columns = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    header = ["Alpha", f"Precision@{k}", f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}"]

    rows = [
        [name.removeprefix("alpha=")]
        + [f"{payload['results'][name][column]:.4f}" for column in columns]
        for name in payload["results"]
    ]

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
    return "\n".join(lines)


def extract_curve(payload: dict, metric: str):
    """(alphas, metric values) straight from the payload — never hard-coded."""
    alphas = list(payload["metadata"]["alpha_values"])
    values = [
        payload["results"][f"alpha={alpha:.2f}"][metric] for alpha in alphas
    ]
    return alphas, values


def _figure_filename(dataset_name: str, filename: str) -> str:
    """Dataset-scoped figure name; SciFact keeps its historical names."""
    if dataset_name == "scifact":
        return filename
    return f"{dataset_name}_{filename}"


def plot_ablation(payload: dict, output_dir="assets/figures") -> list:
    """Plot every metric vs alpha from the actual ablation results.

    Uses the non-interactive Agg backend and plain, readable styling:
    marker lines, labeled axes, a title, and x-ticks at the alpha grid.
    Returns the list of written PNG paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = payload["metadata"]["dataset"]
    k = payload["metadata"]["k"]
    metrics = {
        f"precision@{k}": "fusion_alpha_precision.png",
        f"recall@{k}": "fusion_alpha_recall.png",
        f"mrr@{k}": "fusion_alpha_mrr.png",
        f"ndcg@{k}": "fusion_alpha_ndcg.png",
    }

    paths = []
    for metric, filename in metrics.items():
        alphas, values = extract_curve(payload, metric)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(alphas, values, marker="o", linestyle="-")
        ax.set_xlabel("alpha")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_name}: {metric} vs alpha")
        ax.set_xticks(alphas)
        ax.grid(True, linestyle=":", linewidth=0.5)
        fig.tight_layout()

        path = output_dir / _figure_filename(dataset_name, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="scifact", choices=list(DATASET_NAMES)
    )
    parser.add_argument("--k", type=int, default=10, help="evaluation cutoff")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="hybrid candidate pool per retriever",
    )
    parser.add_argument(
        "--alphas",
        default=None,
        help="comma-separated alpha values, e.g. 0.0,0.25,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL_NAME,
        help="sentence-transformers model for dense retrieval",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="evaluate only the first N queries (smoke tests)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="override the default JSON output path",
    )
    parser.add_argument(
        "--figures-dir",
        default="assets/figures",
        help="directory for metric-vs-alpha plots (full runs only)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    alphas = parse_alpha_values(args.alphas) if args.alphas else list(ALPHA_GRID)

    print(f"Loading {args.dataset} and building indices once ...")
    dataset = load_dataset_by_name(args.dataset)
    payload = run_fusion_ablation(
        dataset,
        k=args.k,
        candidate_k=args.candidate_k,
        alpha_values=alphas,
        dense_model=args.dense_model,
        max_queries=args.max_queries,
    )

    print(
        f"evaluated {payload['metadata']['num_queries']} queries per alpha "
        f"(k={args.k}, candidate_k={args.candidate_k})"
    )
    print()
    print(format_ablation_table(payload))

    output = args.output or output_path_for(args.max_queries, args.dataset)
    save_results(payload, output)
    print(f"\nSaved results to {output}")

    if args.max_queries is None:
        for path in plot_ablation(payload, args.figures_dir):
            print(f"Saved figure to {path}")
    else:
        print("(plots are generated only for full runs; smoke runs save JSON only)")


if __name__ == "__main__":
    main()