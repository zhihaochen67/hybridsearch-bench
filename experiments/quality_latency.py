"""Quality vs latency study of reranker candidate size on SciFact.

    python experiments/quality_latency.py --max-queries 20 \
        --k 10 --candidate-sizes 20,50,100

Compares, over the same SciFact queries:

- BM25, Dense, Hybrid Weighted (retrieval-only, top-10)
- Hybrid Weighted + rerank@{20,50,100} (hybrid top-N candidates reranked
  down to the final top-10)

Quality metrics come from the final reranked ids; query-time latency is
the wall time of the complete per-query call (hybrid candidate retrieval +
cross-encoder scoring + final top-k) measured with ``time.perf_counter()``.
Everything expensive is built exactly once per run — BM25 index, dense
corpus embeddings, FAISS index, the Hybrid Weighted retriever, and the
cross-encoder — and reused for every query and every candidate size.
Warmup queries run untimed per method; one timed pass per method is
recorded (samples = number of queries).

Smoke runs save to ``outputs/scifact_quality_latency_smokeN.json``; the
full run saves to ``outputs/scifact_quality_latency.json`` and generates
the trade-off plots under ``assets/figures/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/quality_latency.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import save_results, select_query_ids
from experiments.latency_benchmark import summarize_latencies
from hybridsearch.data.scifact import SciFactDataset, load_scifact
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics
from hybridsearch.ranking.reranker import CrossEncoderReranker
from hybridsearch.retrieval.bm25 import BM25
from hybridsearch.retrieval.dense import DenseRetriever
from hybridsearch.retrieval.hybrid import HybridRetriever

CANDIDATE_SIZES_DEFAULT = [20, 50, 100]


def parse_candidate_sizes(text: str | None, k: int) -> list[int]:
    """Parse a comma-separated candidate-size string and validate it."""
    if not text or not text.strip():
        raise ValueError("candidate size list must not be empty")
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    validate_candidate_sizes(sizes, k)
    return sizes


def validate_candidate_sizes(sizes, k: int) -> None:
    """Reject empty lists, non-positive entries, and sizes below ``k``."""
    if not sizes:
        raise ValueError("at least one reranker candidate size is required")
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"candidate sizes must be positive integers, got {size!r}"
            )
        if size < k:
            raise ValueError(f"candidate size ({size}) must be >= k ({k})")


def build_methods(corpus, k: int, candidate_sizes, alpha: float,
                  dense_model: str, reranker_model: str,
                  hybrid_candidate_k: int = 50) -> dict:
    """Build every method once; return {name: query_text -> final top-k ids}.

    One BM25 index, one DenseRetriever (one corpus encoding, one FAISS
    index), one Hybrid Weighted retriever (with the project-standard
    candidate pool ``hybrid_candidate_k``), and one cross-encoder are
    shared by every method. For ``rerank@N`` the hybrid returns its top-N
    candidates and the reranker scores all N pairs before truncating to
    the final top-k.
    """
    sparse = BM25(corpus)
    dense = DenseRetriever(corpus, model_name=dense_model)
    hybrid = HybridRetriever(
        sparse, dense, method="weighted", alpha=alpha, candidate_k=hybrid_candidate_k
    )
    reranker = CrossEncoderReranker(model_name=reranker_model)

    methods = {
        "BM25": lambda q: [r["id"] for r in sparse.search(q, top_k=k)],
        "Dense": lambda q: [r["id"] for r in dense.search(q, top_k=k)],
        "Hybrid Weighted": lambda q: [r["id"] for r in hybrid.search(q, top_k=k)],
    }
    for size in candidate_sizes:
        def make(size: int):
            return lambda q: [
                r["id"]
                for r in reranker.rerank(
                    q, hybrid.search(q, top_k=size), top_k=k
                )
            ]

        methods[f"Hybrid + rerank@{size}"] = make(size)

    return methods


def run_method(search_fn, query_ids, query_texts, qrels, k: int,
               warmup: int, timer) -> dict:
    """Warm up untimed, time each query once, and evaluate quality.

    Returns ``{"quality": mean metrics, "latency": summary}`` where the
    latency of one query covers the full ``search_fn`` call — for reranked
    methods that includes hybrid retrieval, cross-encoder scoring, and the
    final top-k.
    """
    if warmup:
        for query_text in query_texts[:warmup]:
            search_fn(query_text)

    latencies_ms = []
    per_query = []
    for query_id, query_text in zip(query_ids, query_texts):
        start = timer()
        ranked_ids = search_fn(query_text)
        latencies_ms.append((timer() - start) * 1000.0)
        per_query.append(evaluate_query(ranked_ids, qrels[query_id], k=k))

    return {
        "quality": mean_metrics(per_query),
        "latency": summarize_latencies(latencies_ms),
    }


def run_quality_latency(
    dataset: SciFactDataset,
    k: int = 10,
    candidate_sizes=None,
    alpha: float = 0.5,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL_NAME,
    hybrid_candidate_k: int = 50,
    max_queries: int | None = None,
    warmup_queries: int = 3,
    method_factory=None,
    timer=time.perf_counter,
) -> dict:
    """Run the quality/latency comparison and return a serializable payload.

    Every method sees the identical query set and order. ``method_factory``
    is an optional zero-argument callable returning a
    ``{name: query_text -> ranked_ids}`` mapping; ``timer`` is a callable
    returning seconds (both exist for offline tests).
    """
    if candidate_sizes is None:
        candidate_sizes = list(CANDIDATE_SIZES_DEFAULT)
    validate_candidate_sizes(candidate_sizes, k)

    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)
    query_texts = [dataset.queries[query_id] for query_id in query_ids]

    if method_factory is None:
        methods = build_methods(
            dataset.corpus, k, candidate_sizes, alpha, dense_model,
            reranker_model, hybrid_candidate_k=hybrid_candidate_k,
        )
    else:
        methods = method_factory()

    results = {
        name: run_method(search_fn, query_ids, query_texts, dataset.qrels,
                         k, warmup_queries, timer)
        for name, search_fn in methods.items()
    }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "k": k,
            "candidate_sizes": [int(size) for size in candidate_sizes],
            "alpha": alpha,
            "hybrid_candidate_k": hybrid_candidate_k,
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "warmup_queries": warmup_queries,
            "timer": getattr(timer, "__name__", type(timer).__name__),
            "timing_passes": 1,
            "methodology": [
                "one timed pass per method; samples = number of queries",
                "latency covers the full per-query call: hybrid candidate "
                "retrieval + cross-encoder scoring + final top-k",
                "dataset loading, index construction, dense corpus encoding, "
                "FAISS build, and model loading are excluded",
            ],
        },
        "results": results,
    }


def output_path_for(max_queries: int | None) -> str:
    """JSON path that keeps smoke and full runs from overwriting each other."""
    if max_queries is not None:
        return f"outputs/scifact_quality_latency_smoke{max_queries}.json"
    return "outputs/scifact_quality_latency.json"


def format_quality_latency_table(payload: dict) -> str:
    """Render quality and latency side by side as a fixed-width table."""
    k = payload["metadata"]["k"]
    header = [
        "Method", f"Precision@{k}", f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}",
        "Mean ms", "Median ms", "P95 ms",
    ]

    rows = []
    for name, entry in payload["results"].items():
        quality = entry["quality"]
        latency = entry["latency"]
        rows.append(
            [
                name,
                f"{quality[f'precision@{k}']:.4f}",
                f"{quality[f'recall@{k}']:.4f}",
                f"{quality[f'mrr@{k}']:.4f}",
                f"{quality[f'ndcg@{k}']:.4f}",
                f"{latency['mean_ms']:.2f}",
                f"{latency['median_ms']:.2f}",
                f"{latency['p95_ms']:.2f}",
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
    return "\n".join(lines)


def rerank_points(payload: dict):
    """(sizes, entries) of the reranked methods, ordered by candidate size."""
    k = payload["metadata"]["k"]
    points = []
    for name, entry in payload["results"].items():
        if "rerank@" in name:
            points.append((int(name.rsplit("@", 1)[1]), entry))
    points.sort(key=lambda item: item[0])
    return [size for size, _ in points], [entry for _, entry in points]


def plot_quality_latency_ndcg(payload: dict, output_dir="assets/figures"):
    """Scatter every method at (mean latency, nDCG@k) with labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = payload["metadata"]["k"]
    names = list(payload["results"])
    xs = [payload["results"][name]["latency"]["mean_ms"] for name in names]
    ys = [payload["results"][name]["quality"][f"ndcg@{k}"] for name in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys)
    for name, x, y in zip(names, xs, ys):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 3),
                    fontsize=8)
    ax.set_xlabel("Mean query latency (ms)")
    ax.set_ylabel(f"nDCG@{k}")
    ax.set_title("SciFact: quality vs latency")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    path = Path(output_dir) / "quality_latency_ndcg.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reranker_candidate_tradeoff(payload: dict, output_dir="assets/figures"):
    """Twin-axis plot: nDCG@k and mean latency vs reranker candidate size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = payload["metadata"]["k"]
    sizes, entries = rerank_points(payload)
    ndcg = [entry["quality"][f"ndcg@{k}"] for entry in entries]
    mean_ms = [entry["latency"]["mean_ms"] for entry in entries]

    fig, ax = plt.subplots(figsize=(7, 4))
    ndcg_line, = ax.plot(sizes, ndcg, marker="o", color="tab:blue",
                         label=f"nDCG@{k}")
    ax.set_xlabel("Reranker candidate size")
    ax.set_ylabel(f"nDCG@{k}", color="tab:blue")
    ax.set_xticks(sizes)
    ax.grid(True, linestyle=":", linewidth=0.5)

    latency_ax = ax.twinx()
    latency_line, = latency_ax.plot(
        sizes, mean_ms, marker="s", color="tab:red", label="Mean latency (ms)"
    )
    latency_ax.set_ylabel("Mean latency (ms)", color="tab:red")

    ax.legend(handles=[ndcg_line, latency_line], loc="center right")
    ax.set_title("SciFact: reranker candidate size trade-off")
    fig.tight_layout()

    path = Path(output_dir) / "reranker_candidate_tradeoff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(payload: dict, output_dir="assets/figures") -> list:
    """Generate the quality/latency figures from the actual payload."""
    return [
        plot_quality_latency_ndcg(payload, output_dir),
        plot_reranker_candidate_tradeoff(payload, output_dir),
    ]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=["scifact"])
    parser.add_argument("--k", type=int, default=10, help="final evaluation cutoff")
    parser.add_argument(
        "--candidate-sizes",
        default=None,
        help="comma-separated reranker candidate sizes, e.g. 20,50,100",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="sparse weight for weighted fusion",
    )
    parser.add_argument(
        "--hybrid-candidate-k",
        type=int,
        default=50,
        help="Hybrid Weighted internal candidate pool",
    )
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL_NAME,
        help="sentence-transformers model for dense retrieval",
    )
    parser.add_argument(
        "--reranker-model",
        default=CrossEncoderReranker.DEFAULT_MODEL_NAME,
        help="cross-encoder model for reranking",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="evaluate only the first N queries (smoke tests)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="number of untimed warmup queries per method",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="override the default JSON output path",
    )
    parser.add_argument(
        "--figures-dir",
        default="assets/figures",
        help="directory for the trade-off plots (full runs only)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    sizes = (
        parse_candidate_sizes(args.candidate_sizes, args.k)
        if args.candidate_sizes
        else list(CANDIDATE_SIZES_DEFAULT)
    )

    print(f"Loading {args.dataset} and building indices/models once (untimed) ...")
    dataset = load_scifact()
    payload = run_quality_latency(
        dataset,
        k=args.k,
        candidate_sizes=sizes,
        alpha=args.alpha,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        hybrid_candidate_k=args.hybrid_candidate_k,
        max_queries=args.max_queries,
        warmup_queries=args.warmup,
    )

    print(
        f"evaluated {payload['metadata']['num_queries']} queries per method "
        f"(k={args.k}, candidate_sizes={payload['metadata']['candidate_sizes']})"
    )
    print()
    print(format_quality_latency_table(payload))

    output = args.output or output_path_for(args.max_queries)
    save_results(payload, output)
    print(f"\nSaved results to {output}")

    if args.max_queries is None:
        for path in plot_all(payload, args.figures_dir):
            print(f"Saved figure to {path}")
    else:
        print("(plots are generated only for full runs; smoke runs save JSON only)")


if __name__ == "__main__":
    main()
