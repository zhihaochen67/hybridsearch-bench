"""First real retrieval benchmark: SciFact with BM25 / Dense / Hybrid.

    python experiments/benchmark.py --dataset scifact --k 10 \
        --candidate-k 50 --alpha 0.5 --max-queries 20

Builds the BM25 and dense indices exactly once per run, evaluates every
query that has qrels with the Phase 5 metrics, prints a table of
arithmetic means across queries, and saves the payload as JSON under
``outputs/``. Model/index construction time is not part of the measured
quality numbers (latency benchmarking comes later).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/benchmark.py` resolve the
    # `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybridsearch.data.scifact import SciFactDataset, load_scifact
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics
from hybridsearch.retrieval.bm25 import BM25
from hybridsearch.retrieval.dense import DenseRetriever
from hybridsearch.retrieval.hybrid import HybridRetriever


def select_query_ids(queries, qrels, max_queries=None) -> list[str]:
    """Query ids that have at least one qrel judgment, in insertion order.

    ``max_queries`` keeps the first ``max_queries`` ids. Raises
    ``ValueError`` when nothing is left to evaluate.
    """
    query_ids = [query_id for query_id in queries if qrels.get(query_id)]
    if max_queries is not None:
        query_ids = query_ids[:max_queries]
    if not query_ids:
        raise ValueError("no queries with qrels to evaluate")
    return query_ids


def evaluate_retriever(retriever, dataset: SciFactDataset, query_ids, k: int):
    """Arithmetic mean of per-query metrics for one retriever."""
    per_query = []
    for query_id in query_ids:
        query_text = dataset.queries[query_id]
        ranked_ids = [
            result["id"] for result in retriever.search(query_text, top_k=k)
        ]
        per_query.append(evaluate_query(ranked_ids, dataset.qrels[query_id], k=k))
    return mean_metrics(per_query)


def build_retrievers(corpus, candidate_k: int, alpha: float, rrf_k: float, dense_model: str):
    """Build the four benchmark methods, sharing one sparse and dense index.

    The BM25 index and the dense corpus embeddings are each built once;
    both hybrid methods reuse the same two retrievers.
    """
    sparse = BM25(corpus)
    dense = DenseRetriever(corpus, model_name=dense_model)

    return {
        "BM25": sparse,
        "Dense": dense,
        "Hybrid Weighted": HybridRetriever(
            sparse, dense, method="weighted", alpha=alpha, candidate_k=candidate_k
        ),
        "Hybrid RRF": HybridRetriever(
            sparse, dense, method="rrf", k=rrf_k, candidate_k=candidate_k
        ),
    }


def run_benchmark(
    dataset: SciFactDataset,
    k: int = 10,
    candidate_k: int = 50,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    max_queries: int | None = None,
    retriever_factory=None,
) -> dict:
    """Evaluate all methods and return a JSON-serializable payload.

    ``retriever_factory`` is an optional zero-argument callable returning a
    ``{method_name: retriever}`` mapping; it replaces the real
    sparse/dense/hybrid construction and keeps tests offline.
    """
    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)

    if retriever_factory is None:
        retrievers = build_retrievers(dataset.corpus, candidate_k, alpha, rrf_k, dense_model)
    else:
        retrievers = retriever_factory()

    results = {
        name: evaluate_retriever(retriever, dataset, query_ids, k)
        for name, retriever in retrievers.items()
    }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "k": k,
            "candidate_k": candidate_k,
            "alpha": alpha,
            "rrf_k": rrf_k,
            "num_documents": len(dataset.corpus),
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "seed": None,
        },
        "results": results,
    }


def save_results(payload: dict, path) -> None:
    """Write the benchmark payload as pretty-printed JSON, creating dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def format_table(payload: dict) -> str:
    """Render the benchmark payload as a fixed-width text table."""
    k = payload["metadata"]["k"]
    columns = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    header = ["Method", f"Precision@{k}", f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}"]

    rows = [
        [name] + [f"{payload['results'][name][column]:.4f}" for column in columns]
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=["scifact"])
    parser.add_argument("--k", type=int, default=10, help="evaluation cutoff")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="hybrid candidate pool per retriever",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="sparse weight for weighted fusion",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        help="RRF constant",
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
        default="outputs/scifact_benchmark.json",
        help="path of the JSON results file",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Loading {args.dataset} ...")
    dataset = load_scifact()
    print(
        f"documents={len(dataset.corpus)} "
        f"queries={len(dataset.queries)} "
        f"qrel_query_ids={len(dataset.qrels)}"
    )

    payload = run_benchmark(
        dataset,
        k=args.k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        dense_model=args.dense_model,
        max_queries=args.max_queries,
    )

    print()
    print(format_table(payload))

    save_results(payload, args.output)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
