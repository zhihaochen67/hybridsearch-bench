"""Query-time latency benchmark for SciFact (BM25 / Dense / Hybrid).

    python experiments/latency_benchmark.py --dataset scifact \
        --k 10 --candidate-k 50 --alpha 0.5

Measures per-query retrieval latency only: dataset loading, BM25 index
construction, dense corpus encoding, FAISS index construction, and model
loading all happen *before* any timing. Each query is timed once with
``time.perf_counter()`` around a single ``search`` call, after a small
per-method warmup. Reports mean, median, and p95 latency per query across
all measured SciFact test queries and saves the payload as JSON under
``outputs/``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/latency_benchmark.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import (
    build_retrievers,
    save_results,
    select_query_ids,
)
from hybridsearch.data.scifact import SciFactDataset, load_scifact
from hybridsearch.retrieval.dense import DenseRetriever


def summarize_latencies(latencies_ms) -> dict:
    """Aggregate latency samples into mean, median, and p95.

    ``median`` is the standard interpolated median; ``p95`` is the
    nearest-rank 95th percentile: the value at 0-based index
    ``ceil(0.95 * n) - 1`` of the sorted samples. Also includes min, max,
    and the sample count.
    """
    if not latencies_ms:
        raise ValueError("no latency samples to summarize")

    ordered = sorted(latencies_ms)
    p95_index = math.ceil(0.95 * len(ordered)) - 1

    return {
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(ordered),
    }


def time_query(retriever, query_text: str, top_k: int, timer=time.perf_counter) -> float:
    """Wall time in milliseconds of one ``search`` call."""
    start = timer()
    retriever.search(query_text, top_k=top_k)
    return (timer() - start) * 1000.0


def measure_method(retriever, query_texts, top_k: int, warmup: int = 0, timer=time.perf_counter) -> list:
    """Per-query latencies in milliseconds for one retriever.

    The first ``warmup`` queries are executed untimed (warming up Python
    allocators, tokenizers, and FAISS), then every query in ``query_texts``
    is timed exactly once, in order, so runs are deterministic.
    """
    if warmup:
        for query_text in query_texts[:warmup]:
            retriever.search(query_text, top_k=top_k)

    return [
        time_query(retriever, query_text, top_k, timer=timer)
        for query_text in query_texts
    ]


def run_latency_benchmark(
    dataset: SciFactDataset,
    k: int = 10,
    candidate_k: int = 50,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    max_queries: int | None = None,
    warmup_queries: int = 3,
    retriever_factory=None,
    timer=time.perf_counter,
) -> dict:
    """Measure per-query latency of all methods and return a payload.

    Index/model construction happens before any timing. ``retriever_factory``
    is an optional zero-argument callable returning a
    ``{method_name: retriever}`` mapping for offline tests; ``timer`` is a
    callable returning seconds (``time.perf_counter`` by default).
    """
    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)
    query_texts = [dataset.queries[query_id] for query_id in query_ids]

    if retriever_factory is None:
        retrievers = build_retrievers(dataset.corpus, candidate_k, alpha, rrf_k, dense_model)
    else:
        retrievers = retriever_factory()

    results = {}
    for name, retriever in retrievers.items():
        latencies_ms = measure_method(
            retriever, query_texts, top_k=k, warmup=warmup_queries, timer=timer
        )
        results[name] = {
            **summarize_latencies(latencies_ms),
            "latencies_ms": latencies_ms,
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
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "warmup_queries": warmup_queries,
            "timer": getattr(timer, "__name__", type(timer).__name__),
        },
        "results": results,
    }


def format_latency_table(payload: dict) -> str:
    """Render mean/median/p95 latencies as a fixed-width text table."""
    header = ["Method", "Mean ms", "Median ms", "P95 ms"]
    rows = [
        [
            name,
            f"{summary['mean_ms']:.2f}",
            f"{summary['median_ms']:.2f}",
            f"{summary['p95_ms']:.2f}",
        ]
        for name, summary in payload["results"].items()
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
    parser.add_argument("--k", type=int, default=10, help="retrieval top_k per query")
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
        help="measure only the first N queries (smoke tests)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="number of untimed warmup queries per method",
    )
    parser.add_argument(
        "--output",
        default="outputs/scifact_latency.json",
        help="path of the JSON results file",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Loading {args.dataset} and building indices (untimed) ...")
    dataset = load_scifact()
    payload = run_latency_benchmark(
        dataset,
        k=args.k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        dense_model=args.dense_model,
        max_queries=args.max_queries,
        warmup_queries=args.warmup,
    )

    print(
        f"measured {payload['metadata']['num_queries']} queries per method "
        f"(top_k={args.k}, candidate_k={args.candidate_k})"
    )
    print()
    print(format_latency_table(payload))

    save_results(payload, args.output)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
