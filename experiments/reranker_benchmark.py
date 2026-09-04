"""SciFact benchmark with cross-encoder reranking (BM25 / Dense / Hybrid).

    python experiments/reranker_benchmark.py --max-queries 20 \
        --k 10 --candidate-k 50

Measures retrieval quality with and without second-stage cross-encoder
reranking:

    Query -> Hybrid Weighted retrieval (candidate_k candidates)
          -> CrossEncoderReranker
          -> final top-k ids
          -> evaluation

The reranker always sees a candidate pool of ``candidate_k >= k``
documents (validated explicitly); it is never asked to rerank an
already-truncated top-k. Every index and model is built once per run —
BM25 index, dense corpus embeddings, FAISS index, both hybrid retrievers,
and the cross-encoder are reused across all queries.

Smoke runs (``--max-queries N``) save to
``outputs/scifact_reranker_smokeN.json``; the full run saves to
``outputs/scifact_reranker_benchmark.json``, so neither the smoke nor the
full run overwrites the other or the retrieval/latency outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/reranker_benchmark.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import (
    build_retrievers,
    format_table,
    save_results,
    select_query_ids,
)
from hybridsearch.data.scifact import SciFactDataset, load_scifact
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics
from hybridsearch.ranking.reranker import CrossEncoderReranker
from hybridsearch.retrieval.dense import DenseRetriever


def rerank_query(query: str, retriever, reranker, candidate_k: int, k: int) -> list:
    """Hybrid -> candidate_k candidates -> cross-encoder -> final top-k ids."""
    candidates = retriever.search(query, top_k=candidate_k)
    reranked = reranker.rerank(query, candidates, top_k=k)
    return [result["id"] for result in reranked]


def build_methods(corpus, k: int, candidate_k: int, alpha: float, rrf_k: float,
                  dense_model: str, reranker_model: str) -> dict:
    """Build every benchmark method once; return {name: query->ranked_ids}.

    All methods share one BM25 index, one dense/FAISS index, and one
    cross-encoder instance. Retrieval-only methods return the top-k of
    ``search``; reranked methods first fetch ``candidate_k`` candidates and
    rerank them down to ``k``.
    """
    retrievers = build_retrievers(corpus, candidate_k, alpha, rrf_k, dense_model)
    sparse = retrievers["BM25"]
    dense = retrievers["Dense"]
    weighted = retrievers["Hybrid Weighted"]
    rrf = retrievers["Hybrid RRF"]

    reranker = CrossEncoderReranker(model_name=reranker_model)

    return {
        "BM25": lambda q: [r["id"] for r in sparse.search(q, top_k=k)],
        "Dense": lambda q: [r["id"] for r in dense.search(q, top_k=k)],
        "Hybrid Weighted": lambda q: [r["id"] for r in weighted.search(q, top_k=k)],
        "Hybrid RRF": lambda q: [r["id"] for r in rrf.search(q, top_k=k)],
        "Hybrid Weighted + Reranker": lambda q: rerank_query(
            q, weighted, reranker, candidate_k, k
        ),
        "Hybrid RRF + Reranker": lambda q: rerank_query(
            q, rrf, reranker, candidate_k, k
        ),
    }


def evaluate_method(search_fn, dataset: SciFactDataset, query_ids, k: int):
    """Arithmetic mean of per-query metrics for one query->ranked_ids callable."""
    per_query = []
    for query_id in query_ids:
        ranked_ids = search_fn(dataset.queries[query_id])
        per_query.append(evaluate_query(ranked_ids, dataset.qrels[query_id], k=k))
    return mean_metrics(per_query)


def run_reranker_benchmark(
    dataset: SciFactDataset,
    k: int = 10,
    candidate_k: int = 50,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL_NAME,
    max_queries: int | None = None,
    method_factory=None,
) -> dict:
    """Evaluate all methods and return a JSON-serializable payload.

    ``candidate_k < k`` raises ``ValueError`` before anything is built.
    ``method_factory`` is an optional zero-argument callable returning a
    ``{method_name: callable(query_text) -> ranked_ids}`` mapping for
    offline tests.
    """
    if candidate_k < k:
        raise ValueError(
            f"candidate_k ({candidate_k}) must be >= k ({k})"
        )

    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)

    if method_factory is None:
        methods = build_methods(
            dataset.corpus, k, candidate_k, alpha, rrf_k, dense_model, reranker_model
        )
    else:
        methods = method_factory()

    results = {
        name: evaluate_method(search_fn, dataset, query_ids, k)
        for name, search_fn in methods.items()
    }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "k": k,
            "candidate_k": candidate_k,
            "alpha": alpha,
            "rrf_k": rrf_k,
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "methods": list(methods),
        },
        "results": results,
    }


def output_path_for(max_queries: int | None) -> str:
    """JSON path that keeps smoke and full runs from overwriting each other."""
    if max_queries is not None:
        return f"outputs/scifact_reranker_smoke{max_queries}.json"
    return "outputs/scifact_reranker_benchmark.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=["scifact"])
    parser.add_argument("--k", type=int, default=10, help="final evaluation cutoff")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="hybrid candidate pool fed to the reranker",
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
        "--output",
        default=None,
        help="override the default JSON output path",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Loading {args.dataset} and building indices/models (untimed) ...")
    dataset = load_scifact()
    payload = run_reranker_benchmark(
        dataset,
        k=args.k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        max_queries=args.max_queries,
    )

    print(
        f"evaluated {payload['metadata']['num_queries']} queries "
        f"(k={args.k}, candidate_k={args.candidate_k})"
    )
    print()
    print(format_table(payload))

    output = args.output or output_path_for(args.max_queries)
    save_results(payload, output)
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
