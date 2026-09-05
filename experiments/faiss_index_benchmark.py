"""FAISS index comparison (Phase 14A): Flat / HNSW / IVF, smoke harness.

    python experiments/faiss_index_benchmark.py --dataset fiqa \
        --indexes flat,hnsw,ivf --max-queries 20 --k 10

Compares FAISS index types on *identical* L2-normalized embeddings. The
corpus and all query embeddings are produced exactly once, before any
timing; then each index is built (timed: construction/training + add
only) and every precomputed query embedding is searched (timed per
query). ANN quality is measured against the exact Flat top-k with
``ann_recall_at_k`` — the fraction of the Flat top-k also returned by the
ANN index, which is an approximation metric, NOT qrel relevance Recall@K.

This phase ships the skeleton plus a small smoke study only; the full
648-query FiQA grid is a later phase. Smoke runs save to
``outputs/<dataset>_faiss_index_smokeN.json``; full runs save to
``outputs/<dataset>_faiss_index_benchmark.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/faiss_index_benchmark.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import save_results, select_query_ids
from experiments.latency_benchmark import summarize_latencies
from hybridsearch.data.registry import DATASET_NAMES, load_dataset_by_name
from hybridsearch.retrieval.dense import DenseRetriever, l2_normalize
from hybridsearch.retrieval.vector_index import (
    HNSW_DEFAULT_EF_CONSTRUCTION,
    HNSW_DEFAULT_EF_SEARCH,
    HNSW_DEFAULT_M,
    IVF_DEFAULT_NLIST,
    IVF_DEFAULT_NPROBE,
    ann_recall_at_k,
    build_faiss_index,
    search_faiss_index,
    serialized_index_size,
    validate_index_type,
)

DEFAULT_INDEXES = ("flat", "hnsw", "ivf")


def resolve_indexes(text: str | None) -> list[str]:
    """Parse a comma-separated index-type list into a validated, ordered,
    de-duplicated list."""
    if not text or not text.strip():
        raise ValueError("index list must not be empty")
    indexes = []
    for part in text.split(","):
        index_type = part.strip()
        if not index_type:
            continue
        validate_index_type(index_type)
        if index_type not in indexes:
            indexes.append(index_type)
    if not indexes:
        raise ValueError("index list must not be empty")
    return indexes


def _load_encoder(model_name: str):
    """Lazily load the sentence-transformers encoder (never in tests)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_all(encoder, corpus_texts, query_texts):
    """Encode and L2-normalize corpus + queries once, outside any timing.

    Returns ``(corpus_embeddings, query_embeddings)`` as float64 arrays —
    identical inputs for every index type.
    """
    import numpy as np

    corpus_embeddings = l2_normalize(
        np.asarray(encoder.encode(corpus_texts), dtype=np.float64)
    )
    query_embeddings = l2_normalize(
        np.asarray(encoder.encode(query_texts), dtype=np.float64)
    )
    return corpus_embeddings, query_embeddings


def run_faiss_index_benchmark(
    dataset,
    k: int = 10,
    max_queries: int | None = None,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    index_specs=None,
    warmup_queries: int = 3,
    encoder=None,
    timer=time.perf_counter,
) -> dict:
    """Build and search each FAISS index type; return a serializable payload.

    ``index_specs`` is a list of ``(index_type, params)`` pairs (default:
    Flat/HNSW/IVF with project defaults). ``encoder`` and ``timer`` exist
    for offline tests; encoding happens once, before any timing, and a
    Flat reference (built un-timed if not part of ``index_specs``) always
    supplies the exact top-k for ``ann_recall_at_k``.
    """
    if index_specs is None:
        index_specs = [(name, {}) for name in DEFAULT_INDEXES]
    index_specs = [(validate_index_type(name), dict(params or {})) for name, params in index_specs]
    if not index_specs:
        raise ValueError("at least one index type is required")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)
    corpus_texts = [document["text"] for document in dataset.corpus]
    query_texts = [dataset.queries[query_id] for query_id in query_ids]

    encoder = encoder or _load_encoder(dense_model)
    corpus_embeddings, query_embeddings = _encode_all(
        encoder, corpus_texts, query_texts
    )

    # Build each requested index, timing construction only (training + add).
    built = {}
    build_times = {}
    for index_type, params in index_specs:
        start = timer()
        info = build_faiss_index(
            corpus_embeddings, index_type=index_type, params=params
        )
        build_times[index_type] = timer() - start
        built[index_type] = info

    # Exact Flat reference for ANN recall (un-timed when not requested).
    if "flat" in built:
        flat_info = built["flat"]
    else:
        flat_info = build_faiss_index(corpus_embeddings, index_type="flat")

    flat_ids_per_query = [
        search_faiss_index(flat_info, query_embedding, k)[1][0].tolist()
        for query_embedding in query_embeddings
    ]

    results = {}
    for index_type, params in index_specs:
        info = built[index_type]

        if warmup_queries:
            for query_embedding in query_embeddings[:warmup_queries]:
                search_faiss_index(info, query_embedding, k)

        latencies_ms = []
        recalls = []
        for query_embedding, flat_ids in zip(query_embeddings, flat_ids_per_query):
            start = timer()
            _, indices = search_faiss_index(info, query_embedding, k)
            latencies_ms.append((timer() - start) * 1000.0)
            ann_ids = indices[0].tolist()
            recalls.append(ann_recall_at_k(ann_ids, flat_ids, k))

        results[index_type] = {
            "index_type": index_type,
            "params": info.params,
            "dimension": info.dimension,
            "n_vectors": info.n_vectors,
            "requires_training": info.requires_training,
            "trained": info.trained,
            "build_time_s": build_times[index_type],
            "serialized_size_bytes": serialized_index_size(info.index),
            "ann_recall_at_k": sum(recalls) / len(recalls),
            "latency": summarize_latencies(latencies_ms),
        }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "k": k,
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "warmup_queries": warmup_queries,
            "indexes": [
                {"index_type": index_type, "params": params}
                for index_type, params in index_specs
            ],
            "timer": getattr(timer, "__name__", type(timer).__name__),
            "timing_methodology": [
                "corpus/query encoding and model loading are excluded from every timing",
                "build_time_s covers FAISS index construction only (training + add)",
                "per-query latency covers FAISS search on precomputed query embeddings",
                "warmup queries are executed untimed",
                "ann_recall_at_k = mean over queries of |ANN_top_k AND Flat_top_k| / k (approximation vs exact Flat, not qrel relevance Recall@k)",
                "deterministic inputs: fixed query order, identical embeddings across index types, FAISS default clustering seed",
            ],
        },
        "results": results,
    }


def output_path_for(dataset_name: str, max_queries: int | None = None) -> str:
    """Dataset-scoped JSON path; smoke runs get their own file."""
    suffix = f"_smoke{max_queries}" if max_queries is not None else ""
    return f"outputs/{dataset_name}_faiss_index_benchmark{suffix}.json"


def format_table(payload: dict) -> str:
    """Render index comparison as a fixed-width text table."""
    k = payload["metadata"]["k"]
    header = [
        "Index", f"ANN Recall@{k}", "Build s", "Size MB",
        "Mean ms", "Median ms", "P95 ms",
    ]
    rows = []
    for name, entry in payload["results"].items():
        latency = entry["latency"]
        rows.append(
            [
                name,
                f"{entry['ann_recall_at_k']:.4f}",
                f"{entry['build_time_s']:.3f}",
                f"{entry['serialized_size_bytes'] / 1e6:.2f}",
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="scifact", choices=list(DATASET_NAMES)
    )
    parser.add_argument(
        "--indexes",
        default="flat,hnsw,ivf",
        help="comma-separated FAISS index types: flat,hnsw,ivf",
    )
    parser.add_argument("--k", type=int, default=10, help="retrieval top_k per query")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="evaluate only the first N queries (smoke tests)",
    )
    parser.add_argument("--hnsw-m", type=int, default=HNSW_DEFAULT_M, help="HNSW M")
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=HNSW_DEFAULT_EF_CONSTRUCTION,
        help="HNSW efConstruction",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=HNSW_DEFAULT_EF_SEARCH,
        help="HNSW efSearch",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=IVF_DEFAULT_NLIST,
        help="IVF number of cells (must be <= corpus size)",
    )
    parser.add_argument(
        "--ivf-nprobe",
        type=int,
        default=IVF_DEFAULT_NPROBE,
        help="IVF cells probed per query (must be <= nlist)",
    )
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL_NAME,
        help="sentence-transformers model for the embeddings",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed warmup queries per index",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="override the default dataset-specific JSON output path",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    indexes = resolve_indexes(args.indexes)

    index_specs = []
    for index_type in indexes:
        if index_type == "hnsw":
            params = {
                "M": args.hnsw_m,
                "efConstruction": args.ef_construction,
                "efSearch": args.ef_search,
            }
        elif index_type == "ivf":
            params = {"nlist": args.ivf_nlist, "nprobe": args.ivf_nprobe}
        else:
            params = {}
        index_specs.append((index_type, params))

    print(f"Loading {args.dataset} and encoding once (untimed) ...")
    dataset = load_dataset_by_name(args.dataset)
    print(
        f"documents={len(dataset.corpus)} qrel_query_ids={len(dataset.qrels)} "
        f"indexes={indexes} k={args.k}"
    )

    payload = run_faiss_index_benchmark(
        dataset,
        k=args.k,
        max_queries=args.max_queries,
        dense_model=args.dense_model,
        index_specs=index_specs,
        warmup_queries=args.warmup,
    )

    print()
    print(format_table(payload))

    output = args.output or output_path_for(args.dataset, args.max_queries)
    save_results(payload, output)
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()