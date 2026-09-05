"""Full FAISS index study: Flat / HNSW / IVF tradeoffs.

    python experiments/faiss_index_benchmark.py --dataset fiqa

Compares FAISS index configurations on the full FiQA evaluation set
(648 judged queries, k = 10):

- Flat: exact ``IndexFlatIP`` reference (ANN Recall@10 == 1.0);
- HNSW grid: M = 32, efConstruction = 200, efSearch in {16, 32, 64, 128};
- IVF grid: nlist in {256, 512, 1024} x nprobe in {8, 16, 32, 64}
  (only combinations with nprobe <= nlist).

Methodology (identical for every configuration):

- the corpus and all 648 query texts are encoded exactly ONCE per
  invocation; every configuration reuses the same float32 L2-normalized
  embedding matrices, so encoding is excluded from every timing;
- indices are built once per *build configuration* — search-time
  parameters (HNSW ``efSearch``, IVF ``nprobe``) are varied on the same
  index without rebuilding; ``build_time_s`` measures construction
  (training + add) only;
- per query: warmup passes run untimed, then ``timed_passes`` timed
  passes over the precomputed query embeddings (default 5); per-query
  latency is the mean of its timed samples and the summary aggregates the
  per-query means;
- ANN Recall@10 = |ANN top10 ∩ Flat top10| / 10 per query, macro-averaged
  over the 648 queries (approximation quality vs exact search — NOT qrel
  relevance Recall@10); recall distribution stats are reported alongside
  the mean;
- a secondary diagnostic evaluates each configuration's top-10 ids
  against the FiQA qrels and reports nDCG@10 (relevance sanity check;
  ``ann_recall`` remains the primary metric);
- ``serialized_size_bytes`` is the FAISS index file size alone (no
  document text, no model weights).

The full run saves ``outputs/<dataset>_faiss_index_benchmark.json`` and
generates the figures under ``assets/figures/``; smoke runs
(``--max-queries N``) save
``outputs/<dataset>_faiss_index_benchmark_smokeN.json`` only.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/faiss_index_benchmark.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import save_results, select_query_ids
from experiments.latency_benchmark import summarize_latencies
from hybridsearch.data.registry import DATASET_NAMES, load_dataset_by_name
from hybridsearch.evaluation.metrics import evaluate_query
from hybridsearch.retrieval.dense import DenseRetriever, l2_normalize
from hybridsearch.retrieval.vector_index import (
    HNSW_DEFAULT_EF_CONSTRUCTION,
    HNSW_DEFAULT_M,
    IVF_DEFAULT_NLIST,
    IVF_DEFAULT_NPROBE,
    ann_recall_at_k,
    build_faiss_index,
    search_faiss_index,
    serialized_index_size,
    validate_hnsw_params,
    validate_index_type,
    validate_ivf_params,
)

HNSW_EF_SEARCH_GRID = (16, 32, 64, 128)
IVF_NLIST_GRID = (256, 512, 1024)
IVF_NPROBE_GRID = (8, 16, 32, 64)


# --- grid construction ---


def parse_int_list(text: str | None, name: str) -> list[int]:
    """Parse a comma-separated list of positive integers (ordered, unique)."""
    if not text or not text.strip():
        raise ValueError(f"{name} list must not be empty")
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(
                f"{name} entries must be integers, got {part!r}"
            ) from None
        if value <= 0:
            raise ValueError(f"{name} entries must be positive, got {value!r}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} list must not be empty")
    return values


def build_grid(
    hnsw_m: int = HNSW_DEFAULT_M,
    ef_construction: int = HNSW_DEFAULT_EF_CONSTRUCTION,
    ef_search_values=(16, 32, 64, 128),
    ivf_nlist_values=(256, 512, 1024),
    ivf_nprobe_values=(8, 16, 32, 64),
):
    """Build the ordered grid: ``(label, index_type, params)`` points plus
    the invalid IVF combinations that were excluded.

    HNSW values are validated eagerly (invalid values fail clearly instead
    of being substituted); IVF combinations with ``nprobe > nlist`` are
    skipped and reported in the returned ``excluded`` list.
    """
    points = []
    excluded = []

    points.append(("flat", "flat", {}))

    for ef_search in ef_search_values:
        params = {
            "M": hnsw_m,
            "efConstruction": ef_construction,
            "efSearch": ef_search,
        }
        validate_hnsw_params(params)
        points.append((f"hnsw_efS{ef_search}", "hnsw", params))

    for nlist in ivf_nlist_values:
        for nprobe in ivf_nprobe_values:
            if nprobe > nlist:
                excluded.append(
                    {"nlist": nlist, "nprobe": nprobe,
                     "reason": "nprobe > nlist"}
                )
                continue
            params = {"nlist": nlist, "nprobe": nprobe}
            validate_ivf_params(params, n_train=nlist)  # basic shape check
            points.append((f"ivf_n{nlist}_p{nprobe}", "ivf", params))

    return points, excluded


def default_grid():
    """The default grid."""
    return build_grid(
        HNSW_DEFAULT_M,
        HNSW_DEFAULT_EF_CONSTRUCTION,
        list(HNSW_EF_SEARCH_GRID),
        list(IVF_NLIST_GRID),
        list(IVF_NPROBE_GRID),
    )


def _build_key(index_type: str, params: dict):
    """Identity of a constructed FAISS index for a grid point.

    Search-time parameters (HNSW ``efSearch``, IVF ``nprobe``) do not
    change the built index, so points that differ only in them share one
    build.
    """
    if index_type == "flat":
        return ("flat",)
    if index_type == "hnsw":
        return ("hnsw", int(params["M"]), int(params["efConstruction"]))
    return ("ivf", int(params["nlist"]))


def read_effective_params(info) -> dict:
    """Read back the parameters the built FAISS index actually uses.

    Only parameters the installed FAISS wrapper exposes are read back
    (HNSW ``M`` is not exposed by this build; the requested value is
    already recorded in ``params``).
    """
    index = info.index
    if info.index_type == "hnsw":
        return {
            "efConstruction": int(index.hnsw.efConstruction),
            "efSearch": int(index.hnsw.efSearch),
        }
    if info.index_type == "ivf":
        return {"nlist": int(index.nlist), "nprobe": int(index.nprobe)}
    return {}


def _load_encoder(model_name: str):
    """Lazily load the sentence-transformers encoder (never in tests)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_and_normalize(encoder, corpus_texts, query_texts):
    """Encode + L2-normalize corpus and queries exactly once.

    Returns float32 matrices shared by every grid point, so encoder work
    is excluded from all build/latency timings by construction.
    """
    import numpy as np

    corpus = l2_normalize(
        np.asarray(encoder.encode(corpus_texts), dtype=np.float64)
    ).astype(np.float32)
    queries = l2_normalize(
        np.asarray(encoder.encode(query_texts), dtype=np.float64)
    ).astype(np.float32)
    return corpus, queries


# --- main harness ---


def run_faiss_index_benchmark(
    dataset,
    k: int = 10,
    max_queries: int | None = None,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    grid=None,
    warmup_passes: int = 3,
    timed_passes: int = 5,
    relevance_check: bool = True,
    encoder=None,
    timer=time.perf_counter,
) -> dict:
    """Run the full index grid and return a JSON-serializable payload.

    ``grid`` is a list of ``(label, index_type, params)`` points (default:
    the study grid). ``encoder`` and ``timer`` exist for offline tests.
    """
    if grid is None:
        grid, excluded = default_grid()
    else:
        excluded = []
    grid = list(grid)
    if not grid:
        raise ValueError("at least one index configuration is required")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if isinstance(warmup_passes, bool) or not isinstance(warmup_passes, int) or warmup_passes < 0:
        raise ValueError(f"warmup_passes must be a non-negative integer, got {warmup_passes!r}")
    if isinstance(timed_passes, bool) or not isinstance(timed_passes, int) or timed_passes < 1:
        raise ValueError(f"timed_passes must be a positive integer, got {timed_passes!r}")

    # Fail clearly before any expensive encoding for invalid configurations.
    for label, index_type, params in grid:
        validate_index_type(index_type)
        if index_type == "hnsw":
            validate_hnsw_params(params)
        elif index_type == "ivf":
            validate_ivf_params(params, n_train=len(dataset.corpus))

    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)
    corpus_texts = [document["text"] for document in dataset.corpus]
    query_texts = [dataset.queries[query_id] for query_id in query_ids]

    encoder = encoder or _load_encoder(dense_model)
    corpus_embeddings, query_embeddings = _encode_and_normalize(
        encoder, corpus_texts, query_texts
    )

    # Build each distinct index once; grid points share builds.
    built = {}
    build_order = []
    for label, index_type, params in grid:
        key = _build_key(index_type, params)
        if key in built:
            continue
        start = timer()
        info = build_faiss_index(corpus_embeddings, index_type, params)
        build_time = timer() - start
        size = serialized_index_size(info.index)
        built[key] = (info, build_time, size)
        build_order.append(key)

    # Exact Flat reference (built once; un-timed only if not in the grid).
    flat_info = built[("flat",)][0] if ("flat",) in built else build_faiss_index(
        corpus_embeddings, index_type="flat"
    )
    flat_ids_per_query = [
        search_faiss_index(flat_info, query_embedding, k)[1][0].tolist()
        for query_embedding in query_embeddings
    ]

    results = {}
    for label, index_type, params in grid:
        info, build_time, size = built[_build_key(index_type, params)]

        # Apply the search-time parameter of this grid point.
        if index_type == "hnsw":
            info.index.hnsw.efSearch = int(params["efSearch"])
        elif index_type == "ivf":
            info.index.nprobe = int(params["nprobe"])
        effective = read_effective_params(info)

        # Untimed warmup passes over the identical precomputed embeddings.
        for _ in range(warmup_passes):
            for query_embedding in query_embeddings:
                search_faiss_index(info, query_embedding, k)

        per_query_samples = [[] for _ in query_embeddings]
        last_ids = [None] * len(query_embeddings)
        last_scores = [None] * len(query_embeddings)
        for _ in range(timed_passes):
            for index, query_embedding in enumerate(query_embeddings):
                start = timer()
                scores, indices = search_faiss_index(info, query_embedding, k)
                per_query_samples[index].append((timer() - start) * 1000.0)
                last_ids[index] = indices[0].tolist()
                last_scores[index] = scores[0].tolist()

        per_query_means = [
            sum(samples) / len(samples) for samples in per_query_samples
        ]
        latency = summarize_latencies(per_query_means)

        recalls = [
            ann_recall_at_k(ann_ids, flat_ids, k)
            for ann_ids, flat_ids in zip(last_ids, flat_ids_per_query)
        ]
        ann_recall = {
            "mean": sum(recalls) / len(recalls),
            "min": min(recalls),
            "queries_with_recall_1_0": sum(r >= 1.0 - 1e-12 for r in recalls),
            "queries_with_recall_ge_0_9": sum(r >= 0.9 for r in recalls),
        }

        qrel_ndcg = None
        if relevance_check:
            # FAISS ids are corpus positions; map them back to document ids.
            doc_ids = [document["id"] for document in dataset.corpus]
            per_query = []
            for query_id, ids, scores in zip(query_ids, last_ids, last_scores):
                order = sorted(
                    range(len(ids)), key=lambda i: (-scores[i], ids[i])
                )
                ranked = [doc_ids[ids[i]] for i in order]
                per_query.append(
                    evaluate_query(ranked, dataset.qrels[query_id], k=k)[f"ndcg@{k}"]
                )
            qrel_ndcg = sum(per_query) / len(per_query)

        results[label] = {
            "label": label,
            "index_type": index_type,
            "params": params,
            "effective_params": effective,
            "dimension": info.dimension,
            "n_vectors": info.n_vectors,
            "requires_training": info.requires_training,
            "trained": info.trained,
            "build_time_s": build_time,
            "serialized_size_bytes": size,
            "ann_recall": ann_recall,
            "latency": latency,
            "qrel_ndcg_at_k": qrel_ndcg,
        }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "dense_model": dense_model,
            "corpus_size": len(dataset.corpus),
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "embedding_dimension": int(corpus_embeddings.shape[1]),
            "k": k,
            "metric": "inner product over L2-normalized embeddings (equals cosine similarity)",
            "normalization": "L2-normalized corpus and query embeddings (float32)",
            "warmup_passes": warmup_passes,
            "timed_passes": timed_passes,
            "samples_per_query": timed_passes,
            "relevance_check": relevance_check,
            "grid": [
                {"label": label, "index_type": index_type, "params": params}
                for label, index_type, params in grid
            ],
            "excluded_combinations": excluded,
            "timer": getattr(timer, "__name__", type(timer).__name__),
            "timing_methodology": [
                "corpus and query texts are encoded exactly once per invocation; encoding and model loading are excluded from every timing",
                "build_time_s covers FAISS index construction only (training + add); indices are built once per build configuration and search-time parameters (HNSW efSearch, IVF nprobe) are varied without rebuilding",
                "per-query latency covers FAISS index.search on precomputed query embeddings; per-query latency = mean over timed passes; summary aggregates per-query means",
                "warmup passes are executed untimed",
                "ann_recall@k = mean over queries of |ANN top-k INTERSECT Flat top-k| / k (approximation vs exact Flat, not qrel relevance Recall@k)",
                "qrel_ndcg_at_k is a secondary relevance diagnostic on the top-k ids, not the primary metric",
                "deterministic inputs: fixed query order, identical embeddings across configurations, FAISS default clustering seed",
            ],
        },
        "results": results,
    }


def output_path_for(dataset_name: str, max_queries: int | None = None) -> str:
    """Dataset-scoped JSON path; smoke runs get their own file."""
    suffix = f"_smoke{max_queries}" if max_queries is not None else ""
    return f"outputs/{dataset_name}_faiss_index_benchmark{suffix}.json"


def format_table(payload: dict) -> str:
    """Render the index study as a fixed-width text table."""
    k = payload["metadata"]["k"]
    header = [
        "Config", f"ANN R@{k}", f">=0.9", "Build s", "Size MB",
        "Mean ms", "Median ms", "P95 ms", "qrel nDCG",
    ]
    rows = []
    for name, entry in payload["results"].items():
        latency = entry["latency"]
        ndcg = entry["qrel_ndcg_at_k"]
        rows.append(
            [
                name,
                f"{entry['ann_recall']['mean']:.4f}",
                f"{entry['ann_recall']['queries_with_recall_ge_0_9']}",
                f"{entry['build_time_s']:.3f}",
                f"{entry['serialized_size_bytes'] / 1e6:.2f}",
                f"{latency['mean_ms']:.3f}",
                f"{latency['median_ms']:.3f}",
                f"{latency['p95_ms']:.3f}",
                "n/a" if ndcg is None else f"{ndcg:.4f}",
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


# --- figure data preparation (payload-driven, never hard-coded) ---


def prepare_hnsw_curve(payload: dict):
    """(efSearch values, recalls, mean latencies) for the HNSW points."""
    points = [
        (
            int(entry["params"]["efSearch"]),
            entry["ann_recall"]["mean"],
            entry["latency"]["mean_ms"],
        )
        for entry in payload["results"].values()
        if entry["index_type"] == "hnsw"
    ]
    points.sort(key=lambda item: item[0])
    return (
        [point[0] for point in points],
        [point[1] for point in points],
        [point[2] for point in points],
    )


def prepare_ivf_curves(payload: dict) -> dict:
    """Per-nlist series: ``{nlist: (nprobes, recalls, latencies)}``."""
    series = defaultdict(list)
    for entry in payload["results"].values():
        if entry["index_type"] != "ivf":
            continue
        nlist = int(entry["params"]["nlist"])
        series[nlist].append(
            (
                int(entry["params"]["nprobe"]),
                entry["ann_recall"]["mean"],
                entry["latency"]["mean_ms"],
            )
        )
    prepared = {}
    for nlist in sorted(series):
        points = sorted(series[nlist], key=lambda item: item[0])
        prepared[nlist] = (
            [point[0] for point in points],
            [point[1] for point in points],
            [point[2] for point in points],
        )
    return prepared


def prepare_size_data(payload: dict):
    """(labels, sizes in MB) for every configuration, in result order."""
    labels = []
    sizes = []
    for name, entry in payload["results"].items():
        labels.append(name)
        sizes.append(entry["serialized_size_bytes"] / 1e6)
    return labels, sizes


def _family_color(family: str) -> str:
    return {"flat": "tab:gray", "hnsw": "tab:blue", "ivf": "tab:red"}[family]


# --- figures ---


def plot_recall_latency(payload: dict, output_dir="assets/figures") -> Path:
    """Scatter mean search latency vs ANN Recall@10 by index family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"flat": "s", "hnsw": "o", "ivf": "^"}
    fig, ax = plt.subplots(figsize=(8, 5))
    labeled_families = set()
    annotated = 0
    for name, entry in payload["results"].items():
        family = entry["index_type"]
        x = entry["latency"]["mean_ms"]
        y = entry["ann_recall"]["mean"]
        label = None
        if family not in labeled_families:
            label = family
            labeled_families.add(family)
        ax.scatter(x, y, marker=markers[family], color=_family_color(family),
                   label=label)
        text = None
        if family == "flat":
            text = "flat"
        elif family == "hnsw":
            text = f"efS{entry['params']['efSearch']}"
        elif family == "ivf":
            # annotate only the best (highest-recall) nprobe per nlist
            nlist = entry["params"]["nlist"]
            best = max(
                (e for e in payload["results"].values()
                 if e["index_type"] == "ivf" and e["params"]["nlist"] == nlist),
                key=lambda e: e["ann_recall"]["mean"],
            )
            if entry is best:
                text = f"n{nlist},p{entry['params']['nprobe']}"
        if text and annotated < 12:
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
            annotated += 1

    ax.set_xscale("log")
    ax.set_xlabel("Mean index-search latency (ms, log scale)")
    ax.set_ylabel(f"ANN Recall@{payload['metadata']['k']}")
    ax.set_ylim(0.9, 1.005)
    ax.set_title("FAISS index tradeoff on FiQA (Flat vs HNSW vs IVF)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(title="Index family")
    fig.tight_layout()

    path = Path(output_dir) / "faiss_recall_latency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_index_size(payload: dict, output_dir="assets/figures") -> Path:
    """Horizontal bars of serialized index size per configuration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, sizes = prepare_size_data(payload)
    colors = [
        _family_color(payload["results"][label]["index_type"]) for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, sizes, color=colors)
    ax.set_xlabel("Serialized FAISS index size (MB)")
    ax.set_title("FAISS index size on FiQA (index bytes only, no texts/models)")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    path = Path(output_dir) / "faiss_index_size.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hnsw_efsearch(payload: dict, output_dir="assets/figures") -> Path:
    """HNSW efSearch vs recall (top) and mean latency (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ef_searches, recalls, latencies = prepare_hnsw_curve(payload)

    fig, (ax_recall, ax_latency) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True
    )
    ax_recall.plot(ef_searches, recalls, marker="o", color="tab:blue")
    ax_recall.set_ylabel(f"ANN Recall@{payload['metadata']['k']}")
    ax_recall.set_title("HNSW efSearch sweep (M=32, efConstruction=200)")
    ax_recall.set_xticks(ef_searches)
    ax_recall.grid(True, linestyle=":", linewidth=0.5)

    ax_latency.plot(ef_searches, latencies, marker="s", color="tab:red")
    ax_latency.set_xlabel("efSearch")
    ax_latency.set_ylabel("Mean index-search latency (ms)")
    ax_latency.set_xticks(ef_searches)
    ax_latency.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    path = Path(output_dir) / "faiss_hnsw_efsearch.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ivf_nprobe(payload: dict, output_dir="assets/figures") -> Path:
    """IVF nprobe vs recall (top) and mean latency (bottom), per nlist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = prepare_ivf_curves(payload)

    fig, (ax_recall, ax_latency) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True
    )
    for nlist, (nprobes, recalls, latencies) in curves.items():
        ax_recall.plot(nprobes, recalls, marker="o", label=f"nlist={nlist}")
        ax_latency.plot(nprobes, latencies, marker="s", label=f"nlist={nlist}")
    ax_recall.set_ylabel(f"ANN Recall@{payload['metadata']['k']}")
    ax_recall.set_title("IVF nprobe sweep (IndexIVFFlat, IP)")
    ax_recall.set_xticks(sorted({nprobe for nprobes, _, _ in curves.values() for nprobe in nprobes}))
    ax_recall.grid(True, linestyle=":", linewidth=0.5)
    ax_recall.legend()

    ax_latency.set_xlabel("nprobe")
    ax_latency.set_ylabel("Mean index-search latency (ms)")
    ax_latency.grid(True, linestyle=":", linewidth=0.5)
    ax_latency.legend()
    fig.tight_layout()

    path = Path(output_dir) / "faiss_ivf_nprobe.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(payload: dict, output_dir="assets/figures") -> list:
    """Generate all figures from the actual payload."""
    return [
        plot_recall_latency(payload, output_dir),
        plot_index_size(payload, output_dir),
        plot_hnsw_efsearch(payload, output_dir),
        plot_ivf_nprobe(payload, output_dir),
    ]


# --- CLI ---


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="scifact", choices=list(DATASET_NAMES)
    )
    parser.add_argument("--k", type=int, default=10, help="retrieval top_k per query")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="evaluate only the first N queries (smoke tests)",
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=HNSW_DEFAULT_M, help="HNSW M"
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=HNSW_DEFAULT_EF_CONSTRUCTION,
        help="HNSW efConstruction",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        default=",".join(str(v) for v in HNSW_EF_SEARCH_GRID),
        help="comma-separated HNSW efSearch values",
    )
    parser.add_argument(
        "--ivf-nlist",
        default=",".join(str(v) for v in IVF_NLIST_GRID),
        help="comma-separated IVF nlist values",
    )
    parser.add_argument(
        "--ivf-nprobe",
        default=",".join(str(v) for v in IVF_NPROBE_GRID),
        help="comma-separated IVF nprobe values (nprobe > nlist is skipped)",
    )
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL_NAME,
        help="sentence-transformers model for the embeddings",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=3,
        help="untimed warmup passes over all query embeddings per index",
    )
    parser.add_argument(
        "--timed-passes",
        type=int,
        default=5,
        help="timed passes over all query embeddings per index",
    )
    parser.add_argument(
        "--skip-relevance",
        action="store_true",
        help="skip the secondary qrel nDCG@10 diagnostic",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="override the default dataset-specific JSON output path",
    )
    parser.add_argument(
        "--figures-dir",
        default="assets/figures",
        help="directory for the figures (full runs only)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    grid, excluded = build_grid(
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search_values=parse_int_list(args.hnsw_ef_search, "HNSW efSearch"),
        ivf_nlist_values=parse_int_list(args.ivf_nlist, "IVF nlist"),
        ivf_nprobe_values=parse_int_list(args.ivf_nprobe, "IVF nprobe"),
    )
    print(
        f"grid: {len(grid)} configurations "
        f"({len(excluded)} excluded IVF combinations)"
    )

    print(f"Loading {args.dataset} and encoding once (untimed) ...")
    dataset = load_dataset_by_name(args.dataset)
    print(
        f"documents={len(dataset.corpus)} qrel_query_ids={len(dataset.qrels)} "
        f"k={args.k} warmup_passes={args.warmup_passes} "
        f"timed_passes={args.timed_passes}"
    )

    payload = run_faiss_index_benchmark(
        dataset,
        k=args.k,
        max_queries=args.max_queries,
        dense_model=args.dense_model,
        grid=grid,
        warmup_passes=args.warmup_passes,
        timed_passes=args.timed_passes,
        relevance_check=not args.skip_relevance,
    )

    print()
    print(format_table(payload))

    output = args.output or output_path_for(args.dataset, args.max_queries)
    save_results(payload, output)
    print(f"\nSaved results to {output}")

    if args.max_queries is None:
        for path in plot_all(payload, args.figures_dir):
            print(f"Saved figure to {path}")
    else:
        print("(figures are generated only for full runs; smoke runs save JSON only)")


if __name__ == "__main__":
    main()