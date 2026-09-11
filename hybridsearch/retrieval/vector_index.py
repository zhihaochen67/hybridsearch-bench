"""Lightweight FAISS index abstraction for dense retrieval (Phase 14A).

Three index types over L2-normalized embeddings, all using the inner
product metric so scores equal cosine similarity:

- ``flat``: ``faiss.IndexFlatIP`` — the exact-search reference;
- ``hnsw``:  ``faiss.IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)`` — graph ANN,
  no training, parameters ``M`` / ``efConstruction`` / ``efSearch``;
- ``ivf``:   ``faiss.IndexIVFFlat(IndexFlatIP quantizer, d, nlist,
  METRIC_INNER_PRODUCT)`` — inverted-file ANN, trained on the corpus
  embeddings, parameters ``nlist`` / ``nprobe``.

Callers hand in already-L2-normalized embeddings; this module never
normalizes, re-encodes, or touches document text. ``build_faiss_index``
returns a small ``FaissIndexInfo`` record (the faiss object plus index
type, dimension, vector count, resolved configuration, and training
flags) so experiment layers can report build configuration and time only
the construction step.

ANN quality is measured against the exact Flat result — see
:func:`ann_recall_at_k`, which is deliberately named to avoid confusion
with qrel relevance ``recall@k``.

The ``faiss`` import is lazy, so importing this module is cheap and
offline-safe.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SUPPORTED_INDEX_TYPES = ("flat", "hnsw", "ivf")

HNSW_DEFAULT_M = 32
HNSW_DEFAULT_EF_CONSTRUCTION = 200
HNSW_DEFAULT_EF_SEARCH = 64

# Conservative defaults sized for the FiQA-scale corpus (~57k documents,
# ~112 vectors per cell at nlist=512). Smaller corpora must pass an
# explicit smaller nlist; the validator fails clearly otherwise.
IVF_DEFAULT_NLIST = 512
IVF_DEFAULT_NPROBE = 32


@dataclass
class FaissIndexInfo:
    """A built FAISS index plus the metadata needed to inspect/report it."""

    index: Any
    index_type: str
    dimension: int
    n_vectors: int
    params: dict
    requires_training: bool
    trained: bool
    _search_lock: Any = field(
        default_factory=threading.RLock, repr=False, compare=False
    )


# --- validation ---


def validate_index_type(index_type: str) -> str:
    """Return ``index_type`` or raise ``ValueError`` for unsupported types."""
    if index_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(
            f"unknown index type {index_type!r}; supported: "
            f"{', '.join(SUPPORTED_INDEX_TYPES)}"
        )
    return index_type


def _positive_int(params: dict, name: str, default: int) -> int:
    """Read a strictly-positive integer parameter (booleans rejected)."""
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _reject_unknown(params: dict, allowed, index_type: str) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"unknown {index_type} parameters: {sorted(unknown)}; "
            f"expected {sorted(allowed)}"
        )


def validate_hnsw_params(params: dict | None = None) -> dict:
    """Resolve and validate HNSW parameters (``M``, ``efConstruction``,
    ``efSearch``), filling the project defaults for missing keys."""
    params = dict(params or {})
    _reject_unknown(params, {"M", "efConstruction", "efSearch"}, "HNSW")

    resolved = {
        "M": _positive_int(params, "M", HNSW_DEFAULT_M),
        "efConstruction": _positive_int(
            params, "efConstruction", HNSW_DEFAULT_EF_CONSTRUCTION
        ),
        "efSearch": _positive_int(params, "efSearch", HNSW_DEFAULT_EF_SEARCH),
    }
    return resolved


def validate_ivf_params(params: dict | None, n_train: int) -> dict:
    """Resolve and validate IVF parameters (``nlist``, ``nprobe``).

    ``n_train`` is the number of corpus embeddings available for training;
    ``nlist > n_train`` (an unsplittable clustering problem) and
    ``nprobe > nlist`` fail with a clear ``ValueError``.
    """
    params = dict(params or {})
    _reject_unknown(params, {"nlist", "nprobe"}, "IVF")

    nlist = _positive_int(params, "nlist", IVF_DEFAULT_NLIST)
    nprobe = _positive_int(params, "nprobe", IVF_DEFAULT_NPROBE)

    if n_train < nlist:
        raise ValueError(
            f"nlist ({nlist}) exceeds the number of training vectors "
            f"({n_train}); reduce nlist for this corpus"
        )
    if nprobe > nlist:
        raise ValueError(f"nprobe ({nprobe}) must be <= nlist ({nlist})")

    return {"nlist": nlist, "nprobe": nprobe}


# --- construction ---


def build_faiss_index(
    embeddings,
    index_type: str = "flat",
    params: dict | None = None,
) -> FaissIndexInfo:
    """Build a FAISS index over L2-normalized embeddings.

    ``embeddings`` is an ``(n_vectors, dimension)`` float array that the
    caller has already L2-normalized (inner product then equals cosine
    similarity). ``params`` configures the index type; unset keys take
    the project defaults. Returns a :class:`FaissIndexInfo` whose
    ``requires_training`` flag tells whether the index needed training
    (IVF does, Flat and HNSW do not); ``trained`` is always ``True`` for
    a successfully built index.

    There is no silent fallback: an unsupported type, invalid parameters,
    or an insufficient training set raises ``ValueError``.
    """
    import faiss

    validate_index_type(index_type)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D (n_vectors, dimension) array")
    n_vectors, dimension = embeddings.shape
    if n_vectors == 0:
        raise ValueError("at least one embedding is required to build an index")

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return FaissIndexInfo(
            index=index,
            index_type="flat",
            dimension=dimension,
            n_vectors=n_vectors,
            params={},
            requires_training=False,
            trained=True,
        )

    if index_type == "hnsw":
        resolved = validate_hnsw_params(params)
        index = faiss.IndexHNSWFlat(
            dimension, resolved["M"], faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = resolved["efConstruction"]
        index.hnsw.efSearch = resolved["efSearch"]
        index.add(embeddings)
        return FaissIndexInfo(
            index=index,
            index_type="hnsw",
            dimension=dimension,
            n_vectors=n_vectors,
            params=resolved,
            requires_training=False,
            trained=True,
        )

    # ivf
    resolved = validate_ivf_params(params, n_vectors)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer, dimension, resolved["nlist"], faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = resolved["nprobe"]
    index.train(embeddings)
    index.add(embeddings)
    return FaissIndexInfo(
        index=index,
        index_type="ivf",
        dimension=dimension,
        n_vectors=n_vectors,
        params=resolved,
        requires_training=True,
        trained=True,
    )


# --- search / comparison ---


def search_faiss_index(index_info: FaissIndexInfo, query_embeddings, top_k: int):
    """Search one ``FaissIndexInfo`` and return ``(scores, indices)``.

    ``query_embeddings`` is one vector ``(dimension,)`` or a batch
    ``(n_queries, dimension)``; the caller is responsible for L2
    normalization (the same convention as index construction). ``top_k``
    is clamped to the number of indexed vectors. For HNSW, ``efSearch``
    is temporarily raised to ``top_k`` so a larger cutoff never silently
    returns fewer neighbours; the configured value is restored before the
    call returns.
    """
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if queries.ndim != 2:
        raise ValueError("query_embeddings must be 1-D or 2-D")
    if queries.shape[1] != index_info.dimension:
        raise ValueError(
            f"query embedding dimension {queries.shape[1]} does not match "
            f"index dimension {index_info.dimension}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    top_k = min(int(top_k), index_info.n_vectors)

    index = index_info.index
    if index_info.index_type == "hnsw":
        # FAISS exposes efSearch as mutable index state rather than a
        # per-call argument on the supported API. Serialize the temporary
        # override so concurrent requests cannot observe or restore one
        # another's value.
        with index_info._search_lock:
            configured_ef_search = index.hnsw.efSearch
            effective_ef_search = max(configured_ef_search, top_k)
            try:
                index.hnsw.efSearch = effective_ef_search
                return index.search(queries, top_k)
            finally:
                index.hnsw.efSearch = configured_ef_search

    return index.search(queries, top_k)


def ann_recall_at_k(ann_ids, flat_ids, k: int) -> float:
    """Fraction of the exact Flat top-k that an ANN index also retrieved.

        ann_recall@k = |ANN_top_k ∩ Flat_top_k| / k

    ``flat_ids`` are the ground-truth exact ``IndexFlatIP`` top-k ids for
    the same query embedding; ``ann_ids`` are the approximate index's
    top-k ids. This measures how well an ANN index approximates the exact
    nearest-neighbor result — it is deliberately NOT qrel relevance
    ``recall@k`` (no judgments are involved). Both id lists must contain
    at least ``k`` entries.

    Caveat: FAISS does not guarantee tie ordering. When several documents
    share the exact same score around the top-k cutoff, two equally valid
    top-k *sets* can differ and the overlap-based recall reflects that;
    real dense embeddings essentially never produce such exact ties.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    ann_ids = list(ann_ids)
    flat_ids = list(flat_ids)
    if len(ann_ids) < k or len(flat_ids) < k:
        raise ValueError(
            f"ann_recall_at_k needs at least k={k} ids from both lists "
            f"(got {len(ann_ids)} ANN and {len(flat_ids)} Flat ids)"
        )
    return len(set(ann_ids[:k]) & set(flat_ids[:k])) / k


def serialized_index_size(index) -> int:
    """Size in bytes of the serialized FAISS index alone.

    Uses ``faiss.write_index`` to a temporary file, reads the file size,
    and deletes it again — no document text or model weights are
    included. For small synthetic tests the temporary file lives in the
    system temp directory.
    """
    import faiss

    fd, path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, path)
        return os.path.getsize(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
