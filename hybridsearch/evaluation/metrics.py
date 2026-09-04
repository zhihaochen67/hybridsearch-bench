"""Information-retrieval evaluation metrics implemented from scratch.

Metrics operate on ranked document-id lists and qrels dicts of
``{doc_id: relevance}``:

- ``relevance > 0`` counts as relevant;
- ``relevance == 0`` counts as non-relevant;
- graded relevance is preserved for nDCG;
- document ids missing from ``qrels`` count as relevance 0.

Duplicate document ids in ``ranked_ids`` are rejected with ``ValueError``
to prevent silently inflated or misleading metric values.

``mrr_at_k`` is the reciprocal rank of a single query; dataset-level MRR
is the arithmetic mean across queries (see ``mean_metrics``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _validate_ranked_ids(ranked_ids: Sequence[str]) -> list:
    ids = list(ranked_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("ranked_ids must not contain duplicate document ids")
    return ids


def _gain(relevance: float) -> float:
    """nDCG gain of one relevance grade: ``2^rel - 1``."""
    return 2.0 ** relevance - 1.0


def _dcg(relevances: Sequence[float]) -> float:
    """DCG with the standard logarithmic discount ``1 / log2(rank + 1)``."""
    return sum(
        _gain(relevance) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def precision_at_k(ranked_ids, qrels, k: int) -> float:
    """Fraction of the first ``k`` positions occupied by relevant documents.

    The denominator is always ``k``: when fewer than ``k`` documents were
    retrieved, the missing positions count as non-relevant.
    """
    k = _validate_k(k)
    ids = _validate_ranked_ids(ranked_ids)

    relevant = sum(1 for doc_id in ids[:k] if qrels.get(doc_id, 0) > 0)
    return relevant / k


def recall_at_k(ranked_ids, qrels, k: int) -> float:
    """Fraction of all relevant documents retrieved within the top ``k``.

    Returns ``0.0`` deterministically when the qrels contain no relevant
    documents.
    """
    k = _validate_k(k)
    ids = _validate_ranked_ids(ranked_ids)

    total_relevant = sum(1 for relevance in qrels.values() if relevance > 0)
    if total_relevant == 0:
        return 0.0

    retrieved = sum(1 for doc_id in ids[:k] if qrels.get(doc_id, 0) > 0)
    return retrieved / total_relevant


def mrr_at_k(ranked_ids, qrels, k: int) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``.

    Ranks are 1-based and only the first relevant document counts. Despite
    the name, this is the reciprocal rank of a single query; dataset-level
    MRR is the arithmetic mean across queries (see ``mean_metrics``).
    Returns ``0.0`` when nothing in the top ``k`` is relevant.
    """
    k = _validate_k(k)
    ids = _validate_ranked_ids(ranked_ids)

    for rank, doc_id in enumerate(ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, qrels, k: int) -> float:
    """Normalized discounted cumulative gain over the top ``k`` positions.

    Uses graded relevance with ``gain(rel) = 2^rel - 1`` and the discount
    ``1 / log2(i + 1)`` at 1-based rank ``i``. The ideal DCG sorts all
    positively-relevant qrel values in descending order and applies the
    same formula to the top ``k`` of that ordering. Returns ``0.0`` when
    ``IDCG == 0`` (no relevant documents at all).
    """
    k = _validate_k(k)
    ids = _validate_ranked_ids(ranked_ids)

    ranked_relevances = [qrels.get(doc_id, 0) for doc_id in ids[:k]]
    dcg = _dcg(ranked_relevances)

    ideal = sorted(
        (relevance for relevance in qrels.values() if relevance > 0),
        reverse=True,
    )
    idcg = _dcg(ideal[:k])

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(ranked_ids, qrels, k: int = 10) -> dict[str, float]:
    """Compute precision, recall, MRR, and nDCG for one query.

    Returns a dict keyed by the metric name including ``k``, e.g.
    ``{"precision@10": ..., "recall@10": ..., "mrr@10": ...,
    "ndcg@10": ...}``.
    """
    return {
        f"precision@{k}": precision_at_k(ranked_ids, qrels, k),
        f"recall@{k}": recall_at_k(ranked_ids, qrels, k),
        f"mrr@{k}": mrr_at_k(ranked_ids, qrels, k),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, qrels, k),
    }


def mean_metrics(per_query_metrics) -> dict[str, float]:
    """Arithmetic mean of per-query metric dicts across queries.

    Every input dict must expose the same keys (as produced by
    ``evaluate_query``). Deterministic: iteration follows insertion order.
    An empty input returns ``{}``.
    """
    if not per_query_metrics:
        return {}

    keys = per_query_metrics[0].keys()
    return {
        key: sum(metrics[key] for metrics in per_query_metrics)
        / len(per_query_metrics)
        for key in keys
    }


if __name__ == "__main__":
    ranked_ids = ["D2", "D1", "D3"]
    qrels = {"D1": 2, "D2": 1, "D3": 0, "D4": 1}

    per_query = evaluate_query(ranked_ids, qrels, k=3)
    print(f"ranked_ids = {ranked_ids}")
    print(f"qrels = {qrels}\n")
    for key, value in per_query.items():
        print(f"{key} = {value:.6f}")

    second = evaluate_query(["D1", "D4", "D2"], qrels, k=3)
    print("\nmean across two queries:")
    for key, value in mean_metrics([per_query, second]).items():
        print(f"  {key} = {value:.6f}")
