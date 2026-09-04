"""Score-fusion primitives for hybrid retrieval.

Two scale-free fusion strategies over two ranked lists of documents:

- weighted fusion: per-list min-max normalization followed by
  ``alpha * normalized_sparse + (1 - alpha) * normalized_dense``;
- reciprocal rank fusion (RRF): ``sum_i 1 / (k + rank_i(d))`` over 1-based
  rank positions.

Both operate over the *union* of the documents in their inputs and attach
per-document diagnostics to the fused scores. The functions are pure: they
know nothing about retriever implementations, only score or rank maps.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize ``scores`` per query.

        normalized(d) = (score(d) - min) / (max - min)

    Constant-score case: when all scores are equal there is no spread to
    normalize by. The normalized value is deterministic: ``1.0`` when the
    constant score is positive (every candidate is a genuine best-of-list
    match, so the list keeps its retrieval signal) and ``0.0`` otherwise
    (all-zero scores, a single zero, or a constant negative score carry no
    positive evidence — equivalent to "not retrieved"). This also avoids
    division by zero.
    """
    if not scores:
        return {}

    minimum = min(scores.values())
    maximum = max(scores.values())

    if maximum == minimum:
        value = 1.0 if maximum > 0 else 0.0
        return {doc_id: value for doc_id in scores}

    span = maximum - minimum
    return {doc_id: (score - minimum) / span for doc_id, score in scores.items()}


def weighted_fusion(
    sparse_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
    alpha: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Fuse sparse and dense scores with weighted normalized summation.

    Each list is min-max normalized independently, then combined over the
    union of documents:

        score(d) = alpha * normalized_sparse(d)
                 + (1 - alpha) * normalized_dense(d)

    A document missing from one list contributes ``0.0`` (raw *and*
    normalized) from that list.

    Returns ``{doc_id: component dict}`` where each component dict has the
    keys ``score`` (final fused score), ``sparse_score``, ``dense_score``,
    ``normalized_sparse_score``, and ``normalized_dense_score``.

    Raises ``ValueError`` if ``alpha`` is not in ``[0.0, 1.0]``.
    """
    if not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha!r}")

    normalized_sparse = min_max_normalize(sparse_scores)
    normalized_dense = min_max_normalize(dense_scores)

    fused: dict[str, dict[str, float]] = {}
    for doc_id in set(sparse_scores) | set(dense_scores):
        sparse_norm = normalized_sparse.get(doc_id, 0.0)
        dense_norm = normalized_dense.get(doc_id, 0.0)

        fused[doc_id] = {
            "score": alpha * sparse_norm + (1.0 - alpha) * dense_norm,
            "sparse_score": float(sparse_scores.get(doc_id, 0.0)),
            "dense_score": float(dense_scores.get(doc_id, 0.0)),
            "normalized_sparse_score": sparse_norm,
            "normalized_dense_score": dense_norm,
        }

    return fused


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: float = 60.0,
) -> dict[str, dict]:
    """Fuse ranked document-id lists with reciprocal rank fusion.

        RRF(d) = sum over lists i of 1 / (k + rank_i(d))

    Ranks are 1-based positions. Fusion is over the union: a document that
    appears in only one list keeps only that list's contribution, and a
    document duplicated inside a single list uses its best (lowest) rank.

    Returns ``{doc_id: {"score": rrf_score, "ranks": [rank_0, rank_1, ...]}}``
    where ``ranks`` parallels the input lists (``None`` for lists in which
    the document does not appear).

    Raises ``ValueError`` if ``k`` is not a positive number.
    """
    if not isinstance(k, (int, float)) or not k > 0:
        raise ValueError(f"k must be a positive number, got {k!r}")

    rrf_scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, list] = defaultdict(lambda: [None] * len(ranked_lists))

    for list_index, ranked_list in enumerate(ranked_lists):
        best_rank: dict[str, int] = {}
        for position, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in best_rank:
                best_rank[doc_id] = position

        for doc_id, rank in best_rank.items():
            rrf_scores[doc_id] += 1.0 / (k + rank)
            ranks[doc_id][list_index] = rank

    return {
        doc_id: {"score": rrf_scores[doc_id], "ranks": ranks[doc_id]}
        for doc_id in rrf_scores
    }


def rank_by_score(scores: Mapping[str, float]) -> list[str]:
    """Order document ids deterministically.

    Descending score first, then ascending document id as the tie-break, so
    repeated fusions of identical inputs always produce identical orders.
    """
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
