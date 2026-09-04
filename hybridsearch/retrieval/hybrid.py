"""Hybrid retrieval: combine sparse (BM25) and dense retrieval.

Two fusion strategies are supported:

- ``method="weighted"``: min-max normalized weighted summation,
  ``alpha * normalized_bm25 + (1 - alpha) * normalized_dense``
  (see :func:`hybridsearch.retrieval.fusion.weighted_fusion`);
- ``method="rrf"``: reciprocal rank fusion over the two ranked lists
  (see :func:`hybridsearch.retrieval.fusion.reciprocal_rank_fusion`).

The hybrid retriever asks each underlying retriever for ``candidate_k``
candidates, fuses over the union, and returns the final ``top_k`` documents.
The dense corpus is encoded once when the ``DenseRetriever`` is built;
hybrid search never re-encodes it, it only reuses the existing search APIs.
"""

from __future__ import annotations

from hybridsearch.retrieval.fusion import (
    rank_by_score,
    reciprocal_rank_fusion,
    weighted_fusion,
)


class HybridRetriever:
    """Fuse a sparse and a dense retriever.

    Parameters
    ----------
    sparse_retriever, dense_retriever:
        Objects exposing ``search(query, top_k)`` returning a list of
        ``{"id": ..., "text": ..., "score": ...}`` dicts, such as ``BM25``
        and ``DenseRetriever``.
    method:
        ``"weighted"`` or ``"rrf"``.
    alpha:
        Sparse weight for ``method="weighted"``. Must be in ``[0.0, 1.0]``.
        Validated on construction even when ``method="rrf"``, where it is
        unused.
    k:
        RRF constant for ``method="rrf"``. Must be positive. Validated on
        construction even when ``method="weighted"``, where it is unused.
    candidate_k:
        Number of candidates fetched from each retriever before fusion.
        When ``candidate_k < top_k`` the fetch size is raised to ``top_k``
        so the final ranking is not artificially starved.
    """

    def __init__(
        self,
        sparse_retriever,
        dense_retriever,
        method: str = "weighted",
        alpha: float = 0.5,
        k: float = 60.0,
        candidate_k: int = 50,
    ):
        if not isinstance(method, str) or method not in ("weighted", "rrf"):
            raise ValueError(f"method must be 'weighted' or 'rrf', got {method!r}")
        if not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha!r}")
        if not isinstance(k, (int, float)) or not k > 0:
            raise ValueError(f"k must be a positive number, got {k!r}")
        if not isinstance(candidate_k, int) or candidate_k <= 0:
            raise ValueError(
                f"candidate_k must be a positive integer, got {candidate_k!r}"
            )

        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.method = method
        self.alpha = float(alpha)
        self.k = float(k)
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 10):
        """Return the top ``top_k`` documents after fusion.

        Every result contains ``id``, ``text``, and the fused ``score``.

        ``method="weighted"`` adds the diagnostics ``bm25_score``,
        ``dense_score``, ``normalized_bm25_score``, and
        ``normalized_dense_score``; a document missing from one retriever's
        candidate list contributes raw and normalized ``0.0`` from it.

        ``method="rrf"`` adds the diagnostics ``sparse_rank`` and
        ``dense_rank`` (1-based, ``None`` when the document is absent from
        that retriever's candidate list). Sparse candidates with
        non-positive scores are excluded from the sparse ranked list before
        fusion: BM25 nonmatches score exactly 0, so their document-id
        tie-break order never becomes rank signal (such documents get
        ``sparse_rank=None``). Dense scores are always used as-is —
        negative cosine similarities still rank normally.

        Ties on the fused score are broken deterministically on ascending
        document id.

        Behavior notes: an empty/blank query raises ``ValueError`` (like
        ``DenseRetriever``); ``top_k <= 0`` raises ``ValueError``; and when
        ``candidate_k < top_k`` the per-retriever fetch is raised to
        ``top_k`` so up to ``top_k`` fused results can still be produced.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        top_k = int(top_k)
        fetch_k = max(self.candidate_k, top_k)

        sparse_results = self.sparse_retriever.search(query, top_k=fetch_k)
        dense_results = self.dense_retriever.search(query, top_k=fetch_k)

        if self.method == "weighted":
            fused = weighted_fusion(
                {result["id"]: result["score"] for result in sparse_results},
                {result["id"]: result["score"] for result in dense_results},
                self.alpha,
            )
        else:
            # BM25-style nonmatches score exactly 0.0; keep them out of the
            # sparse ranked list so document-id tie-breaking of zero-score
            # results cannot inject artificial rank signal into RRF. The
            # dense list is passed through untouched (cosine similarities
            # may legitimately be negative and still rank).
            sparse_ids = [
                result["id"] for result in sparse_results if result["score"] > 0
            ]
            fused = reciprocal_rank_fusion(
                [
                    sparse_ids,
                    [result["id"] for result in dense_results],
                ],
                self.k,
            )

        texts = {
            result["id"]: result["text"]
            for result in [*sparse_results, *dense_results]
            if "text" in result
        }

        ordered = rank_by_score(
            {doc_id: fused[doc_id]["score"] for doc_id in fused}
        )

        if self.method == "weighted":
            return [
                {
                    "id": doc_id,
                    "text": texts.get(doc_id),
                    "score": fused[doc_id]["score"],
                    "bm25_score": fused[doc_id]["sparse_score"],
                    "dense_score": fused[doc_id]["dense_score"],
                    "normalized_bm25_score": fused[doc_id][
                        "normalized_sparse_score"
                    ],
                    "normalized_dense_score": fused[doc_id][
                        "normalized_dense_score"
                    ],
                }
                for doc_id in ordered[:top_k]
            ]

        return [
            {
                "id": doc_id,
                "text": texts.get(doc_id),
                "score": fused[doc_id]["score"],
                "sparse_rank": fused[doc_id]["ranks"][0],
                "dense_rank": fused[doc_id]["ranks"][1],
            }
            for doc_id in ordered[:top_k]
        ]


if __name__ == "__main__":
    from hybridsearch.data.toy import TOY_CORPUS
    from hybridsearch.retrieval.bm25 import BM25
    from hybridsearch.retrieval.dense import DenseRetriever

    print("Building sparse (BM25) and dense (sentence-transformers) retrievers ...")
    sparse = BM25(TOY_CORPUS)
    dense = DenseRetriever(TOY_CORPUS)

    queries = ["car repair", "machine learning"]

    for method in ("weighted", "rrf"):
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            method=method,
            alpha=0.5,
        )
        print(f"=== {method} fusion (alpha={retriever.alpha}, k={retriever.k}) ===")
        for query in queries:
            print(f"\nQuery: {query}")
            for rank, result in enumerate(retriever.search(query, top_k=3), start=1):
                if method == "weighted":
                    detail = (
                        f"bm25={result['bm25_score']:.4f} "
                        f"dense={result['dense_score']:.4f} "
                        f"norm_bm25={result['normalized_bm25_score']:.4f} "
                        f"norm_dense={result['normalized_dense_score']:.4f}"
                    )
                else:
                    detail = (
                        f"sparse_rank={result['sparse_rank']} "
                        f"dense_rank={result['dense_rank']}"
                    )
                print(
                    f"{rank}. {result['id']} | score={result['score']:.4f} | "
                    f"{detail} | {result['text']}"
                )
        print()
