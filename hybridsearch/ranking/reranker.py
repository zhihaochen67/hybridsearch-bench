"""Neural cross-encoder reranking.

A cross-encoder jointly encodes ``(query, document)`` pairs instead of
independently embedding query and document, which makes it a strong
second-stage reranker over candidates from BM25, dense, or hybrid
retrieval.

The pipeline is deliberately compositional — the reranker knows nothing
about how candidates were retrieved:

    candidates = retriever.search(query, top_k=candidate_k)
    final = reranker.rerank(query, candidates, top_k=final_k)

Candidates only need ``{"id": ..., "text": ...}`` dicts. Every existing
field is preserved and ``reranker_score`` is added.

The real model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` via
``sentence_transformers.CrossEncoder``. It is loaded lazily on first use
and only when no ``scorer`` is injected, so tests stay offline and
deterministic.
"""

from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    """Rerank candidate documents with a pretrained cross-encoder.

    Parameters
    ----------
    model_name:
        Name of the sentence-transformers cross-encoder to load when no
        ``scorer`` is injected. Defaults to
        ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    scorer:
        Optional injected scoring model with a ``predict(list_of_pairs)``
        method returning one score per ``(query, text)`` pair. When
        provided, ``model_name`` is only kept as metadata and the real
        model is never loaded.
    """

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name=None, scorer=None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self._scorer = scorer

    def _ensure_scorer(self):
        """Return the scorer, lazily loading the real model if needed.

        The model is loaded at most once: later calls reuse the same
        scorer instance.
        """
        if self._scorer is None:
            from sentence_transformers import CrossEncoder

            self._scorer = CrossEncoder(self.model_name)
        return self._scorer

    def rerank(self, query: str, candidates, top_k: int = 10):
        """Rerank ``candidates`` for ``query`` and return the top ``top_k``.

        All ``(query, text)`` pairs are scored in a single batch call and
        every result is a copy of its candidate dict with the additional
        ``reranker_score`` field, sorted by descending reranker score with
        ascending document id as the deterministic tie-break.

        Behavior notes, consistent with the existing retrievers:
        an empty/blank query raises ``ValueError``; an empty candidate
        list raises ``ValueError``; ``top_k <= 0`` raises ``ValueError``;
        and ``top_k`` beyond the candidate count is clamped so every
        candidate is returned.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not candidates:
            raise ValueError("candidates must be a non-empty list")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        top_k = min(int(top_k), len(candidates))

        pairs = []
        for candidate in candidates:
            if "id" not in candidate or "text" not in candidate:
                raise ValueError(
                    "each candidate must be a dict with 'id' and 'text' keys"
                )
            pairs.append((query, candidate["text"]))

        raw_scores = self._ensure_scorer().predict(pairs)
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)

        if scores.shape[0] != len(candidates):
            raise ValueError(
                f"scorer returned {scores.shape[0]} scores for "
                f"{len(candidates)} candidates"
            )

        results = [
            {**candidate, "reranker_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]

        # Descending reranker score, ascending id for deterministic ties.
        results.sort(key=lambda item: (-item["reranker_score"], item["id"]))

        return results[:top_k]


if __name__ == "__main__":
    from hybridsearch.data.toy import TOY_CORPUS
    from hybridsearch.retrieval.bm25 import BM25
    from hybridsearch.retrieval.dense import DenseRetriever
    from hybridsearch.retrieval.hybrid import HybridRetriever

    sparse = BM25(TOY_CORPUS)
    dense = DenseRetriever(TOY_CORPUS)
    reranker = CrossEncoderReranker()

    for method in ("weighted", "rrf"):
        hybrid = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            method=method,
            alpha=0.5,
        )
        for query in ("car repair", "machine learning"):
            candidates = hybrid.search(query, top_k=4)
            final = reranker.rerank(query, candidates, top_k=3)

            print(f"Method: {method} | Query: {query}\n")
            print("Before reranking (hybrid candidates, top 4):")
            for rank, candidate in enumerate(candidates, start=1):
                print(
                    f"  {rank}. {candidate['id']} | hybrid_score="
                    f"{candidate['score']:.4f} | {candidate['text']}"
                )
            print("\nAfter reranking (cross-encoder, top 3):")
            for rank, result in enumerate(final, start=1):
                print(
                    f"  {rank}. {result['id']} | reranker_score="
                    f"{result['reranker_score']:.4f} | {result['text']}"
                )
            print()
