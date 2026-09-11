"""Dense retrieval with sentence-transformers embeddings and FAISS.

Cosine-similarity retrieval pipeline:

    corpus
      -> encode documents
      -> L2-normalize document embeddings
      -> build FAISS IndexFlatIP

    query
      -> encode query
      -> L2-normalize query embedding
      -> FAISS nearest-neighbor search
      -> top-k documents

Since FAISS ``IndexFlatIP`` computes inner products, L2-normalizing the
embeddings first makes those inner products equal to cosine similarities.
Documents are encoded once, when the index is built, and never re-encoded
per query.

The embedding model is configurable via ``model_name`` and can be replaced
with an injected ``encoder`` (anything exposing ``encode(texts) -> array``),
which keeps tests fast, offline, and deterministic.
"""

from __future__ import annotations

import numpy as np

from hybridsearch.retrieval.vector_index import (
    build_faiss_index,
    search_faiss_index,
    validate_index_type,
)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize ``vectors`` along the last axis.

    Zero vectors are returned unchanged (instead of NaN) because cosine
    similarity is undefined for them.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class DenseRetriever:
    """Dense retriever over a corpus of ``{"id": ..., "text": ...}`` documents.

    Parameters
    ----------
    corpus:
        List of documents. Each document must have an ``"id"`` and a ``"text"``.
    model_name:
        Name of the sentence-transformers model to load when no ``encoder``
        is injected. Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
    encoder:
        Optional encoder object with an ``encode(list_of_texts)`` method
        returning an ``(n_texts, dim)`` array. When provided, ``model_name``
        is only kept as metadata and no model is downloaded.
    index_type:
        FAISS index type: ``"flat"`` (default — exact inner-product
        search), ``"hnsw"``, or ``"ivf"``. See
        :mod:`hybridsearch.retrieval.vector_index`.
    index_params:
        Optional configuration dict for the index type (e.g. HNSW
        ``{"M": 32, "efConstruction": 200, "efSearch": 64}`` or IVF
        ``{"nlist": 512, "nprobe": 32}``). Unset keys take project
        defaults; invalid values raise ``ValueError``.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, corpus, model_name=None, encoder=None,
                 index_type="flat", index_params=None):
        self.corpus = list(corpus)
        if not self.corpus:
            raise ValueError("corpus must contain at least one document")

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.doc_ids = [document["id"] for document in self.corpus]
        self.texts = {document["id"]: document["text"] for document in self.corpus}

        self.index_type = validate_index_type(index_type)
        self.index_params = dict(index_params) if index_params else None

        self._encoder = encoder
        self._build_index()

    def _ensure_encoder(self):
        """Return the encoder, lazily loading sentence-transformers if needed."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def encode(self, texts):
        """Embed ``texts`` with the encoder and return an ``(n, dim)`` array."""
        embeddings = self._ensure_encoder().encode(texts)
        return np.asarray(embeddings, dtype=np.float64)

    def _build_index(self):
        """Encode the corpus once, normalize, and build the FAISS index."""
        corpus_texts = [document["text"] for document in self.corpus]
        self.document_embeddings = l2_normalize(self.encode(corpus_texts))

        self.index_info = build_faiss_index(
            self.document_embeddings,
            index_type=self.index_type,
            params=self.index_params,
        )
        self.index = self.index_info.index

    def rebuild_index(self, index_type=None, index_params=None):
        """Rebuild the FAISS index from the already-encoded corpus embeddings.

        The corpus is never re-encoded: this only reconstructs the index
        over ``self.document_embeddings``, which is useful when an
        experiment wants to switch index types (or parameters) without
        paying for another encoder pass.
        """
        if index_type is not None:
            self.index_type = validate_index_type(index_type)
        if index_type is not None or index_params is not None:
            self.index_params = dict(index_params) if index_params else None
        self.index_info = build_faiss_index(
            self.document_embeddings,
            index_type=self.index_type,
            params=self.index_params,
        )
        self.index = self.index_info.index

    def search(self, query: str, top_k: int = 10):
        """Return documents by descending score, then ascending document id.

        Flat search has exact cutoff semantics: if a score tie crosses the
        requested cutoff, all documents in that boundary tie are considered
        before the id tie-break is applied. Approximate indexes can only apply
        the ordering contract to the candidate set returned by FAISS.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        top_k = min(int(top_k), len(self.corpus))

        query_embedding = l2_normalize(self.encode([query]))[0]

        search_k = top_k
        if self.index_type == "flat" and top_k < len(self.corpus):
            # Fetch one result past the cutoff so an exact Flat index can
            # detect a tie that FAISS would otherwise truncate arbitrarily.
            search_k = top_k + 1

        scores, indices = search_faiss_index(
            self.index_info, query_embedding, search_k
        )

        if (
            self.index_type == "flat"
            and search_k > top_k
            and scores[0][top_k] == scores[0][top_k - 1]
        ):
            # IndexFlatIP range search is exact and uses a strict threshold.
            # Move one float32 step below the kth score to include the whole
            # boundary tie without repeatedly rerunning larger top-k scans.
            boundary_score = np.float32(scores[0][top_k - 1])
            threshold = np.nextafter(
                boundary_score, np.float32(-np.inf)
            )
            _, boundary_scores, boundary_indices = self.index.range_search(
                np.asarray(query_embedding[None, :], dtype=np.float32),
                float(threshold),
            )
            scores = boundary_scores[None, :]
            indices = boundary_indices[None, :]

        results = [
            {
                "id": self.doc_ids[int(index)],
                "text": self.texts[self.doc_ids[int(index)]],
                "score": float(score),
            }
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0
        ]

        # Flat has globally complete boundary ties after the expansion above.
        # For ANN indexes this deterministically orders only FAISS's returned
        # candidate set; approximate search cannot promise a global tie set.
        results.sort(key=lambda item: (-item["score"], item["id"]))

        return results[:top_k]


if __name__ == "__main__":
    from hybridsearch.data.toy import TOY_CORPUS

    retriever = DenseRetriever(TOY_CORPUS)

    for query in ["car repair", "machine learning"]:
        print(f"Query: {query}\n")

        for rank, result in enumerate(retriever.search(query, top_k=3), start=1):
            print(
                f"{rank}. {result['id']} | "
                f"score={result['score']:.4f} | "
                f"{result['text']}"
            )

        print()
