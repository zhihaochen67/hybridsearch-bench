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
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, corpus, model_name=None, encoder=None):
        self.corpus = list(corpus)
        if not self.corpus:
            raise ValueError("corpus must contain at least one document")

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.doc_ids = [document["id"] for document in self.corpus]
        self.texts = {document["id"]: document["text"] for document in self.corpus}

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
        import faiss

        corpus_texts = [document["text"] for document in self.corpus]
        self.document_embeddings = l2_normalize(self.encode(corpus_texts))

        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.document_embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10):
        """Return the top ``top_k`` documents by descending cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        top_k = min(int(top_k), len(self.corpus))

        query_embedding = l2_normalize(self.encode([query]))[0]

        scores, indices = self.index.search(
            query_embedding.astype(np.float32)[None, :], top_k
        )

        results = [
            {
                "id": self.doc_ids[int(index)],
                "text": self.texts[self.doc_ids[int(index)]],
                "score": float(score),
            }
            for score, index in zip(scores[0], indices[0])
        ]

        # FAISS does not guarantee tie ordering, so break ties on document id
        # to keep results fully deterministic.
        results.sort(key=lambda item: (-item["score"], item["id"]))

        return results


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