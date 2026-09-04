"""BM25 sparse retrieval implemented from scratch.

The classic Okapi BM25 scoring function:

    score(D, Q) = sum over q in Q of
        idf(q) * (tf(q, D) * (k1 + 1)) /
                 (tf(q, D) + k1 * (1 - b + b * |D| / avgdl))

with the standard smoothed inverse document frequency:

    idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
"""

import math
from collections import defaultdict

from hybridsearch.retrieval.simple import tokenize


class BM25:
    """In-memory BM25 index over a corpus of ``{"id": ..., "text": ...}`` documents."""

    def __init__(self, corpus, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self.doc_ids = [document["id"] for document in corpus]
        self.texts = {document["id"]: document["text"] for document in corpus}

        # term -> {doc_id: term frequency}
        self.term_frequencies = defaultdict(dict)
        # term -> number of documents containing the term
        self.document_frequencies = defaultdict(int)
        self.doc_lengths = {}

        for document in corpus:
            doc_id = document["id"]
            terms = tokenize(document["text"])
            self.doc_lengths[doc_id] = len(terms)

            frequencies = defaultdict(int)
            for term in terms:
                frequencies[term] += 1

            for term, frequency in frequencies.items():
                self.term_frequencies[term][doc_id] = frequency
                self.document_frequencies[term] += 1

        self.doc_count = len(corpus)
        self.average_document_length = (
            sum(self.doc_lengths.values()) / self.doc_count if self.doc_count else 0.0
        )

    def term_frequency(self, term: str, doc_id: str) -> int:
        """Number of times ``term`` occurs in document ``doc_id``."""
        return self.term_frequencies.get(term, {}).get(doc_id, 0)

    def document_frequency(self, term: str) -> int:
        """Number of documents containing ``term``."""
        return self.document_frequencies.get(term, 0)

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency of ``term``."""
        document_frequency = self.document_frequency(term)
        return math.log(
            1.0
            + (self.doc_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def score(self, query: str, doc_id: str) -> float:
        """BM25 score of ``query`` against the document ``doc_id``."""
        terms = tokenize(query)
        if not terms or self.average_document_length == 0.0:
            return 0.0

        doc_length = self.doc_lengths[doc_id]
        total = 0.0

        for term in terms:
            term_frequency = self.term_frequency(term, doc_id)
            if term_frequency == 0:
                continue

            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * doc_length / self.average_document_length
            )
            total += (
                self.idf(term)
                * (term_frequency * (self.k1 + 1.0))
                / denominator
            )

        return total

    def search(self, query: str, top_k: int = 10):
        """Return the top ``top_k`` documents, sorted by descending BM25 score."""
        results = [
            {
                "id": doc_id,
                "text": self.texts[doc_id],
                "score": self.score(query, doc_id),
            }
            for doc_id in self.doc_ids
        ]

        # Tie-break on id so results are fully deterministic.
        results.sort(key=lambda item: (-item["score"], item["id"]))

        return results[:top_k]


if __name__ == "__main__":
    from hybridsearch.data.toy import TOY_CORPUS

    bm25 = BM25(TOY_CORPUS)

    query = "machine learning"
    print(f"Query: {query}\n")

    for rank, result in enumerate(bm25.search(query, top_k=3), start=1):
        print(
            f"{rank}. {result['id']} | "
            f"score={result['score']:.4f} | "
            f"{result['text']}"
        )