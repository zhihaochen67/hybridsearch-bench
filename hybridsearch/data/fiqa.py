"""FiQA dataset loading and normalization (Phase 13).

Loads the public BEIR FiQA dataset from Hugging Face and normalizes it
into the same internal representation already used by SciFact:

    corpus  -> [{"id": str, "text": str}, ...]
    queries -> {query_id: query_text}
    qrels   -> {query_id: {doc_id: relevance_grade}}

Actual published layout (verified against the live datasets):

- ``BeIR/fiqa`` config ``corpus``, split ``corpus``: columns ``_id``,
  ``title``, ``text`` (57,638 documents; every title is empty and 38
  documents have empty text — those records are kept as-is);
- ``BeIR/fiqa`` config ``queries``, split ``queries``: columns ``_id``,
  ``title``, ``text`` (6,648 queries; the query text lives in ``text``);
- ``BeIR/fiqa-qrels`` split ``test``: columns ``query-id``, ``corpus-id``,
  ``score`` (648 query ids, 1,706 judged pairs, binary relevance;
  integer ids are normalized to strings here).

Ids are preserved exactly as strings. Duplicate ids and qrel entries
referencing unknown documents or queries raise ``ValueError`` instead of
being silently dropped. The ``datasets`` import is lazy, so importing
this module never touches the network and unit tests stay fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hybridsearch.data.common import (
    CORPUS_ID_COLUMN,
    CORPUS_TEXT_COLUMN,
    CORPUS_TITLE_COLUMN,
    QREL_DOC_COLUMN,
    QREL_QUERY_COLUMN,
    QREL_SCORE_COLUMN,
    QUERY_ID_COLUMN,
    QUERY_TEXT_COLUMN,
    Dataset,
    normalize_corpus,
    normalize_qrels,
    normalize_queries,
)

FIQA_HF_PATH = "BeIR/fiqa"
FIQA_QRELS_HF_PATH = "BeIR/fiqa-qrels"
FIQA_QRELS_SPLIT = "test"


@dataclass
class FiQADataset(Dataset):
    """Normalized FiQA: corpus list, query map, and qrel map."""

    name: str = "fiqa"
    source: str = field(default=FIQA_HF_PATH)


def load_fiqa() -> FiQADataset:
    """Download (once, cached) and normalize BEIR FiQA from Hugging Face.

    Expected columns are checked and a clear ``ValueError`` is raised if
    the schema ever changes. Requires the ``datasets`` package and network
    access on first use.
    """
    from datasets import load_dataset

    raw_splits = {
        "corpus": load_dataset(
            FIQA_HF_PATH, "corpus", split="corpus", trust_remote_code=False
        ),
        "queries": load_dataset(
            FIQA_HF_PATH, "queries", split="queries", trust_remote_code=False
        ),
        "qrels": load_dataset(
            FIQA_QRELS_HF_PATH, split=FIQA_QRELS_SPLIT, trust_remote_code=False
        ),
    }

    expected_columns = {
        "corpus": {CORPUS_ID_COLUMN, CORPUS_TITLE_COLUMN, CORPUS_TEXT_COLUMN},
        "queries": {QUERY_ID_COLUMN, QUERY_TEXT_COLUMN},
        "qrels": {QREL_QUERY_COLUMN, QREL_DOC_COLUMN, QREL_SCORE_COLUMN},
    }
    for split, raw_split in raw_splits.items():
        missing = expected_columns[split] - set(raw_split.column_names)
        if missing:
            raise ValueError(
                f"FiQA {split!r} split is missing expected columns: "
                f"{sorted(missing)}"
            )

    # FiQA's published corpus genuinely contains 38 empty-text posts (see
    # the module docstring); keep those records rather than dropping them.
    corpus = normalize_corpus(raw_splits["corpus"], require_text=False)
    queries = normalize_queries(raw_splits["queries"])
    corpus_ids = {document["id"] for document in corpus}
    query_ids = set(queries)

    qrels = normalize_qrels(
        raw_splits["qrels"],
        corpus_ids=corpus_ids,
        query_ids=query_ids,
    )

    return FiQADataset(corpus=corpus, queries=queries, qrels=qrels)


if __name__ == "__main__":
    dataset = load_fiqa()

    print(f"source: {dataset.source}")
    print(f"documents: {len(dataset.corpus)}")
    print(f"queries: {len(dataset.queries)}")
    print(f"qrel query ids: {len(dataset.qrels)}")
    print(
        "total judged pairs: "
        f"{sum(len(entries) for entries in dataset.qrels.values())}"
    )
    print(f"sample document: {dataset.corpus[0]}")
    first_query = next(iter(dataset.queries))
    print(f"sample query: {first_query!r} -> {dataset.queries[first_query]!r}")