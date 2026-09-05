"""SciFact dataset loading and normalization.

Loads the public BEIR SciFact dataset from Hugging Face and normalizes it
into the lightweight internal representation used across this project:

    corpus  -> [{"id": str, "text": str}, ...]
    queries -> {query_id: query_text}
    qrels   -> {query_id: {doc_id: relevance_grade}}

Document titles and bodies are joined with a single space. All ids are
preserved exactly as strings. Inconsistencies (duplicate ids, qrel entries
referencing unknown documents or queries, missing expected columns) fail
with ``ValueError`` instead of being silently dropped.

The generic normalization helpers live in ``hybridsearch.data.common`` and
are re-exported here so the SciFact public API is unchanged; the FiQA
loader (Phase 13) shares the same helpers.

The ``datasets`` import is lazy, so importing this module never touches
the network and unit tests stay fully offline.
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

SCIFACT_HF_PATH = "BeIR/scifact"
SCIFACT_QRELS_HF_PATH = "BeIR/scifact-qrels"
SCIFACT_QRELS_SPLIT = "test"


@dataclass
class SciFactDataset(Dataset):
    """Normalized SciFact: corpus list, query map, and qrel map."""

    name: str = "scifact"
    source: str = field(default=SCIFACT_HF_PATH)


def load_scifact() -> SciFactDataset:
    """Download (once, cached) and normalize BEIR SciFact from Hugging Face.

    Actual published layout (verified against the live datasets):

    - ``BeIR/scifact`` config ``corpus``, split ``corpus``: columns
      ``_id``, ``title``, ``text``;
    - ``BeIR/scifact`` config ``queries``, split ``queries``: columns
      ``_id``, ``title``, ``text`` (query text lives in ``text``);
    - ``BeIR/scifact-qrels`` split ``test``: columns ``query-id``,
      ``corpus-id``, ``score`` (integer ids, normalized to strings here).

    Expected columns are checked and a clear ``ValueError`` is raised if
    the schema ever changes. Requires the ``datasets`` package and network
    access on first use.
    """
    from datasets import load_dataset

    raw_splits = {
        "corpus": load_dataset(
            SCIFACT_HF_PATH, "corpus", split="corpus", trust_remote_code=False
        ),
        "queries": load_dataset(
            SCIFACT_HF_PATH, "queries", split="queries", trust_remote_code=False
        ),
        "qrels": load_dataset(
            SCIFACT_QRELS_HF_PATH, split=SCIFACT_QRELS_SPLIT, trust_remote_code=False
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
                f"SciFact {split!r} split is missing expected columns: "
                f"{sorted(missing)}"
            )

    corpus = normalize_corpus(raw_splits["corpus"])
    queries = normalize_queries(raw_splits["queries"])
    corpus_ids = {document["id"] for document in corpus}
    query_ids = set(queries)

    qrels = normalize_qrels(
        raw_splits["qrels"],
        corpus_ids=corpus_ids,
        query_ids=query_ids,
    )

    return SciFactDataset(corpus=corpus, queries=queries, qrels=qrels)


if __name__ == "__main__":
    dataset = load_scifact()

    print(f"source: {dataset.source}")
    print(f"documents: {len(dataset.corpus)}")
    print(f"queries: {len(dataset.queries)}")
    print(f"qrel query ids: {len(dataset.qrels)}")
    print(f"total judged pairs: "
          f"{sum(len(entries) for entries in dataset.qrels.values())}")
    print(f"sample document: {dataset.corpus[0]}")
    first_query = next(iter(dataset.queries))
    print(f"sample query: {first_query!r} -> {dataset.queries[first_query]!r}")