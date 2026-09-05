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

The ``datasets`` import is lazy, so importing this module never touches
the network and unit tests stay fully offline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field
from typing import Any

SCIFACT_HF_PATH = "BeIR/scifact"
SCIFACT_QRELS_HF_PATH = "BeIR/scifact-qrels"
SCIFACT_QRELS_SPLIT = "test"

# Column names of the BEIR SciFact splits on Hugging Face (verified against
# the published schema; loading fails clearly if they ever change).
CORPUS_ID_COLUMN = "_id"
CORPUS_TITLE_COLUMN = "title"
CORPUS_TEXT_COLUMN = "text"
QUERY_ID_COLUMN = "_id"
QUERY_TEXT_COLUMN = "text"
QREL_QUERY_COLUMN = "query-id"
QREL_DOC_COLUMN = "corpus-id"
QREL_SCORE_COLUMN = "score"


@dataclass
class SciFactDataset:
    """Normalized SciFact: corpus list, query map, and qrel map."""

    corpus: list[dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str = "scifact"
    source: str = field(default=SCIFACT_HF_PATH)

    def corpus_ids(self) -> set[str]:
        """Document ids of the corpus."""
        return {document["id"] for document in self.corpus}

    def query_ids(self) -> set[str]:
        """Ids of all loaded queries."""
        return set(self.queries)


# --- field helpers ---


def _string_field(record: Mapping[str, Any], column: str, kind: str) -> str:
    if column not in record:
        raise ValueError(f"{kind} record is missing the {column!r} column")
    value = str(record[column]).strip()
    if not value:
        raise ValueError(f"{kind} record has an empty {column!r} value")
    return value


def _optional_string(record: Mapping[str, Any], column: str) -> str:
    return str(record.get(column) or "").strip()


def _check_no_duplicates(ids: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for doc_id in ids:
        if doc_id in seen:
            raise ValueError(f"duplicate {kind} id: {doc_id!r}")
        seen.add(doc_id)


def _check_missing_ids(
    ids: Iterable[str],
    allowed: Set[str],
    id_kind: str,
    target_kind: str,
) -> None:
    missing = sorted(set(ids) - allowed)
    if missing:
        preview = ", ".join(repr(m) for m in missing[:3])
        raise ValueError(
            f"qrels reference {len(missing)} {id_kind} id(s) missing from "
            f"{target_kind}: {preview}"
        )


# --- normalization ---


def normalize_corpus(
    records: Iterable[Mapping[str, Any]],
    id_column: str = CORPUS_ID_COLUMN,
    title_column: str = CORPUS_TITLE_COLUMN,
    text_column: str = CORPUS_TEXT_COLUMN,
) -> list[dict[str, str]]:
    """Normalize raw corpus records into ``[{"id", "text"}]``.

    The id is preserved exactly as a string. Title and body are joined
    with a single space; a missing or empty title contributes nothing.
    Duplicate ids raise ``ValueError``.
    """
    corpus: list[dict[str, str]] = []
    ids: list[str] = []

    for record in records:
        doc_id = _string_field(record, id_column, "corpus")
        title = _optional_string(record, title_column)
        body = _string_field(record, text_column, "corpus")

        parts = [part for part in (title, body) if part]
        text = " ".join(parts).strip()
        if not text:
            raise ValueError(f"corpus record {doc_id!r} has no text content")

        ids.append(doc_id)
        corpus.append({"id": doc_id, "text": text})

    _check_no_duplicates(ids, "corpus")
    return corpus


def normalize_queries(
    records: Iterable[Mapping[str, Any]],
    id_column: str = QUERY_ID_COLUMN,
    text_column: str = QUERY_TEXT_COLUMN,
) -> dict[str, str]:
    """Normalize raw query records into ``{query_id: query_text}``.

    Query ids are preserved exactly as strings. Duplicate ids raise
    ``ValueError``.
    """
    queries: dict[str, str] = {}

    for record in records:
        query_id = _string_field(record, id_column, "query")
        text = _string_field(record, text_column, "query")

        if query_id in queries:
            raise ValueError(f"duplicate query id: {query_id!r}")
        queries[query_id] = text

    return queries


def normalize_qrels(
    records: Iterable[Mapping[str, Any]],
    corpus_ids: Set[str] | None = None,
    query_ids: Set[str] | None = None,
    query_column: str = QREL_QUERY_COLUMN,
    doc_column: str = QREL_DOC_COLUMN,
    score_column: str = QREL_SCORE_COLUMN,
) -> dict[str, dict[str, int]]:
    """Normalize raw qrel records into ``{query_id: {doc_id: grade}}``.

    Ids are preserved exactly as strings and relevance grades are kept as
    integers (``> 0`` means relevant, graded values feed nDCG). When
    ``corpus_ids``/``query_ids`` are provided, qrel entries referencing
    unknown documents or queries raise ``ValueError`` so misaligned ids
    never pass silently.
    """
    qrels: dict[str, dict[str, int]] = {}

    for record in records:
        query_id = _string_field(record, query_column, "qrel")
        doc_id = _string_field(record, doc_column, "qrel")
        if score_column not in record:
            raise ValueError(f"qrel record is missing the {score_column!r} column")
        grade = int(record[score_column])

        qrels.setdefault(query_id, {})[doc_id] = grade

    if corpus_ids is not None:
        _check_missing_ids(
            (doc_id for entries in qrels.values() for doc_id in entries),
            corpus_ids,
            "document",
            "the corpus",
        )
    if query_ids is not None:
        _check_missing_ids(qrels, query_ids, "query", "the query set")

    return qrels


# --- loading ---


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
