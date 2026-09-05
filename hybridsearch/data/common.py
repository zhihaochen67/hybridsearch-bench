"""Dataset-agnostic normalization and container shared by dataset loaders.

Holds the small generic pieces used by both the SciFact and the FiQA
loaders: the normalized ``Dataset`` container plus strict normalization
helpers for BEIR-style raw records. The column names are the BEIR
convention (corpus/queries: ``_id`` / ``title`` / ``text``; qrels:
``query-id`` / ``corpus-id`` / ``score``), which both datasets follow.

Normalization is deliberately strict: duplicate ids, missing required
columns, empty required values, and qrel entries referencing unknown
documents or queries all raise ``ValueError`` instead of being silently
dropped. The single opt-out is ``normalize_corpus(require_text=False)``,
used by the FiQA loader, which keeps the handful of genuinely empty-text
corpus records that exist in the published dataset rather than dropping
or failing on them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from typing import Any

# BEIR-style column names shared by SciFact and FiQA.
CORPUS_ID_COLUMN = "_id"
CORPUS_TITLE_COLUMN = "title"
CORPUS_TEXT_COLUMN = "text"
QUERY_ID_COLUMN = "_id"
QUERY_TEXT_COLUMN = "text"
QREL_QUERY_COLUMN = "query-id"
QREL_DOC_COLUMN = "corpus-id"
QREL_SCORE_COLUMN = "score"


@dataclass
class Dataset:
    """Normalized dataset: corpus list, query map, and qrel map.

    Subclass or instantiate with a ``name`` and ``source`` to describe a
    concrete dataset (see ``SciFactDataset`` and ``FiQADataset``).
    """

    corpus: list[dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str = "dataset"
    source: str = "unknown"

    def corpus_ids(self) -> set[str]:
        """Document ids of the corpus."""
        return {document["id"] for document in self.corpus}

    def query_ids(self) -> set[str]:
        """Ids of all loaded queries."""
        return set(self.queries)


# --- field helpers ---


def _string_field(
    record: Mapping[str, Any], column: str, kind: str, allow_empty: bool = False
) -> str:
    if column not in record:
        raise ValueError(f"{kind} record is missing the {column!r} column")
    value = str(record[column]).strip()
    if not value and not allow_empty:
        raise ValueError(f"{kind} record has an empty {column!r} value")
    return value


def _optional_string(record: Mapping[str, Any], column: str) -> str:
    return str(record.get(column) or "").strip()


def _check_no_duplicates(ids: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f"duplicate {kind} id: {item_id!r}")
        seen.add(item_id)


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
    require_text: bool = True,
) -> list[dict[str, str]]:
    """Normalize raw corpus records into ``[{"id", "text"}]``.

    Ids are preserved exactly as strings. Title and body are joined with
    a single space; a missing or empty title contributes nothing.
    Duplicate ids raise ``ValueError``.

    With ``require_text=True`` (the default) a record whose text column
    is missing or blank raises ``ValueError``. With
    ``require_text=False`` such records are kept with ``"text": ""``;
    this opt-out exists for FiQA, whose published corpus genuinely
    contains a few empty posts that must not be dropped silently.
    """
    corpus: list[dict[str, str]] = []
    ids: list[str] = []

    for record in records:
        doc_id = _string_field(record, id_column, "corpus")
        title = _optional_string(record, title_column)
        body = _string_field(
            record, text_column, "corpus", allow_empty=not require_text
        )

        parts = [part for part in (title, body) if part]
        text = " ".join(parts).strip()
        if not text and require_text:
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
    ``corpus_ids`` / ``query_ids`` are provided, qrel entries referencing
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