"""Systematic retrieval failure analysis on SciFact.

    python experiments/failure_analysis.py --examples-per-category 3

Runs the four analyzed methods (BM25, Dense, Hybrid Weighted alpha=0.5,
Hybrid Weighted + rerank@20) over every SciFact query that has qrels,
collects per-query metrics and diagnostics, and selects deterministic,
metric-driven examples for six categories:

- BM25 wins, Dense loses          (bm25_ndcg - dense_ndcg > 0)
- Dense wins, BM25 loses          (dense_ndcg - bm25_ndcg > 0)
- Hybrid wins                     (hybrid_ndcg > max(bm25, dense))
- Reranker improves               (reranked_ndcg - hybrid_ndcg > 0)
- Reranker hurts                  (hybrid_ndcg - reranked_ndcg > 0)
- All methods fail                (Recall@10 == 0 for all four methods)

nDCG@10 is the primary comparison metric. Examples are ranked by the
category score (ties broken by ascending query id) and limited to
``--examples-per-category``. Categories with no qualifying query are
reported as empty — nothing is fabricated.

The full JSON goes to ``outputs/scifact_failure_analysis.json`` and a
machine-generated Markdown report to ``analysis/failure_analysis.md``.
Smoke runs (``--max-queries N``) write ``outputs/scifact_failure_analysis_smokeN.json``
and ``outputs/failure_analysis_smokeN.md`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Let the documented CLI `python experiments/failure_analysis.py`
    # resolve the `hybridsearch` package without installing the project.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark import save_results, select_query_ids
from hybridsearch.data.scifact import SciFactDataset, load_scifact
from hybridsearch.evaluation.metrics import evaluate_query, mean_metrics
from hybridsearch.ranking.reranker import CrossEncoderReranker
from hybridsearch.retrieval.bm25 import BM25
from hybridsearch.retrieval.dense import DenseRetriever
from hybridsearch.retrieval.hybrid import HybridRetriever
from hybridsearch.retrieval.simple import tokenize

METHOD_NAMES = ("BM25", "Dense", "Hybrid Weighted", "Hybrid + rerank@20")

CATEGORIES = (
    "bm25_wins",
    "dense_wins",
    "hybrid_wins",
    "reranker_improves",
    "reranker_hurts",
    "all_fail",
)

CATEGORY_DEFINITIONS = {
    "bm25_wins": "BM25 nDCG@10 minus Dense nDCG@10 > 0; ranked by largest difference",
    "dense_wins": "Dense nDCG@10 minus BM25 nDCG@10 > 0; ranked by largest difference",
    "hybrid_wins": "Hybrid Weighted nDCG@10 > max(BM25, Dense) nDCG@10; ranked by largest margin",
    "reranker_improves": "Reranked nDCG@10 minus Hybrid Weighted nDCG@10 > 0; ranked by largest gain",
    "reranker_hurts": "Hybrid Weighted nDCG@10 minus Reranked nDCG@10 > 0; ranked by largest loss",
    "all_fail": "Recall@10 == 0 for BM25, Dense, Hybrid Weighted, and Hybrid + rerank@20; ordered by ascending query id",
}

CATEGORY_TITLES = {
    "bm25_wins": "BM25 Wins, Dense Loses",
    "dense_wins": "Dense Wins, BM25 Loses",
    "hybrid_wins": "Hybrid Wins",
    "reranker_improves": "Reranker Improves",
    "reranker_hurts": "Reranker Hurts",
    "all_fail": "All Methods Fail",
}


# --- per-query diagnostics ---


def snippet(text: str, limit: int = 200) -> str:
    """Whitespace-normalized, length-limited text excerpt."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def compute_lexical_overlap(query_text: str, doc_text: str):
    """(count, ratio) of unique query tokens that appear verbatim in the doc."""
    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        return 0, 0.0
    doc_tokens = set(tokenize(doc_text))
    count = len(query_tokens & doc_tokens)
    return count, count / len(query_tokens)


def first_relevant_rank(ranked_ids, qrels):
    """1-based rank of the first relevant document, or None."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if qrels.get(doc_id, 0) > 0:
            return rank
    return None


def relevant_in_top(ranked_ids, qrels) -> list:
    """Relevant document ids appearing in the ranked list, in rank order."""
    return [doc_id for doc_id in ranked_ids if qrels.get(doc_id, 0) > 0]


def ranks_of_relevant(ranked_ids, qrels) -> dict:
    """{relevant_doc_id: 1-based rank in the list, or None}."""
    position = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
    return {
        doc_id: position.get(doc_id)
        for doc_id, grade in qrels.items()
        if grade > 0
    }


def build_query_record(query_id, query_text, qrels, corpus_texts, method_outputs, k):
    """Collect metrics and diagnostics for one query.

    ``method_outputs`` maps every analyzed method name to
    ``{"ranked_ids": [...]}`` and must contain an extra ``"reranker"``
    entry with ``hybrid_top20_ids`` and ``reranked_top10`` (id + score
    dicts) for the reranker diagnostics.
    """
    relevant = []
    for doc_id, grade in sorted(qrels.items()):
        if grade <= 0:
            continue
        doc_text = corpus_texts.get(doc_id, "")
        overlap_count, overlap_ratio = compute_lexical_overlap(query_text, doc_text)
        relevant.append(
            {
                "id": doc_id,
                "grade": grade,
                "snippet": snippet(doc_text),
                "overlap_count": overlap_count,
                "overlap_ratio": overlap_ratio,
            }
        )

    record = {
        "query_id": query_id,
        "query": query_text,
        "relevant": relevant,
        "methods": {},
        "reranker": {},
    }

    for name in METHOD_NAMES:
        ranked_ids = method_outputs[name]["ranked_ids"]
        record["methods"][name] = {
            "ranked_ids": ranked_ids,
            "metrics": evaluate_query(ranked_ids, qrels, k=k),
            "first_relevant_rank": first_relevant_rank(ranked_ids, qrels),
            "relevant_in_top10": relevant_in_top(ranked_ids, qrels),
        }

    reranker_output = method_outputs["reranker"]
    record["reranker"] = {
        "hybrid_top20_ids": reranker_output["hybrid_top20_ids"],
        "reranked_top10": reranker_output["reranked_top10"],
        "relevant_rank_before": ranks_of_relevant(
            reranker_output["hybrid_top20_ids"], qrels
        ),
        "relevant_rank_after": ranks_of_relevant(
            [item["id"] for item in reranker_output["reranked_top10"]], qrels
        ),
    }

    return record


# --- method runner ---


def make_query_runner(sparse, dense, hybrid, reranker, k: int, reranker_pool: int):
    """Return callable(query_text) -> per-method outputs for one query."""

    def run(query_text):
        candidates = hybrid.search(query_text, top_k=reranker_pool)
        reranked = reranker.rerank(query_text, candidates, top_k=k)
        return {
            "BM25": {
                "ranked_ids": [r["id"] for r in sparse.search(query_text, top_k=k)]
            },
            "Dense": {
                "ranked_ids": [r["id"] for r in dense.search(query_text, top_k=k)]
            },
            "Hybrid Weighted": {
                "ranked_ids": [r["id"] for r in hybrid.search(query_text, top_k=k)]
            },
            "Hybrid + rerank@20": {
                "ranked_ids": [item["id"] for item in reranked]
            },
            "reranker": {
                "hybrid_top20_ids": [r["id"] for r in candidates],
                "reranked_top10": [
                    {"id": item["id"], "reranker_score": item["reranker_score"]}
                    for item in reranked
                ],
            },
        }

    return run


# --- category selection ---


def category_scores(record, k) -> dict:
    """Selection score of one query for each difference category."""
    metrics = record["methods"]
    bm25 = metrics["BM25"]["metrics"][f"ndcg@{k}"]
    dense = metrics["Dense"]["metrics"][f"ndcg@{k}"]
    hybrid = metrics["Hybrid Weighted"]["metrics"][f"ndcg@{k}"]
    reranked = metrics["Hybrid + rerank@20"]["metrics"][f"ndcg@{k}"]
    return {
        "bm25_wins": bm25 - dense,
        "dense_wins": dense - bm25,
        "hybrid_wins": hybrid - max(bm25, dense),
        "reranker_improves": reranked - hybrid,
        "reranker_hurts": hybrid - reranked,
    }


def is_all_fail(record, k) -> bool:
    """True when every analyzed method has Recall@k == 0."""
    return all(
        record["methods"][name]["metrics"][f"recall@{k}"] == 0.0
        for name in METHOD_NAMES
    )


def serialize_case(record, category, score, k) -> dict:
    """JSON-ready representation of one selected query."""
    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "category": category,
        "selection_score": score,
        "relevant": record["relevant"],
        "methods": {
            name: {
                "ranked_ids": entry["ranked_ids"],
                "metrics": entry["metrics"],
                "first_relevant_rank": entry["first_relevant_rank"],
                "relevant_in_top10": entry["relevant_in_top10"],
            }
            for name, entry in record["methods"].items()
        },
        "reranker": record["reranker"],
    }


def select_categories(records, k, examples_per_category) -> tuple[dict, dict]:
    """Select examples for every category and count qualifiers."""
    counts: dict[str, int] = {}
    selected: dict[str, list] = {}

    for category in ("bm25_wins", "dense_wins", "hybrid_wins",
                     "reranker_improves", "reranker_hurts"):
        qualifiers = []
        for record in records:
            score = category_scores(record, k)[category]
            if score > 0:
                qualifiers.append((record, score))
        counts[category] = len(qualifiers)

        qualifiers.sort(key=lambda item: (-item[1], item[0]["query_id"]))
        selected[category] = [
            serialize_case(record, category, score, k)
            for record, score in qualifiers[:examples_per_category]
        ]

    failures = [record for record in records if is_all_fail(record, k)]
    counts["all_fail"] = len(failures)
    failures.sort(key=lambda record: record["query_id"])
    selected["all_fail"] = [
        serialize_case(record, "all_fail", None, k)
        for record in failures[:examples_per_category]
    ]

    return selected, counts


# --- experiment entry point ---


def run_failure_analysis(
    dataset: SciFactDataset,
    k: int = 10,
    alpha: float = 0.5,
    hybrid_candidate_k: int = 50,
    reranker_pool: int = 20,
    dense_model: str = DenseRetriever.DEFAULT_MODEL_NAME,
    reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL_NAME,
    max_queries: int | None = None,
    examples_per_category: int = 3,
    query_runner=None,
) -> dict:
    """Run the failure analysis and return the serializable payload.

    ``query_runner`` is an optional callable ``query_text -> method_outputs``
    replacing the real models for offline tests.
    """
    query_ids = select_query_ids(dataset.queries, dataset.qrels, max_queries=max_queries)
    corpus_texts = {document["id"]: document["text"] for document in dataset.corpus}

    if query_runner is None:
        sparse = BM25(dataset.corpus)
        dense = DenseRetriever(dataset.corpus, model_name=dense_model)
        hybrid = HybridRetriever(
            sparse, dense, method="weighted", alpha=alpha, candidate_k=hybrid_candidate_k
        )
        reranker = CrossEncoderReranker(model_name=reranker_model)
        query_runner = make_query_runner(sparse, dense, hybrid, reranker, k, reranker_pool)

    records = []
    for query_id in query_ids:
        query_text = dataset.queries[query_id]
        outputs = query_runner(query_text)
        outputs = _fill_candidate_ids(outputs)
        records.append(
            build_query_record(
                query_id, query_text, dataset.qrels[query_id], corpus_texts, outputs, k
            )
        )

    selected, counts = select_categories(records, k, examples_per_category)

    mean_by_method = {
        name: mean_metrics([record["methods"][name]["metrics"] for record in records])
        for name in METHOD_NAMES
    }

    return {
        "metadata": {
            "dataset": dataset.name,
            "source": dataset.source,
            "k": k,
            "alpha": alpha,
            "hybrid_candidate_k": hybrid_candidate_k,
            "reranker_pool": reranker_pool,
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "num_queries": len(query_ids),
            "max_queries": max_queries,
            "examples_per_category": examples_per_category,
            "methods": list(METHOD_NAMES),
            "category_definitions": CATEGORY_DEFINITIONS,
        },
        "summary": {
            "category_counts": counts,
            "mean_metrics": mean_by_method,
        },
        "categories": selected,
    }


def _fill_candidate_ids(outputs: dict) -> dict:
    """Populate missing ``hybrid_top20_ids`` for offline query runners.

    Real runners derive it from the reranker input; fakes may omit it, in
    which case the reranked top-10 ids are used as the candidate list.
    """
    reranker = outputs["reranker"]
    if not reranker.get("hybrid_top20_ids"):
        reranker["hybrid_top20_ids"] = [
            item["id"] for item in reranker["reranked_top10"]
        ]
    return outputs


def output_path_for(max_queries: int | None) -> str:
    if max_queries is not None:
        return f"outputs/scifact_failure_analysis_smoke{max_queries}.json"
    return "outputs/scifact_failure_analysis.json"


def report_path_for(max_queries: int | None) -> str:
    if max_queries is not None:
        return f"outputs/failure_analysis_smoke{max_queries}.md"
    return "analysis/failure_analysis.md"


# --- Markdown report generation ---


def _fmt_metrics(metrics: dict) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _case_observation(case, category, k) -> list[str]:
    methods = case["methods"]
    ndcg = {name: methods[name]["metrics"][f"ndcg@{k}"] for name in METHOD_NAMES}

    if category == "bm25_wins":
        return [
            f"BM25 nDCG@{k} is {ndcg['BM25']:.4f} vs Dense {ndcg['Dense']:.4f}.",
            _rank_observation(case, "BM25", "Dense"),
        ]
    if category == "dense_wins":
        return [
            f"Dense nDCG@{k} is {ndcg['Dense']:.4f} vs BM25 {ndcg['BM25']:.4f}.",
            _rank_observation(case, "Dense", "BM25"),
        ]
    if category == "hybrid_wins":
        return [
            f"Hybrid Weighted nDCG@{k} is {ndcg['Hybrid Weighted']:.4f}, "
            f"beating BM25 ({ndcg['BM25']:.4f}) and Dense ({ndcg['Dense']:.4f})."
        ]
    if category in ("reranker_improves", "reranker_hurts"):
        direction = "improves" if category == "reranker_improves" else "hurts"
        lines = [
            f"The reranker {direction} nDCG@{k}: Hybrid {ndcg['Hybrid Weighted']:.4f} "
            f"vs reranked {ndcg['Hybrid + rerank@20']:.4f}."
        ]
        for rel in case["relevant"]:
            before = case["reranker"]["relevant_rank_before"].get(rel["id"])
            after = case["reranker"]["relevant_rank_after"].get(rel["id"])
            lines.append(
                f"Relevant document {rel['id']} moves from hybrid-top-20 rank "
                f"{before} to final rank {after}."
            )
        return lines
    # all_fail
    return ["Recall@10 is 0 for BM25, Dense, Hybrid Weighted, and the reranked pipeline."]


def _rank_observation(case, winner: str, loser: str) -> str:
    winner_rank = case["methods"][winner]["first_relevant_rank"]
    loser_rank = case["methods"][loser]["first_relevant_rank"]
    winner_text = f"#{winner_rank}" if winner_rank else "not in top-10"
    loser_text = f"#{loser_rank}" if loser_rank else "not in top-10"
    return (
        f"{winner} ranks the first relevant document {winner_text} "
        f"while {loser} ranks it {loser_text}."
    )


def _case_hypothesis(case, category) -> list[str]:
    hypotheses = []
    if category in ("bm25_wins", "dense_wins") and case["relevant"]:
        rel = case["relevant"][0]
        hypotheses.append(
            f"The query and relevant document {rel['id']} share "
            f"{rel['overlap_count']} exact query token(s) "
            f"(ratio {rel['overlap_ratio']:.2f}); overlap may favor lexical "
            f"retrieval, but overlap alone does not prove causation."
        )
    if category == "hybrid_wins" and case["relevant"]:
        rel = case["relevant"][0]
        rank_hybrid = case["methods"]["Hybrid Weighted"]["first_relevant_rank"]
        rank_bm25 = case["methods"]["BM25"]["first_relevant_rank"]
        rank_dense = case["methods"]["Dense"]["first_relevant_rank"]
        hypotheses.append(
            f"Fusing the rankings moves relevant document {rel['id']} to "
            f"rank {rank_hybrid} (BM25: {rank_bm25}, Dense: {rank_dense}); "
            f"combining complementary evidence can promote documents "
            f"neither retriever ranks first. Hypothesis only."
        )
    if category == "reranker_improves":
        hypotheses.append(
            "The cross-encoder's pairwise scores disagree with the hybrid "
            "fusion ranking in this case, which may explain the reordering; "
            "hypothesis only."
        )
    if category == "reranker_hurts":
        hypotheses.append(
            "The cross-encoder's pairwise scores may promote a spuriously "
            "scored candidate above the relevant one; hypothesis only."
        )
    if category == "all_fail":
        in_pool = [
            rel["id"]
            for rel in case["relevant"]
            if case["reranker"]["relevant_rank_before"].get(rel["id"]) is not None
        ]
        pool_note = (
            f" Relevant document(s) {in_pool} are present in the reranker "
            f"candidate pool but still miss the top-10."
            if in_pool
            else " No relevant document reaches the reranker candidate pool."
        )
        hypotheses.append(
            "None of the four methods retrieves a relevant document in the "
            "top 10 for this query." + pool_note
        )
    return hypotheses


def render_report(payload: dict) -> str:
    """Generate the Markdown report from the actual payload."""
    metadata = payload["metadata"]
    k = metadata["k"]
    lines = [
        "# SciFact Failure Analysis",
        "",
        "## Methodology",
        "",
        f"- dataset: {metadata['dataset']} ({metadata['source']})",
        f"- queries evaluated: {metadata['num_queries']} (all with qrels)",
        f"- methods: {', '.join(metadata['methods'])}",
        f"- evaluation k = {k}; hybrid alpha = {metadata['alpha']}; "
        f"hybrid candidate_k = {metadata['hybrid_candidate_k']}; "
        f"reranker pool = {metadata['reranker_pool']}",
        f"- dense model: {metadata['dense_model']}",
        f"- reranker model: {metadata['reranker_model']}",
        "- primary comparison metric: nDCG@10 (MRR@10 / Recall@10 as diagnostics)",
        "- category definitions:",
    ]
    for category, definition in metadata["category_definitions"].items():
        lines.append(f"  - {category}: {definition}")
    lines += [
        f"- selection: up to {metadata['examples_per_category']} examples per "
        f"category, ranked by category score with ascending query id as the "
        f"deterministic tie-break",
        "",
        "## Summary",
        "",
        "Category counts (qualifying queries):",
        "",
        "| Category | Qualifying queries |",
        "|---|---|",
    ]
    for category in CATEGORIES:
        lines.append(
            f"| {CATEGORY_TITLES[category]} | "
            f"{payload['summary']['category_counts'][category]} |"
        )
    lines += [
        "",
        "Mean metrics per method:",
        "",
        "| Method | " + " | ".join(
            f"{key}" for key in next(iter(payload["summary"]["mean_metrics"].values()))
        ) + " |",
        "|---|---|---:|---:|---:|",
    ]
    for name, metrics in payload["summary"]["mean_metrics"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{value:.4f}" for value in metrics.values()
        ) + " |")

    for category in CATEGORIES:
        cases = payload["categories"][category]
        lines += [
            "",
            f"## {CATEGORY_TITLES[category]}",
            "",
            f"Qualifying queries: {payload['summary']['category_counts'][category]}; "
            f"shown: {len(cases)}.",
        ]
        if not cases:
            lines.append("")
            lines.append("_No qualifying query exists._")
            continue
        for case in cases:
            lines += _render_case(case, category, k)

    return "\n".join(lines) + "\n"


def _render_case(case, category, k) -> list[str]:
    lines = [
        "",
        f"### Query {case['query_id']}: {case['query']}",
        "",
        "Relevant document(s):",
    ]
    for rel in case["relevant"]:
        lines.append(
            f"- {rel['id']} (grade {rel['grade']}): {rel['snippet']} — exact "
            f"query-token overlap {rel['overlap_count']} "
            f"(ratio {rel['overlap_ratio']:.2f})"
        )
    lines.append("")
    lines.append("Per-method top-10:")
    for name in METHOD_NAMES:
        entry = case["methods"][name]
        ranked = ", ".join(entry["ranked_ids"])
        lines.append(
            f"- {name}: [{ranked}] — {_fmt_metrics(entry['metrics'])} — "
            f"first relevant rank {entry['first_relevant_rank']} — "
            f"relevant in top-10: {entry['relevant_in_top10']}"
        )
    lines += [
        "",
        "Observation:",
    ]
    for observation in _case_observation(case, category, k):
        lines.append(f"  {observation}")
    lines += [
        "",
        "Hypothesis:",
    ]
    for hypothesis in _case_hypothesis(case, category):
        lines.append(f"  {hypothesis}")
    return lines


# --- CLI ---


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=["scifact"])
    parser.add_argument("--k", type=int, default=10, help="evaluation cutoff")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="sparse weight for weighted fusion",
    )
    parser.add_argument(
        "--hybrid-candidate-k",
        type=int,
        default=50,
        help="Hybrid Weighted internal candidate pool",
    )
    parser.add_argument(
        "--reranker-pool",
        type=int,
        default=20,
        help="reranker candidate pool size",
    )
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL_NAME,
        help="sentence-transformers model for dense retrieval",
    )
    parser.add_argument(
        "--reranker-model",
        default=CrossEncoderReranker.DEFAULT_MODEL_NAME,
        help="cross-encoder model for reranking",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="analyze only the first N queries (smoke tests)",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="number of examples selected per category",
    )
    parser.add_argument("--output", default=None, help="override the JSON output path")
    parser.add_argument("--report", default=None, help="override the Markdown report path")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Loading {args.dataset} and building indices/models once (untimed) ...")
    dataset = load_scifact()
    payload = run_failure_analysis(
        dataset,
        k=args.k,
        alpha=args.alpha,
        hybrid_candidate_k=args.hybrid_candidate_k,
        reranker_pool=args.reranker_pool,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        max_queries=args.max_queries,
        examples_per_category=args.examples_per_category,
    )

    print(f"analyzed {payload['metadata']['num_queries']} queries")
    print("category counts:")
    for category in CATEGORIES:
        print(f"  {category}: {payload['summary']['category_counts'][category]}")

    output = args.output or output_path_for(args.max_queries)
    save_results(payload, output)
    print(f"Saved results to {output}")

    report_path = Path(args.report or report_path_for(args.max_queries))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
