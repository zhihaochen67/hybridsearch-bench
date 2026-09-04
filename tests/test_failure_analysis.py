"""Offline tests for the failure-analysis harness (Phase 11).

No SciFact download and no real models: a synthetic ``SciFactDataset`` and
a fake per-query method runner drive every test deterministically.
"""

import json

import pytest

from experiments.failure_analysis import (
    CATEGORIES,
    build_query_record,
    category_scores,
    compute_lexical_overlap,
    first_relevant_rank,
    is_all_fail,
    output_path_for,
    relevant_in_top,
    render_report,
    report_path_for,
    run_failure_analysis,
    select_categories,
    serialize_case,
)
from hybridsearch.data.scifact import SciFactDataset
from hybridsearch.evaluation.metrics import evaluate_query

K = 2

CORPUS = [
    {"id": "R1", "text": "bm25 wins here extra terms"},
    {"id": "R2", "text": "dense wins here extra terms"},
    {"id": "R3", "text": "hybrid wins here extra terms"},
    {"id": "R4", "text": "improve rerank here extra terms"},
    {"id": "R5", "text": "hurt rerank here extra terms"},
    {"id": "R6", "text": "all fail here extra terms"},
    {"id": "X1", "text": "distractor one"},
    {"id": "X2", "text": "distractor two"},
    {"id": "X3", "text": "distractor three"},
]

TEXTS = {document["id"]: document["text"] for document in CORPUS}

DATASET = SciFactDataset(
    corpus=CORPUS,
    queries={
        "q_bm25": "bm25 wins here",
        "q_dense": "dense wins here",
        "q_hybrid": "hybrid wins here",
        "q_improve": "improve rerank here",
        "q_hurt": "hurt rerank here",
        "q_fail": "all fail here",
        "q_norel": "no qrels here",  # excluded from analysis
    },
    qrels={
        "q_bm25": {"R1": 1},
        "q_dense": {"R2": 1},
        "q_hybrid": {"R3": 1},
        "q_improve": {"R4": 1},
        "q_hurt": {"R5": 1},
        "q_fail": {"R6": 1},
    },
)


def make_outputs(bm25_ids, dense_ids, hybrid_ids, reranked_ids, candidate_ids=None):
    """Build a fake method-outputs dict shaped like the real runner's."""
    return {
        "BM25": {"ranked_ids": bm25_ids},
        "Dense": {"ranked_ids": dense_ids},
        "Hybrid Weighted": {"ranked_ids": hybrid_ids},
        "Hybrid + rerank@20": {"ranked_ids": reranked_ids},
        "reranker": {
            "hybrid_top20_ids": list(candidate_ids or hybrid_ids),
            "reranked_top10": [
                {"id": doc_id, "reranker_score": 10.0 - index}
                for index, doc_id in enumerate(reranked_ids)
            ],
        },
    }


def make_record(query_id, qrels, bm25_ids, dense_ids, hybrid_ids=None,
                reranked_ids=None, candidate_ids=None, query_text=None):
    outputs = make_outputs(
        bm25_ids, dense_ids, hybrid_ids or bm25_ids,
        reranked_ids if reranked_ids is not None else (hybrid_ids or bm25_ids),
        candidate_ids,
    )
    return build_query_record(
        query_id, query_text or f"query {query_id}", qrels, TEXTS, outputs, K
    )


def run_full():
    def runner(query_text):
        return {
            "bm25 wins here": make_outputs(
                ["R1", "X1"], ["X1", "X2"], ["R1", "X1"], ["R1", "X1"],
                ["R1", "X1", "X2"],
            ),
            "dense wins here": make_outputs(
                ["X1", "X2"], ["R2", "X1"], ["R2", "X1"], ["R2", "X1"],
                ["R2", "X1", "X2"],
            ),
            "hybrid wins here": make_outputs(
                ["X1", "R3"], ["X2", "R3"], ["R3", "X1"], ["R3", "X1"],
                ["R3", "X1", "X2"],
            ),
            "improve rerank here": make_outputs(
                ["X1", "X2"], ["X1", "X2"], ["X1", "R4"], ["R4", "X1"],
                ["X1", "R4"],
            ),
            "hurt rerank here": make_outputs(
                ["X1", "X2"], ["X1", "X2"], ["R5", "X1"], ["X1", "R5"],
                ["R5", "X1"],
            ),
            "all fail here": make_outputs(
                ["X1", "X2"], ["X2", "X1"], ["X1", "X2"], ["X2", "X1"],
                ["X1", "X2"],
            ),
        }[query_text]

    return run_failure_analysis(
        DATASET, k=K, alpha=0.5, hybrid_candidate_k=3, reranker_pool=2,
        dense_model="fake/dense", reranker_model="fake/reranker",
        examples_per_category=2, query_runner=runner,
    )


# --- category selection ---


def test_bm25_wins_selection_ranking():
    records = [
        make_record("qA", {"R1": 1}, ["R1", "X1"], ["X1", "X2"]),  # +1.0
        make_record("qB", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),  # +0.6309
        make_record("qC", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),  # +0.6309 tie
    ]
    selected, counts = select_categories(records, K, examples_per_category=3)

    assert counts["bm25_wins"] == 3
    assert [case["query_id"] for case in selected["bm25_wins"]] == ["qA", "qB", "qC"]
    assert selected["bm25_wins"][0]["selection_score"] == pytest.approx(1.0)
    # Rank-2 hit: DCG = 1/log2(3).
    assert selected["bm25_wins"][1]["selection_score"] == pytest.approx(
        1 / 1.584962500721156
    )


def test_bm25_wins_score_is_ndcg_difference():
    record = make_record("qA", {"R1": 1}, ["X1", "R1"], ["X1", "X2"])

    score = category_scores(record, K)["bm25_wins"]
    expected = (
        evaluate_query(["X1", "R1"], {"R1": 1}, k=K)["ndcg@2"]
        - evaluate_query(["X1", "X2"], {"R1": 1}, k=K)["ndcg@2"]
    )

    assert score == pytest.approx(expected)
    assert score > 0


def test_dense_wins_selection_ranking():
    records = [
        make_record("qD", {"R2": 1}, ["X1", "X2"], ["R2", "X1"]),  # +1.0
        make_record("qE", {"R2": 1}, ["X1", "X2"], ["X1", "R2"]),  # +0.631
    ]
    selected, counts = select_categories(records, K, examples_per_category=3)

    assert counts["dense_wins"] == 2
    assert [case["query_id"] for case in selected["dense_wins"]] == ["qD", "qE"]


def test_hybrid_wins_selection_is_strict():
    wins = make_record("qH", {"R3": 1}, ["X1", "R3"], ["X2", "R3"],
                       ["R3", "X1"])
    ties = make_record("qT", {"R3": 1}, ["R3", "X1"], ["X1", "R3"],
                       ["R3", "X1"])  # hybrid == max(bm25) -> not a win
    loses = make_record("qL", {"R3": 1}, ["R3", "X1"], ["X2", "R3"],
                        ["X1", "R3"])

    selected, counts = select_categories([wins, ties, loses], K, examples_per_category=3)

    assert counts["hybrid_wins"] == 1
    assert [case["query_id"] for case in selected["hybrid_wins"]] == ["qH"]
    assert selected["hybrid_wins"][0]["selection_score"] == pytest.approx(
        1.0 - 1 / 1.584962500721156
    )


def test_reranker_improves_selection():
    improved = make_record("qI", {"R4": 1}, ["X1", "X2"], ["X1", "X2"],
                           ["X1", "R4"], ["R4", "X1"], candidate_ids=["X1", "R4"])
    unchanged = make_record("qU", {"R4": 1}, ["X1", "X2"], ["X1", "X2"],
                            ["R4", "X1"], ["R4", "X1"], candidate_ids=["R4", "X1"])

    selected, counts = select_categories([improved, unchanged], K, examples_per_category=3)

    assert counts["reranker_improves"] == 1
    assert [case["query_id"] for case in selected["reranker_improves"]] == ["qI"]
    assert selected["reranker_improves"][0]["selection_score"] == pytest.approx(
        1.0 - 1 / 1.584962500721156
    )


def test_reranker_hurts_selection():
    hurt = make_record("qH2", {"R5": 1}, ["X1", "X2"], ["X1", "X2"],
                       ["R5", "X1"], ["X1", "R5"], candidate_ids=["R5", "X1"])
    unchanged = make_record("qU2", {"R5": 1}, ["X1", "X2"], ["X1", "X2"],
                            ["R5", "X1"], ["R5", "X1"], candidate_ids=["R5", "X1"])

    selected, counts = select_categories([hurt, unchanged], K, examples_per_category=3)

    assert counts["reranker_hurts"] == 1
    assert [case["query_id"] for case in selected["reranker_hurts"]] == ["qH2"]


def test_all_fail_definition_requires_zero_recall_for_all_methods():
    fails = make_record("qF", {"R6": 1}, ["X1", "X2"], ["X2", "X1"],
                        ["X1", "X2"], ["X2", "X1"])
    almost = make_record("qA2", {"R6": 1}, ["R6", "X1"], ["X2", "X1"],
                         ["X1", "X2"], ["X2", "X1"])  # BM25 finds it

    assert is_all_fail(fails, K) is True
    assert is_all_fail(almost, K) is False

    selected, counts = select_categories([fails, almost], K, examples_per_category=3)
    assert counts["all_fail"] == 1
    assert [case["query_id"] for case in selected["all_fail"]] == ["qF"]
    assert selected["all_fail"][0]["selection_score"] is None


def test_no_qualifying_examples_reported_as_empty():
    records = [
        make_record("qA", {"R1": 1}, ["R1", "X1"], ["X1", "X2"]),  # bm25 wins
    ]
    selected, counts = select_categories(records, K, examples_per_category=3)

    assert counts["dense_wins"] == 0
    assert selected["dense_wins"] == []
    assert counts["hybrid_wins"] == 0
    assert selected["hybrid_wins"] == []
    assert counts["reranker_hurts"] == 0
    assert selected["reranker_hurts"] == []


def test_examples_per_category_limiting():
    records = [
        make_record("q1", {"R1": 1}, ["R1", "X1"], ["X1", "X2"]),  # +1.0
        make_record("q2", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),  # +0.631
        make_record("q3", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),  # +0.631
        make_record("q4", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),  # +0.631
    ]
    selected, counts = select_categories(records, K, examples_per_category=2)

    assert counts["bm25_wins"] == 4
    assert [case["query_id"] for case in selected["bm25_wins"]] == ["q1", "q2"]


def test_deterministic_tie_breaking_by_query_id():
    records = [
        make_record("qB", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),
        make_record("qA", {"R1": 1}, ["X1", "R1"], ["X1", "X2"]),
    ]
    selected, _ = select_categories(records, K, examples_per_category=3)

    assert [case["query_id"] for case in selected["bm25_wins"]] == ["qA", "qB"]


# --- diagnostics ---


def test_first_relevant_rank_extraction():
    qrels = {"R1": 1, "X1": 0}

    assert first_relevant_rank(["X1", "R1"], qrels) == 2
    assert first_relevant_rank(["X1", "X2"], qrels) is None
    assert relevant_in_top(["R1", "X1", "R2"], {"R1": 1, "R2": 1, "X1": 0}) == ["R1", "R2"]


def test_lexical_overlap_diagnostics():
    count, ratio = compute_lexical_overlap("machine learning", "machine learning algorithms")
    assert count == 2
    assert ratio == pytest.approx(1.0)

    count, ratio = compute_lexical_overlap("car repair", "machine learning")
    assert count == 0
    assert ratio == pytest.approx(0.0)

    count, ratio = compute_lexical_overlap("car repair", "car repair shop")
    assert count == 2
    assert ratio == pytest.approx(1.0)


def test_per_query_metric_payload_correctness():
    record = make_record(
        "q_bm25", {"R1": 1}, ["R1", "X1"], ["X1", "X2"],
        query_text="bm25 wins here",
    )

    entry = record["methods"]["BM25"]
    assert entry["metrics"] == evaluate_query(["R1", "X1"], {"R1": 1}, k=K)
    assert entry["first_relevant_rank"] == 1
    assert entry["relevant_in_top10"] == ["R1"]
    assert record["relevant"][0]["id"] == "R1"
    assert record["relevant"][0]["overlap_count"] == 3
    assert record["relevant"][0]["overlap_ratio"] == pytest.approx(1.0)


def test_reranker_before_after_rank_diagnostics():
    record = make_record("q_improve", {"R4": 1}, ["X1", "X2"], ["X1", "X2"],
                         ["X1", "R4"], ["R4", "X1"], candidate_ids=["X1", "R4"])

    assert record["reranker"]["hybrid_top20_ids"] == ["X1", "R4"]
    assert record["reranker"]["relevant_rank_before"] == {"R4": 2}
    assert record["reranker"]["relevant_rank_after"] == {"R4": 1}
    assert record["reranker"]["reranked_top10"][0]["id"] == "R4"


# --- full payload, JSON schema, report, determinism ---


def test_full_payload_counts_and_categories():
    payload = run_full()

    assert payload["metadata"]["num_queries"] == 6
    # q_improve and q_hurt also qualify as hybrid wins (hybrid beats both
    # single retrievers there), so the count is 3.
    assert payload["summary"]["category_counts"] == {
        "bm25_wins": 1,
        "dense_wins": 1,
        "hybrid_wins": 3,
        "reranker_improves": 1,
        "reranker_hurts": 1,
        "all_fail": 1,
    }
    assert set(payload["categories"]) == set(CATEGORIES)
    for cases in payload["categories"].values():
        assert isinstance(cases, list)


def test_json_schema():
    payload = run_full()

    assert set(payload["metadata"]) == {
        "dataset", "source", "k", "alpha", "hybrid_candidate_k",
        "reranker_pool", "dense_model", "reranker_model", "num_queries",
        "max_queries", "examples_per_category", "methods",
        "category_definitions",
    }
    assert set(payload["summary"]) == {"category_counts", "mean_metrics"}
    assert set(payload["summary"]["mean_metrics"]) == set(payload["metadata"]["methods"])

    case = payload["categories"]["reranker_improves"][0]
    assert set(case) == {
        "query_id", "query", "category", "selection_score", "relevant",
        "methods", "reranker",
    }
    for name in payload["metadata"]["methods"]:
        assert set(case["methods"][name]) == {
            "ranked_ids", "metrics", "first_relevant_rank", "relevant_in_top10",
        }


def test_markdown_report_generated_from_payload():
    payload = run_full()

    report = render_report(payload)

    assert report.startswith("# SciFact Failure Analysis")
    for title in ("## BM25 Wins, Dense Loses", "## Dense Wins, BM25 Loses",
                  "## Hybrid Wins", "## Reranker Improves",
                  "## Reranker Hurts", "## All Methods Fail",
                  "## Methodology", "## Summary"):
        assert title in report
    for text in ("bm25 wins here", "dense wins here", "improve rerank here",
                 "all fail here"):
        assert text in report
    assert "Observation:" in report
    assert "Hypothesis:" in report
    # Hybrid-win cases carry a data-driven hypothesis about rank movement.
    assert "Fusing the rankings moves relevant document" in report

    # Different payload -> different report (no hard-coded examples).
    modified = dict(payload)
    modified["categories"] = {category: [] for category in CATEGORIES}
    assert render_report(modified) != report


def test_repeated_execution_is_deterministic():
    first = run_full()
    second = run_full()

    assert first == second
    assert render_report(first) == render_report(second)


def test_smoke_and_full_paths_do_not_collide():
    assert output_path_for(20) == "outputs/scifact_failure_analysis_smoke20.json"
    assert output_path_for(None) == "outputs/scifact_failure_analysis.json"
    assert report_path_for(20) == "outputs/failure_analysis_smoke20.md"
    assert report_path_for(None) == "analysis/failure_analysis.md"
    assert output_path_for(20) != output_path_for(None)
    assert report_path_for(20) != report_path_for(None)
