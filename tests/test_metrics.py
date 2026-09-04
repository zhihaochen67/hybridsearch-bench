"""Deterministic tests for the IR evaluation metrics (Phase 5).

Every expectation is hand-computable from the metric definitions in
``hybridsearch/evaluation/metrics.py``; nothing here touches models or the
network.
"""

import math

import pytest

from hybridsearch.evaluation.metrics import (
    evaluate_query,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Reference case (also printed by the module demo):
#   ranked_ids = ["D2", "D1", "D3"]
#   qrels = {"D1": 2, "D2": 1, "D3": 0, "D4": 1}
#   k = 3
#   precision@3 = 2/3, recall@3 = 2/3, mrr@3 = 1.0,
#   ndcg@3 = (1 + 3/log2(3)) / (3 + 1/log2(3) + 1/2).
REFERENCE_RANKED = ["D2", "D1", "D3"]
REFERENCE_QRELS = {"D1": 2, "D2": 1, "D3": 0, "D4": 1}


def _gain(rel):
    """Independent re-derivation of the nDCG gain, for hand-checked math."""
    return 2**rel - 1


# --- precision@k ---


def test_precision_at_k_reference_value():
    assert precision_at_k(REFERENCE_RANKED, REFERENCE_QRELS, 3) == pytest.approx(
        2 / 3
    )


def test_precision_denominator_is_k_even_with_fewer_results():
    # Only D1 is relevant and only two documents were returned, so with
    # k=5 the denominator must still be 5 (missing positions are
    # non-relevant): 1/5.
    assert precision_at_k(["D1", "D2"], {"D1": 1, "D2": 0}, 5) == pytest.approx(
        1 / 5
    )


def test_precision_k_one():
    assert precision_at_k(["D1", "D2"], {"D1": 1, "D2": 0}, 1) == pytest.approx(
        1.0
    )
    assert precision_at_k(["D2", "D1"], {"D1": 1, "D2": 0}, 1) == pytest.approx(
        0.0
    )


# --- recall@k ---


def test_recall_at_k_reference_value():
    assert recall_at_k(REFERENCE_RANKED, REFERENCE_QRELS, 3) == pytest.approx(
        2 / 3
    )


def test_recall_no_relevant_qrels_returns_zero():
    assert recall_at_k(["D1", "D2"], {"D1": 0, "D2": 0}, 3) == 0.0


def test_recall_finds_all_relevant():
    assert recall_at_k(["D1", "D2"], {"D1": 1, "D2": 1}, 2) == pytest.approx(1.0)


# --- mrr@k ---


def test_mrr_first_relevant_at_rank_one():
    assert mrr_at_k(["D1", "D2"], {"D1": 2, "D2": 1}, 2) == pytest.approx(1.0)


def test_mrr_first_relevant_at_later_rank():
    # The first (and only relevant) document sits at rank 3: 1/3.
    assert mrr_at_k(
        ["D1", "D2", "D3"], {"D1": 0, "D2": 0, "D3": 1}, 3
    ) == pytest.approx(1 / 3)


def test_mrr_no_relevant_result_returns_zero():
    assert mrr_at_k(["D1", "D2"], {"D1": 0, "D2": 0}, 2) == 0.0


def test_mrr_only_considers_first_relevant_document():
    # D2 (rank 2) and D3 (rank 3) are both relevant: only the first counts,
    # so the value is 1/2, not 1/2 + 1/3.
    assert mrr_at_k(
        ["D1", "D2", "D3"], {"D1": 0, "D2": 1, "D3": 1}, 3
    ) == pytest.approx(0.5)


def test_mrr_ignores_relevant_documents_beyond_k():
    assert mrr_at_k(["D1", "D2"], {"D1": 0, "D2": 1}, 1) == 0.0


# --- ndcg@k ---


def test_ndcg_perfect_ranking_is_one():
    ranked = ["D1", "D2", "D4"]
    qrels = {"D1": 2, "D2": 1, "D3": 0, "D4": 1}

    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(1.0)


def test_ndcg_reference_hand_computed():
    expected_dcg = _gain(1) / math.log2(2) + _gain(2) / math.log2(3)
    expected_idcg = (
        _gain(2) / math.log2(2)
        + _gain(1) / math.log2(3)
        + _gain(1) / math.log2(4)
    )

    assert ndcg_at_k(REFERENCE_RANKED, REFERENCE_QRELS, 3) == pytest.approx(
        expected_dcg / expected_idcg
    )
    # Sanity-check the hand-computed decimal value.
    assert ndcg_at_k(REFERENCE_RANKED, REFERENCE_QRELS, 3) == pytest.approx(
        0.70028, rel=1e-4
    )


def test_ndcg_uses_graded_relevance():
    ranked = ["D1", "D2", "D3"]
    qrels = {"D1": 3, "D2": 1, "D3": 0, "D4": 2}

    dcg = _gain(3) / math.log2(2) + _gain(1) / math.log2(3)
    idcg = (
        _gain(3) / math.log2(2)
        + _gain(2) / math.log2(3)
        + _gain(1) / math.log2(4)
    )
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(dcg / idcg)

    # A binary reading of the same qrels gives a different value, proving
    # that graded relevance is actually used.
    binary = {doc_id: min(1, rel) for doc_id, rel in qrels.items()}
    assert ndcg_at_k(ranked, qrels, 3) != pytest.approx(ndcg_at_k(ranked, binary, 3))


def test_ndcg_idcg_zero_returns_zero():
    assert ndcg_at_k(["D1", "D2"], {"D1": 0, "D2": 0}, 2) == 0.0


# --- unknown ids and k handling ---


def test_unknown_document_ids_treated_as_relevance_zero():
    qrels = {"D1": 1, "D2": 0}

    # D9 is unjudged: it never counts as relevant but still occupies a slot.
    assert precision_at_k(["D9", "D1"], qrels, 2) == pytest.approx(1 / 2)
    assert recall_at_k(["D9", "D1"], qrels, 2) == pytest.approx(1.0)
    assert mrr_at_k(["D9", "D1"], qrels, 2) == pytest.approx(0.5)
    assert ndcg_at_k(["D9", "D1"], qrels, 2) == pytest.approx(
        (1 / math.log2(3)) / 1.0
    )


def test_k_larger_than_result_list():
    ranked = ["D1", "D2"]
    qrels = {"D1": 1, "D2": 1, "D3": 1}

    assert precision_at_k(ranked, qrels, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranked, qrels, 5) == pytest.approx(2 / 3)


@pytest.mark.parametrize("k", [0, -1, -100, "3", None, 2.5, True])
def test_invalid_k_raises(k):
    ranked, qrels = ["D1"], {"D1": 1}

    for metric in (precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            metric(ranked, qrels, k)

    with pytest.raises(ValueError):
        evaluate_query(ranked, qrels, k=k)


# --- duplicate-id policy ---


@pytest.mark.parametrize(
    "metric", [precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k]
)
def test_duplicate_ranked_ids_raise(metric):
    with pytest.raises(ValueError):
        metric(["D1", "D1", "D2"], {"D1": 1, "D2": 1}, 3)


def test_duplicates_rejected_even_beyond_k():
    # The policy validates the whole ranked list, not just the top-k prefix.
    with pytest.raises(ValueError):
        precision_at_k(["D1", "D2", "D2"], {"D1": 1, "D2": 0}, 1)


# --- aggregation ---


def test_evaluate_query_keys_and_values():
    metrics = evaluate_query(REFERENCE_RANKED, REFERENCE_QRELS, k=3)

    assert set(metrics) == {"precision@3", "recall@3", "mrr@3", "ndcg@3"}
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["recall@3"] == pytest.approx(2 / 3)
    assert metrics["mrr@3"] == pytest.approx(1.0)

    expected_dcg = _gain(1) / math.log2(2) + _gain(2) / math.log2(3)
    expected_idcg = (
        _gain(2) / math.log2(2)
        + _gain(1) / math.log2(3)
        + _gain(1) / math.log2(4)
    )
    assert metrics["ndcg@3"] == pytest.approx(expected_dcg / expected_idcg)


def test_mean_metrics_arithmetic_mean():
    means = mean_metrics(
        [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
        ]
    )

    assert means == {"a": 2.0, "b": 3.0}


def test_mean_metrics_empty_input():
    assert mean_metrics([]) == {}


def test_mean_metrics_single_query_is_identity():
    metrics = evaluate_query(REFERENCE_RANKED, REFERENCE_QRELS, k=3)

    assert mean_metrics([metrics]) == metrics


# --- determinism ---


def test_repeated_runs_are_deterministic():
    first = evaluate_query(REFERENCE_RANKED, REFERENCE_QRELS, k=3)
    second = evaluate_query(REFERENCE_RANKED, REFERENCE_QRELS, k=3)

    assert first == second
    assert mean_metrics([first, second]) == mean_metrics([second, first])
