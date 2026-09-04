"""Deterministic unit tests for hybrid score fusion (Phase 3).

Pure-function tests: no retriever, no model, no network.
"""

import pytest

from hybridsearch.retrieval.fusion import (
    min_max_normalize,
    rank_by_score,
    reciprocal_rank_fusion,
    weighted_fusion,
)


# --- min-max normalization ---


def test_min_max_normalization():
    assert min_max_normalize({"a": 2.0, "b": 4.0, "c": 6.0}) == {
        "a": 0.0,
        "b": 0.5,
        "c": 1.0,
    }


def test_min_max_normalization_with_negative_scores():
    assert min_max_normalize({"a": -10.0, "b": 0.0, "c": 10.0}) == {
        "a": 0.0,
        "b": 0.5,
        "c": 1.0,
    }


def test_min_max_normalization_constant_zero_scores_stay_zero():
    assert min_max_normalize({"a": 0.0, "b": 0.0, "c": 0.0}) == {
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def test_min_max_normalization_constant_positive_scores_keep_signal():
    # All candidates tie at a positive score: every one of them is a
    # genuine best-of-list match, so none may look like "not retrieved".
    assert min_max_normalize({"a": 5.0, "b": 5.0, "c": 5.0}) == {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
    }


def test_min_max_normalization_single_positive_score_keeps_signal():
    assert min_max_normalize({"a": 3.0}) == {"a": 1.0}


def test_min_max_normalization_single_zero_score_stays_zero():
    assert min_max_normalize({"a": 0.0}) == {"a": 0.0}


def test_min_max_normalization_constant_negative_scores_stay_zero():
    assert min_max_normalize({"a": -2.0, "b": -2.0}) == {
        "a": 0.0,
        "b": 0.0,
    }


def test_min_max_normalization_empty_mapping():
    assert min_max_normalize({}) == {}


# --- weighted fusion ---


def test_weighted_fusion_math():
    sparse = {"a": 10.0, "b": 20.0, "c": 30.0}  # norm: 0.0, 0.5, 1.0
    dense = {"a": 0.7, "b": 0.8, "c": 0.9}  # norm: 0.0, 0.5, 1.0

    fused = weighted_fusion(sparse, dense, alpha=0.5)

    assert fused["a"]["score"] == pytest.approx(0.0)
    assert fused["b"]["score"] == pytest.approx(0.5)
    assert fused["c"]["score"] == pytest.approx(1.0)


def test_weighted_fusion_alpha_scales_components():
    sparse = {"a": 10.0, "b": 20.0, "c": 25.0}  # norm: 0.0, 2/3, 1.0
    dense = {"a": 0.7, "b": 0.8, "c": 0.9}  # norm: 0.0, 0.5, 1.0

    fused = weighted_fusion(sparse, dense, alpha=0.4)

    assert fused["a"]["score"] == pytest.approx(0.0)
    assert fused["b"]["score"] == pytest.approx(0.4 * 2 / 3 + 0.6 * 0.5)
    assert fused["c"]["score"] == pytest.approx(0.4 * 1.0 + 0.6 * 1.0)


def test_weighted_fusion_alpha_one_is_sparse_only():
    sparse = {"a": 3.0, "b": 9.0}  # norm: 0.0, 1.0
    dense = {"a": 0.2, "b": 0.9}  # norm: 0.0, 1.0

    fused = weighted_fusion(sparse, dense, alpha=1.0)

    assert fused["a"]["score"] == pytest.approx(0.0)
    assert fused["b"]["score"] == pytest.approx(1.0)
    # Dense scores must not influence the result at all.
    assert fused["b"]["score"] == pytest.approx(
        min_max_normalize(sparse)["b"]
    )


def test_weighted_fusion_alpha_zero_is_dense_only():
    sparse = {"a": 0.01, "b": 0.9}  # would rank b first if used
    dense = {"a": 5.0, "b": 1.0}  # norm: 1.0, 0.0

    fused = weighted_fusion(sparse, dense, alpha=0.0)

    assert fused["a"]["score"] == pytest.approx(1.0)
    assert fused["b"]["score"] == pytest.approx(0.0)
    assert fused["a"]["score"] == pytest.approx(
        min_max_normalize(dense)["a"]
    )


def test_weighted_fusion_union_and_missing_documents():
    sparse = {"a": 1.0, "b": 2.0}  # norm: 0.0, 1.0
    dense = {"b": 0.9, "c": 0.5}  # norm: 1.0, 0.0

    fused = weighted_fusion(sparse, dense, alpha=0.5)

    assert set(fused) == {"a", "b", "c"}

    # "a" only in sparse, "c" only in dense, "b" in both.
    assert fused["a"]["score"] == pytest.approx(0.5 * 0.0 + 0.5 * 0.0)
    assert fused["b"]["score"] == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)
    assert fused["c"]["score"] == pytest.approx(0.5 * 0.0 + 0.5 * 0.0)

    # Missing raw and normalized contributions are treated as 0.
    assert fused["a"]["dense_score"] == 0.0
    assert fused["a"]["normalized_dense_score"] == 0.0
    assert fused["c"]["sparse_score"] == 0.0
    assert fused["c"]["normalized_sparse_score"] == 0.0


def test_weighted_fusion_diagnostics_are_preserved():
    fused = weighted_fusion({"a": 1.0, "b": 3.0}, {"b": 0.8}, alpha=0.5)

    assert set(fused["a"]) == {
        "score",
        "sparse_score",
        "dense_score",
        "normalized_sparse_score",
        "normalized_dense_score",
    }
    assert fused["a"]["sparse_score"] == 1.0
    assert fused["a"]["normalized_sparse_score"] == 0.0
    assert fused["b"]["dense_score"] == pytest.approx(0.8)
    # Constant positive list: the retrieval signal is preserved.
    assert fused["b"]["normalized_dense_score"] == 1.0


def test_constant_positive_normalization_preserves_retrieval_signal():
    sparse = {"a": 2.0, "b": 2.0}  # all equal positive
    dense = {"a": 0.8, "c": 0.6}  # norm: 1.0, 0.0

    fused = weighted_fusion(sparse, dense, alpha=0.5)

    assert fused["a"]["normalized_sparse_score"] == pytest.approx(1.0)
    assert fused["b"]["normalized_sparse_score"] == pytest.approx(1.0)

    # "b" was retrieved by the sparse retriever only; the constant-positive
    # list keeps that signal, so "b" still outranks "c".
    assert fused["a"]["score"] == pytest.approx(1.0)
    assert fused["b"]["score"] == pytest.approx(0.5)
    assert fused["c"]["score"] == pytest.approx(0.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.5, "0.5", None])
def test_weighted_fusion_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        weighted_fusion({"a": 1.0}, {"a": 1.0}, alpha=alpha)


# --- reciprocal rank fusion ---


def test_rrf_math():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)

    assert fused["a"]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"]["ranks"] == [1, 2]
    assert fused["b"]["ranks"] == [2, 1]


def test_rrf_ranks_are_one_based():
    fused = reciprocal_rank_fusion([["a"]], k=60)

    # Rank 1 contributes exactly 1 / (k + 1), never 1 / k.
    assert fused["a"]["score"] == pytest.approx(1 / 61)
    assert fused["a"]["ranks"] == [1]


def test_rrf_union_behavior():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)

    assert set(fused) == {"a", "b", "c"}

    # Documents present in only one list are kept with that contribution.
    assert fused["a"]["score"] == pytest.approx(1 / 61)
    assert fused["a"]["ranks"] == [1, None]
    assert fused["c"]["score"] == pytest.approx(1 / 62)
    assert fused["c"]["ranks"] == [None, 2]
    assert fused["b"]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["b"]["ranks"] == [2, 1]


def test_rrf_duplicate_within_one_list_uses_best_rank():
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=60)

    assert fused["a"]["score"] == pytest.approx(1 / 61)
    assert fused["a"]["ranks"] == [1]
    assert fused["b"]["score"] == pytest.approx(1 / 63)


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == {}


@pytest.mark.parametrize("k", [0, -1, "60", None])
def test_rrf_invalid_k_raises(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=k)


# --- deterministic ordering ---


def test_rank_by_score_descending_then_id():
    scores = {"b": 2.0, "a": 2.0, "c": 1.0, "d": 3.0}

    assert rank_by_score(scores) == ["d", "a", "b", "c"]


def test_rank_by_score_repeated_calls_are_identical():
    scores = {"x": 0.3, "y": 0.3, "z": 0.1}

    assert rank_by_score(scores) == rank_by_score(scores)
