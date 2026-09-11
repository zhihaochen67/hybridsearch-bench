"""Offline tests for the FAISS index abstraction (Phase 14A).

Everything runs on synthetic numpy embeddings with the real FAISS
library — no network, no models, fully deterministic (seeded data, FAISS
default clustering seed). The DenseRetriever integration tests reuse the
fake encoder pattern from the earlier phases.
"""

import numpy as np
import pytest

from hybridsearch.retrieval.dense import DenseRetriever, l2_normalize
from hybridsearch.retrieval.vector_index import (
    FaissIndexInfo,
    SUPPORTED_INDEX_TYPES,
    ann_recall_at_k,
    build_faiss_index,
    search_faiss_index,
    serialized_index_size,
    validate_hnsw_params,
    validate_index_type,
    validate_ivf_params,
)


def make_embeddings(n=64, d=8, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float64)
    return l2_normalize(vectors)


EMBEDDINGS = make_embeddings()


def flat_reference_ids(query_embedding, embeddings, k):
    """Exact top-k ids by brute-force inner product (ties: lower id)."""
    scores = embeddings @ query_embedding
    order = sorted(range(len(embeddings)), key=lambda i: (-scores[i], i))
    return order[:k]


# --- flat ---


def test_flat_returns_exact_nearest_neighbors():
    info = build_faiss_index(EMBEDDINGS, "flat")
    query = EMBEDDINGS[3]

    _, indices = search_faiss_index(info, query, 10)

    assert indices[0].tolist() == flat_reference_ids(query, EMBEDDINGS, 10)


def test_flat_topk_scores_match_brute_force():
    info = build_faiss_index(EMBEDDINGS, "flat")
    query = EMBEDDINGS[3]

    scores, indices = search_faiss_index(info, query, 5)

    for score, index in zip(scores[0], indices[0]):
        assert score == pytest.approx(float(EMBEDDINGS[int(index)] @ query))


def test_flat_batch_search():
    info = build_faiss_index(EMBEDDINGS, "flat")

    scores, indices = search_faiss_index(info, EMBEDDINGS[:4], 3)

    assert scores.shape == (4, 3)
    assert indices.shape == (4, 3)


# --- hnsw ---


def test_hnsw_builds_and_searches():
    info = build_faiss_index(
        EMBEDDINGS, "hnsw", {"M": 16, "efConstruction": 100, "efSearch": 64}
    )

    assert info.index.ntotal == 64
    assert info.index.d == 8
    assert info.requires_training is False
    assert info.trained is True

    _, indices = search_faiss_index(info, EMBEDDINGS[0], 10)
    assert indices.shape == (1, 10)
    assert all(index >= 0 for index in indices[0])


def test_hnsw_recovers_exact_neighbors_on_tiny_corpus():
    # With efSearch >= corpus size, HNSW is exact on this small set.
    info = build_faiss_index(EMBEDDINGS[:16], "hnsw", {"M": 16, "efSearch": 64})
    query = EMBEDDINGS[0]

    _, indices = search_faiss_index(info, query, 8)

    assert indices[0].tolist() == flat_reference_ids(query, EMBEDDINGS[:16], 8)


def test_hnsw_efSearch_override_is_request_scoped():
    class RecordingHnsw:
        efSearch = 4

    class RecordingIndex:
        def __init__(self):
            self.hnsw = RecordingHnsw()
            self.seen_ef_search = []

        def search(self, queries, top_k):
            self.seen_ef_search.append(self.hnsw.efSearch)
            return (
                np.zeros((len(queries), top_k), dtype=np.float32),
                np.zeros((len(queries), top_k), dtype=np.int64),
            )

    index = RecordingIndex()
    info = FaissIndexInfo(
        index=index,
        index_type="hnsw",
        dimension=8,
        n_vectors=64,
        params={"M": 16, "efConstruction": 200, "efSearch": 4},
        requires_training=False,
        trained=True,
    )

    _, first_indices = search_faiss_index(info, EMBEDDINGS[0], 16)
    _, second_indices = search_faiss_index(info, EMBEDDINGS[1], 2)

    assert first_indices.shape == (1, 16)
    assert second_indices.shape == (1, 2)
    assert index.seen_ef_search == [16, 4]
    assert index.hnsw.efSearch == 4


def test_hnsw_efSearch_override_is_serialized_across_requests():
    import threading

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    class RecordingHnsw:
        efSearch = 4

    class BlockingIndex:
        def __init__(self):
            self.hnsw = RecordingHnsw()
            self.seen_ef_search = []

        def search(self, queries, top_k):
            self.seen_ef_search.append(self.hnsw.efSearch)
            if len(self.seen_ef_search) == 1:
                first_entered.set()
                release_first.wait(timeout=1.0)
            else:
                second_entered.set()
            return (
                np.zeros((len(queries), top_k), dtype=np.float32),
                np.zeros((len(queries), top_k), dtype=np.int64),
            )

    index = BlockingIndex()
    info = FaissIndexInfo(
        index=index,
        index_type="hnsw",
        dimension=8,
        n_vectors=64,
        params={"M": 16, "efConstruction": 200, "efSearch": 4},
        requires_training=False,
        trained=True,
    )
    errors = []

    def run_search(query, top_k):
        try:
            search_faiss_index(info, query, top_k)
        except BaseException as error:  # surfaced in the main test thread
            errors.append(error)

    first = threading.Thread(target=run_search, args=(EMBEDDINGS[0], 16))
    second = threading.Thread(target=run_search, args=(EMBEDDINGS[1], 2))
    first.start()
    assert first_entered.wait(timeout=1.0)
    second.start()

    # The second request cannot enter FAISS while the first request owns the
    # temporary efSearch override.
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert index.seen_ef_search == [16, 4]
    assert index.hnsw.efSearch == 4


# --- ivf ---


def test_ivf_trains_builds_and_searches():
    info = build_faiss_index(EMBEDDINGS, "ivf", {"nlist": 8, "nprobe": 2})

    assert info.index.is_trained
    assert info.trained is True
    assert info.requires_training is True
    assert info.index.ntotal == 64

    _, indices = search_faiss_index(info, EMBEDDINGS[0], 10)
    assert indices.shape == (1, 10)
    assert all(index >= 0 for index in indices[0])


def test_ivf_full_probe_is_exact():
    info = build_faiss_index(EMBEDDINGS[:16], "ivf", {"nlist": 4, "nprobe": 4})
    query = EMBEDDINGS[0]

    _, indices = search_faiss_index(info, query, 8)

    assert indices[0].tolist() == flat_reference_ids(query, EMBEDDINGS[:16], 8)


# --- normalized inner-product behavior ---


def test_normalized_query_in_corpus_scores_one_for_every_type():
    corpus = EMBEDDINGS[:20]
    query = corpus[7]  # exact member of the corpus

    for index_type in SUPPORTED_INDEX_TYPES:
        info = build_faiss_index(corpus, index_type, {"nlist": 4, "nprobe": 4} if index_type == "ivf" else {})
        scores, indices = search_faiss_index(info, query, 1)

        assert int(indices[0][0]) == 7
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


# --- validation ---


@pytest.mark.parametrize("bad", ["pizza", "", "Flat", None])
def test_index_type_validation(bad):
    with pytest.raises(ValueError):
        validate_index_type(bad)


@pytest.mark.parametrize(
    "params",
    [
        {"M": 0},
        {"M": -4},
        {"M": True},
        {"M": 2.5},
        {"efConstruction": 0},
        {"efSearch": -1},
        {"bogus": 3},
    ],
)
def test_hnsw_param_validation(params):
    with pytest.raises(ValueError):
        validate_hnsw_params(params)


def test_hnsw_defaults():
    resolved = validate_hnsw_params({})

    assert resolved == {"M": 32, "efConstruction": 200, "efSearch": 64}


@pytest.mark.parametrize(
    "params",
    [
        {"nlist": 0},
        {"nlist": -2},
        {"nlist": True},
        {"nprobe": 0},
        {"nprobe": -1},
        {"nprobe": 9},  # > nlist
        {"bogus": 3},
    ],
)
def test_ivf_param_validation(params):
    with pytest.raises(ValueError):
        validate_ivf_params(params, n_train=64)


def test_ivf_defaults():
    resolved = validate_ivf_params({}, n_train=57638)

    assert resolved == {"nlist": 512, "nprobe": 32}


def test_too_small_training_set_handled_clearly():
    with pytest.raises(ValueError, match="training"):
        build_faiss_index(EMBEDDINGS[:4], "ivf", {"nlist": 8, "nprobe": 2})


# --- ann recall vs flat ---


def test_ann_recall_at_k_exact_value():
    ann = [1, 2, 3, 4, 5]
    flat = [1, 2, 6, 7, 8]

    assert ann_recall_at_k(ann, flat, 5) == pytest.approx(2 / 5)
    assert ann_recall_at_k(ann, flat, 2) == pytest.approx(1.0)
    assert ann_recall_at_k([9, 9, 9], [1, 2, 3], 3) == pytest.approx(0.0)


def test_flat_vs_flat_ann_recall_is_one():
    first = build_faiss_index(EMBEDDINGS, "flat")
    second = build_faiss_index(EMBEDDINGS, "flat")

    recalls = []
    for query in EMBEDDINGS[:5]:
        _, first_ids = search_faiss_index(first, query, 10)
        _, second_ids = search_faiss_index(second, query, 10)
        recalls.append(ann_recall_at_k(first_ids[0].tolist(), second_ids[0].tolist(), 10))

    assert recalls == pytest.approx([1.0] * 5)


@pytest.mark.parametrize("k", [0, -1])
def test_ann_recall_rejects_bad_k(k):
    with pytest.raises(ValueError):
        ann_recall_at_k([1, 2], [1, 2], k)


def test_ann_recall_rejects_short_lists():
    with pytest.raises(ValueError, match="at least"):
        ann_recall_at_k([1], [1, 2, 3], 3)


# --- serialized size ---


def test_serialized_index_size_roundtrip():
    info = build_faiss_index(EMBEDDINGS[:16], "flat")

    size = serialized_index_size(info.index)

    assert size > 0
    assert size == serialized_index_size(info.index)  # reproducible


def test_serialized_size_grows_with_corpus():
    small = serialized_index_size(build_faiss_index(EMBEDDINGS[:8], "flat").index)
    large = serialized_index_size(build_faiss_index(EMBEDDINGS[:64], "flat").index)

    assert large > small


# --- metadata + determinism ---


def test_faiss_index_info_metadata():
    info = build_faiss_index(EMBEDDINGS, "hnsw", {"M": 16})

    assert info.index_type == "hnsw"
    assert info.dimension == 8
    assert info.n_vectors == 64
    assert info.params == {"M": 16, "efConstruction": 200, "efSearch": 64}


@pytest.mark.parametrize(
    "index_type,params",
    [
        ("flat", {}),
        ("hnsw", {"M": 16}),
        ("ivf", {"nlist": 8, "nprobe": 4}),
    ],
)
def test_repeated_builds_are_deterministic(index_type, params):
    first = build_faiss_index(EMBEDDINGS, index_type, params)
    second = build_faiss_index(EMBEDDINGS, index_type, params)

    for query in EMBEDDINGS[:3]:
        _, first_ids = search_faiss_index(first, query, 10)
        _, second_ids = search_faiss_index(second, query, 10)
        assert first_ids.tolist() == second_ids.tolist()


# --- DenseRetriever integration ---


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = {
            text: np.asarray(vector, dtype=np.float64)
            for text, vector in vectors.items()
        }
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=np.float64)


VECTORS = {
    "alpha": [2, 0, 0],
    "beta": [0, 4, 0],
    "gamma": [0, 0, 6],
    "delta": [2, 0, 0],
    "q-x": [2, 0, 0],
    "q-mixed": [3, 4, 0],
}
CORPUS = [
    {"id": "A", "text": "alpha"},
    {"id": "B", "text": "beta"},
    {"id": "C", "text": "gamma"},
    {"id": "D", "text": "delta"},
]


def test_dense_retriever_default_remains_flat():
    import faiss

    retriever = DenseRetriever(CORPUS, encoder=FakeEncoder(VECTORS))

    assert retriever.index_info.index_type == "flat"
    assert isinstance(retriever.index, faiss.IndexFlatIP)


def test_dense_retriever_explicit_flat_matches_default():
    default = DenseRetriever(CORPUS, encoder=FakeEncoder(VECTORS))
    explicit = DenseRetriever(
        CORPUS, encoder=FakeEncoder(VECTORS), index_type="flat"
    )

    assert default.search("q-mixed", top_k=4) == explicit.search("q-mixed", top_k=4)


def test_dense_retriever_hnsw_and_ivf_search():
    hnsw = DenseRetriever(
        CORPUS, encoder=FakeEncoder(VECTORS), index_type="hnsw",
        index_params={"M": 16, "efSearch": 64},
    )
    assert hnsw.index_info.index_type == "hnsw"
    assert [r["id"] for r in hnsw.search("q-x", top_k=2)] == ["A", "D"]

    ivf = DenseRetriever(
        CORPUS, encoder=FakeEncoder(VECTORS), index_type="ivf",
        index_params={"nlist": 4, "nprobe": 4},
    )
    assert ivf.index_info.index_type == "ivf"
    assert ivf.index_info.requires_training is True
    assert ivf.search("q-mixed", top_k=1)[0]["id"] == "B"


def test_dense_retriever_rebuild_index_reuses_embeddings():
    encoder = FakeEncoder(VECTORS)
    retriever = DenseRetriever(CORPUS, encoder=encoder)

    retriever.rebuild_index("hnsw", {"M": 16})
    assert retriever.index_info.index_type == "hnsw"
    assert [r["id"] for r in retriever.search("q-x", top_k=2)] == ["A", "D"]

    retriever.rebuild_index("flat")
    assert retriever.index_info.index_type == "flat"

    corpus_calls = [call for call in encoder.calls if len(call) == len(CORPUS)]
    assert corpus_calls == [["alpha", "beta", "gamma", "delta"]]


def test_dense_retriever_rebuild_index_updates_params_without_type():
    encoder = FakeEncoder(VECTORS)
    retriever = DenseRetriever(
        CORPUS,
        encoder=encoder,
        index_type="hnsw",
        index_params={"M": 16, "efConstruction": 100, "efSearch": 8},
    )

    retriever.rebuild_index(
        index_params={"M": 8, "efConstruction": 80, "efSearch": 12}
    )

    assert retriever.index_info.index_type == "hnsw"
    assert retriever.index_info.params == {
        "M": 8,
        "efConstruction": 80,
        "efSearch": 12,
    }
    assert retriever.index.hnsw.efSearch == 12
    corpus_calls = [call for call in encoder.calls if len(call) == len(CORPUS)]
    assert corpus_calls == [["alpha", "beta", "gamma", "delta"]]


def test_dense_retriever_rejects_unknown_index_type():
    with pytest.raises(ValueError, match="index type"):
        DenseRetriever(CORPUS, encoder=FakeEncoder(VECTORS), index_type="pizza")
