from hybridsearch.retrieval.simple import score_document, search


def test_score_document():
    assert score_document(
        "machine learning",
        "machine learning algorithms",
    ) == 2

    assert score_document(
        "machine learning",
        "neural network deep learning",
    ) == 1

    assert score_document(
        "machine learning",
        "automobile maintenance",
    ) == 0


def test_search_ranking():
    results = search("machine learning", top_k=3)

    assert [result["id"] for result in results] == [
        "D1",
        "D4",
        "D2",
    ]

    assert [result["score"] for result in results] == [
        2,
        2,
        1,
    ]
