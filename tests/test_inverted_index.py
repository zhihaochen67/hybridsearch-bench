from hybridsearch.data.toy import TOY_CORPUS
from hybridsearch.indexing import build_inverted_index


def test_inverted_index():
    index = build_inverted_index(TOY_CORPUS)

    assert index["machine"] == {"D1", "D4"}
    assert index["learning"] == {"D1", "D2", "D4"}
    assert index["automobile"] == {"D3"}
    assert index["neural"] == {"D2"}
