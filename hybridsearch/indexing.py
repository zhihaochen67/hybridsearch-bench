from collections import defaultdict

from hybridsearch.data.toy import TOY_CORPUS
from hybridsearch.retrieval.simple import tokenize


def build_inverted_index(corpus):
    index = defaultdict(set)

    for document in corpus:
        doc_id = document["id"]

        for term in tokenize(document["text"]):
            index[term].add(doc_id)

    return dict(index)


if __name__ == "__main__":
    inverted_index = build_inverted_index(TOY_CORPUS)

    for term in sorted(inverted_index):
        print(f"{term}: {sorted(inverted_index[term])}")
