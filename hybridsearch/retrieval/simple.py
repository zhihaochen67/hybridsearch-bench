from hybridsearch.data.toy import TOY_CORPUS


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def score_document(query: str, document: str) -> int:
    query_terms = tokenize(query)
    document_terms = set(tokenize(document))

    return sum(term in document_terms for term in query_terms)


def search(query: str, top_k: int = 3):
    results = []

    for document in TOY_CORPUS:
        score = score_document(query, document["text"])

        results.append(
            {
                "id": document["id"],
                "text": document["text"],
                "score": score,
            }
        )

    results.sort(key=lambda item: (-item["score"], item["id"]))

    return results[:top_k]


if __name__ == "__main__":
    query = "machine learning"

    print(f"Query: {query}\n")

    for rank, result in enumerate(search(query), start=1):
        print(
            f"{rank}. {result['id']} | "
            f"score={result['score']} | "
            f"{result['text']}"
        )
