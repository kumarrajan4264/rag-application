from app.embeddings.embedding_model import EmbeddingModel
from app.vector_store.chroma_store import ChromaVectorStore


def main():

    print("=" * 70)
    print("METADATA FILTER TEST")
    print("=" * 70)

    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="rag_documents",
    )

    question = (
        "What machine learning experience "
        "does the candidate have?"
    )

    query_vector = embedding_model.embed_text(
        question
    )

    results = vector_store.search(
        query_vector=query_vector,
        limit=5,
        metadata_filter={
            "section_title": "EXPERIENCE"
        },
    )

    print()
    print("QUERY")
    print("------")
    print(question)

    print()
    print("FILTER")
    print("------")
    print("section_title = EXPERIENCE")

    print()
    print("RESULTS")
    print("-------")

    print(f"Results returned : {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.payload.get(
            "metadata",
            {},
        )

        print()
        print(
            f"[{index}] "
            f"Section: "
            f"{metadata.get('section_title')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page_number')}"
        )

        print(
            f"Score: "
            f"{result.score}"
        )

    print()
    print("=" * 70)
    print("METADATA FILTER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()