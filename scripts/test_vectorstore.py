from app.vector_store.base import VectorStore
from app.vector_store.chroma_store import ChromaVectorStore


def main():

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="rag_documents",
    )

    print("=" * 70)
    print("VECTOR STORE ABSTRACTION TEST")
    print("=" * 70)

    print(
        f"Implementation : "
        f"{vector_store.__class__.__name__}"
    )

    print(
        f"Is VectorStore : "
        f"{isinstance(vector_store, VectorStore)}"
    )

    print(
        f"Stored vectors : "
        f"{vector_store.count()}"
    )

    print("=" * 70)
    print("VECTOR STORE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()