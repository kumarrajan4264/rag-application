from app.retrieval.qdrant_client import QdrantVectorStore


vector_store = QdrantVectorStore()

vector_store.create_collection()


print("\nCurrent collections:")

print(
    vector_store.test_connection()
)