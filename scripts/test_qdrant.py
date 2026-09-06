from app.retrieval.qdrant_client import QdrantVectorStore


vector_store = QdrantVectorStore()

result = vector_store.test_connection()


print("=" * 70)
print("QDRANT CONNECTION TEST")
print("=" * 70)

print(result)