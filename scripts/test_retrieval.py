from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.qdrant_client import QdrantVectorStore


QUERY = """
What experience does the candidate have
with machine learning and risk prediction?
"""


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embedding_model = EmbeddingModel()


# --------------------------------------------------
# 2. Create query embedding
# --------------------------------------------------

query_vector = embedding_model.embed_text(
    QUERY
)


# --------------------------------------------------
# 3. Connect to Qdrant
# --------------------------------------------------

vector_store = QdrantVectorStore()


# --------------------------------------------------
# 4. Search
# --------------------------------------------------

results = vector_store.search(
    query_vector=query_vector,
    limit=5,
)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("=" * 70)
print("SEMANTIC RETRIEVAL TEST")
print("=" * 70)

print(f"Query: {QUERY.strip()}")

print(f"\nResults found: {len(results)}")


for index, result in enumerate(results, start=1):

    print("\n" + "-" * 70)

    print(f"Rank       : {index}")
    print(f"Score      : {result.score}")
    print(f"Point ID   : {result.id}")

    payload = result.payload

    print(
        f"Chunk ID   : "
        f"{payload.get('chunk_id')}"
    )

    print(
        f"Source     : "
        f"{payload.get('metadata', {}).get('source')}"
    )

    print(
        f"Page       : "
        f"{payload.get('metadata', {}).get('page_number')}"
    )

    print("\nContent:")

    print(payload.get("content"))