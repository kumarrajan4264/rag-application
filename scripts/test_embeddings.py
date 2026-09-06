from app.embeddings.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()


text = """
Machine learning models should be deployed
using the approved production platform.
"""


vector = embedding_model.embed_text(text)


print("=" * 70)
print("EMBEDDING TEST")
print("=" * 70)

print(f"Model name       : {embedding_model.model_name}")
print(f"Vector dimensions : {len(vector)}")

print("\nFirst 10 values:")
print(vector[:10])