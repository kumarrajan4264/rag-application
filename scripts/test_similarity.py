from sentence_transformers import util

from app.embeddings.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()


texts = [
    "How should machine learning models be deployed?",
    "Machine learning models should be deployed in production.",
    "The weather is very pleasant today.",
]


vectors = embedding_model.embed_documents(texts)


similarity_01 = util.cos_sim(
    vectors[0],
    vectors[1],
)

similarity_02 = util.cos_sim(
    vectors[0],
    vectors[2],
)


print("=" * 70)
print("SEMANTIC SIMILARITY TEST")
print("=" * 70)

print(
    "Query ↔ ML deployment:",
    similarity_01.item(),
)

print(
    "Query ↔ Weather:",
    similarity_02.item(),
)