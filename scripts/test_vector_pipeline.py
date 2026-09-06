from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.qdrant_client import QdrantVectorStore


PDF_PATH = "data/raw/samsung_guide.pdf"


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

loader = PDFLoader()

documents = loader.load(PDF_PATH)


# --------------------------------------------------
# 2. Chunk documents
# --------------------------------------------------

chunker = TextChunker(
    chunk_size=500,
    overlap=100,
)


chunks = []

for document in documents:

    chunks.extend(
        chunker.chunk_document(document)
    )


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()


# --------------------------------------------------
# 4. Connect Qdrant
# --------------------------------------------------

vector_store = QdrantVectorStore()


# --------------------------------------------------
# 5. Insert chunks
# --------------------------------------------------

for chunk in chunks:

    vector = embedding_model.embed_text(
        chunk.content
    )

    payload = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": chunk.metadata,
    }

    vector_store.upsert_chunk(
        chunk_id=chunk.chunk_id,
        vector=vector,
        payload=payload,
    )


print("=" * 70)
print("VECTOR PIPELINE COMPLETE")
print("=" * 70)

print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(chunks)}")
print("Vectors inserted into Qdrant.")