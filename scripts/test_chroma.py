from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.ingestion.chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vector_store.chroma_store import ChromaVectorStore


def main():

    pdf_path = "data/raw/samsung_guide.pdf"

    print("=" * 70)
    print("CHROMA DB TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load document
    # ---------------------------------------------------------

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    print(f"Documents loaded : {len(documents)}")

    # ---------------------------------------------------------
    # 2. Extract sections
    # ---------------------------------------------------------

    extractor = SectionExtractor()

    all_sections = []

    for document in documents:
        sections = extractor.extract_sections(document)
        all_sections.extend(sections)

    print(f"Sections extracted : {len(all_sections)}")

    # ---------------------------------------------------------
    # 3. Create chunks
    # ---------------------------------------------------------

    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
    )

    all_chunks = []

    for document in documents:

        sections = extractor.extract_sections(document)

        chunks = chunker.chunk_document(
            document=document,
            sections=sections,
        )

        all_chunks.extend(chunks)

    print(f"Chunks created : {len(all_chunks)}")

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    embedder = EmbeddingModel()

    texts = [
        chunk.content
        for chunk in all_chunks
    ]

    embeddings = embedder.embed_documents(texts)

    print(f"Embeddings generated : {len(embeddings)}")
    print(
        f"Embedding dimension : {len(embeddings[0])}"
    )

    # Attach embeddings to chunks
    for chunk, embedding in zip(
        all_chunks,
        embeddings,
    ):
        chunk.embedding = embedding

    # ---------------------------------------------------------
    # 5. Create Chroma store
    # ---------------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="rag_documents",
    )

    # ---------------------------------------------------------
    # 6. Insert chunks
    # ---------------------------------------------------------

    vector_store.add_chunks(all_chunks)

    print(
        f"Chunks stored in Chroma : "
        f"{vector_store.count()}"
    )

    # ---------------------------------------------------------
    # 7. Test semantic search
    # ---------------------------------------------------------

    query = (
        "What machine learning experience "
        "does the candidate have?"
    )

    print()
    print("=" * 70)
    print("SEMANTIC SEARCH")
    print("=" * 70)

    print(f"Query : {query}")

    query_embedding = embedder.embed_text(
        query
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    print()
    print(f"Results : {len(results['documents'][0])}")
    print("=" * 70)

    for index, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):

        print()
        print(f"Result {index + 1}")
        print("-" * 70)

        print(
            f"Section : "
            f"{metadata.get('section_title')}"
        )

        print(
            f"Page : "
            f"{metadata.get('page_number')}"
        )

        print(
            f"Distance : "
            f"{distance:.4f}"
        )

        print()
        print(document[:500])

    print()
    print("=" * 70)
    print("CHROMA TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()