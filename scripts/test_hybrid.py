from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.ingestion.chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vector_store.chroma_store import ChromaVectorStore


def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    pdf_loader = PDFLoader()

    section_extractor = SectionExtractor()

    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
    )

    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="rag_documents",
    )

    documents = pdf_loader.load(
        "data/raw/samsung_guide.pdf"
    )

    all_chunks = []

    for document in documents:

        sections = (
            section_extractor.extract_sections(
                document
            )
        )

        chunks = chunker.chunk_document(
            document,
            sections,
        )

        all_chunks.extend(chunks)

    bm25_retriever = BM25Retriever(
        all_chunks
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    question = (
        "What experience does the candidate "
        "have with machine learning and "
        "risk prediction?"
    )

    query_vector = (
        embedding_model.embed_text(
            question
        )
    )

    results = hybrid_retriever.search(
        query_vector=query_vector,
        query=question,
        limit=5,
        metadata_filter={
            "section_title": "EXPERIENCE"
        },
    )

    print()
    print("QUERY")
    print("-----")
    print(question)

    print()
    print("RESULTS")
    print("-------")

    print(
        f"Chunks searched : {len(all_chunks)}"
    )

    print(
        f"Results returned : {len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.metadata

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
            f"RRF Score: "
            f"{result.score}"
        )

        print(
            f"Content:\n"
            f"{result.content[:500]}"
        )

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()