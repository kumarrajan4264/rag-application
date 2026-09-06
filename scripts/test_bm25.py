from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.ingestion.chunker import TextChunker
from app.retrieval.bm25_retriever import BM25Retriever


def main():

    print("=" * 70)
    print("BM25 RETRIEVAL TEST")
    print("=" * 70)

    pdf_loader = PDFLoader()
    section_extractor = SectionExtractor()
    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
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

    retriever = BM25Retriever(
        all_chunks
    )

    question = (
        "What experience does the candidate "
        "have with machine learning and "
        "risk prediction?"
    )

    results = retriever.search(
        query=question,
        limit=5,
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

    for index, chunk in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"[{index}] "
            f"Section: "
            f"{chunk.metadata.get('section_title')}"
        )

        print(
            f"Page: "
            f"{chunk.metadata.get('page_number')}"
        )

        print(
            f"Content length: "
            f"{len(chunk.content)}"
        )

        print(
            f"Content:\n"
            f"{chunk.content[:500]}"
        )

    print()
    print("=" * 70)
    print("BM25 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()