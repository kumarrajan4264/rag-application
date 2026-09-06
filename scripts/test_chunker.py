from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.ingestion.chunker import TextChunker


def main():

    pdf_path = "data/raw/samsung_guide.pdf"

    loader = PDFLoader()
    extractor = SectionExtractor()
    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
    )

    documents = loader.load(pdf_path)

    all_chunks = []

    for document in documents:

        sections = extractor.extract_sections(
            document
        )

        chunks = chunker.chunk_document(
            document=document,
            sections=sections,
        )

        all_chunks.extend(chunks)

    print("=" * 70)
    print("SECTION-AWARE CHUNK QUALITY REPORT")
    print("=" * 70)

    print(f"Documents        : {len(documents)}")
    print(f"Sections         : {len(sections)}")
    print(f"Total chunks     : {len(all_chunks)}")

    if all_chunks:

        sizes = [
            len(chunk.content)
            for chunk in all_chunks
        ]

        print(
            f"Average size     : {sum(sizes) / len(sizes):.2f}"
        )
        print(
            f"Minimum size     : {min(sizes)}"
        )
        print(
            f"Maximum size     : {max(sizes)}"
        )

    print(
        f"Chunk size       : {chunker.chunk_size}"
    )
    print(
        f"Chunk overlap    : {chunker.chunk_overlap}"
    )

    print("=" * 70)

    for index, chunk in enumerate(all_chunks):

        print()
        print("=" * 70)

        print(f"Chunk ID        : {chunk.chunk_id}")
        print(f"Global Index    : {index}")
        print(
            f"Section         : "
            f"{chunk.metadata.get('section_title')}"
        )
        print(
            f"Section ID      : "
            f"{chunk.metadata.get('section_id')}"
        )
        print(
            f"Page            : "
            f"{chunk.metadata.get('page_number')}"
        )
        print(
            f"Size            : "
            f"{len(chunk.content)}"
        )

        print("-" * 70)
        print(chunk.content)


if __name__ == "__main__":
    main()