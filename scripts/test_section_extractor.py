from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor


def main():

    pdf_path = "data/raw/samsung_guide.pdf"

    loader = PDFLoader()

    documents = loader.load(pdf_path)

    extractor = SectionExtractor()

    print("=" * 70)
    print("SECTION EXTRACTION TEST")
    print("=" * 70)

    for document in documents:

        sections = extractor.extract_sections(
            document
        )

        print(f"Documents : {len(documents)}")
        print(f"Sections  : {len(sections)}")
        print("=" * 70)

        for index, section in enumerate(sections):

            print()
            print("=" * 70)

            print(
                f"Section {index}"
            )

            print(
                f"ID        : {section.section_id}"
            )

            print(
                f"Title     : {section.title}"
            )

            print(
                f"Page      : {section.page_number}"
            )

            print(
                f"Size      : {len(section.content)}"
            )

            print("-" * 70)

            print(section.content)


if __name__ == "__main__":
    main()