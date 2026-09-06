from app.ingestion.pdf_loader import PDFLoader


PDF_PATH = "data/raw/samsung_guide.pdf"


loader = PDFLoader()

documents = loader.load(PDF_PATH)


print("=" * 70)
print("PDF LOADER TEST")
print("=" * 70)

print(f"Pages extracted: {len(documents)}")


for document in documents[:3]:

    print("\n" + "=" * 70)

    print(f"Document ID : {document.document_id}")
    print(f"Source      : {document.source}")
    print(f"Type        : {document.source_type}")

    print(
        f"Page        : "
        f"{document.metadata['page_number']}"
    )

    print("-" * 70)

    print(document.content[:1000])