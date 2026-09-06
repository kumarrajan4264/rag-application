from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.document_service import DocumentIngestionService


def main():

    print("=" * 70)
    print("INGESTION PIPELINE TEST")
    print("=" * 70)

    pipeline = DocumentIngestionService()

    result = pipeline.ingest_pdf(
        "data/raw/samsung_guide.pdf"
    )

    print()
    print("INGESTION RESULT")
    print("----------------")

    print(
        f"Documents processed : "
        f"{result['documents']}"
    )

    print(
        f"Chunks created      : "
        f"{result['chunks']}"
    )

    print(
        f"Vectors in Chroma   : "
        f"{result['vectors']}"
    )

    print()
    print("=" * 70)
    print("INGESTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()