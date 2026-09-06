from app.models.document import Document
from app.models.chunk import DocumentChunk


document = Document(
    document_id="DOC_001",
    source="samsung_ml_guide.pdf",
    source_type="pdf",
    content="Samsung machine learning deployment guide.",
    metadata={
        "page_number": 1,
        "section": "Introduction",
    },
)


chunk = DocumentChunk(
    chunk_id="DOC_001_001",
    document_id=document.document_id,
    content="Machine learning models should be deployed...",
    metadata={
        "page_number": 1,
        "section": "Introduction",
    },
)


print("DOCUMENT")
print(document.model_dump())

print("\nCHUNK")
print(chunk.model_dump())