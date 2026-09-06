from app.models.document import Document
from app.models.chunk import DocumentChunk


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

    def chunk_document(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        text = document.content

        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk = DocumentChunk(
                    chunk_id=(
                        f"{document.document_id}_"
                        f"{document.metadata.get('page_number', 0)}_"
                        f"{chunk_index}"
                    ),

                    document_id=document.document_id,

                    content=chunk_text,

                    metadata={
                        **document.metadata,

                        "source": document.source,

                        "source_type": document.source_type,

                        "chunk_index": chunk_index,

                        "chunk_size": len(chunk_text),
                    },
                )

                chunks.append(chunk)

                chunk_index += 1

            start += self.chunk_size - self.overlap

        return chunks