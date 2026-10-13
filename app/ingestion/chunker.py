import re
from uuid import NAMESPACE_URL, uuid5

from app.models.document import Document, DocumentChunk, DocumentSection


class TextChunker:
    """Section-aware chunker preserving section and page context."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, min_chunk_size: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document, sections: list[DocumentSection] | None = None) -> list[DocumentChunk]:
        if sections is None:
            return self._chunk_legacy_document(document)
        chunks = []
        for section in sections:
            chunks.extend(self._chunk_section(document, section))
        return self._reindex_chunks(chunks)

    def _chunk_section(self, document: Document, section: DocumentSection) -> list[DocumentChunk]:
        text = self.clean_text(section.content)
        pieces = self._split_text(text)
        chunks = []
        for index, piece in enumerate(pieces):
            chunk_id = self._generate_chunk_id(document.document_id, section.section_id, index, piece)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    content=piece,
                    metadata={
                        "chunk_id": chunk_id,
                        "source": document.source,
                        "source_type": document.source_type,
                        "section_id": section.section_id,
                        "section_title": section.title,
                        "page_number": section.page_number,
                        "chunk_index": index,
                        "content_length": len(piece),
                    },
                )
            )
        return chunks

    def _chunk_legacy_document(self, document: Document) -> list[DocumentChunk]:
        text = self.clean_text(document.content)
        pieces = self._split_text(text)
        return [
            DocumentChunk(
                chunk_id=str(uuid5(NAMESPACE_URL, f"{document.document_id}:{i}:{piece}")),
                document_id=document.document_id,
                content=piece,
                metadata={
                    "chunk_id": str(uuid5(NAMESPACE_URL, f"{document.document_id}:{i}:{piece}")),
                    "source": document.source,
                    "source_type": document.source_type,
                    "page_number": document.metadata.get("page_number"),
                    "chunk_index": i,
                },
            )
            for i, piece in enumerate(pieces)
        ]

    def _reindex_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        for index, chunk in enumerate(chunks):
            chunk.metadata["global_chunk_index"] = index
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + self.min_chunk_size:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            next_start = max(end - self.chunk_overlap, start + 1)
            start = next_start
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _generate_chunk_id(document_id: str, section_id: str, index: int, content: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{document_id}:{section_id}:{index}:{content}"))
