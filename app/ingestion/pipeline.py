from app.embeddings.embedding_model import EmbeddingModel
from app.ingestion.chunker import TextChunker
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.vector_store.chroma_store import ChromaVectorStore


class DocumentIngestionService:
    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "rag_documents"):
        self.pdf_loader = PDFLoader()
        self.section_extractor = SectionExtractor()
        self.chunker = TextChunker(chunk_size=800, chunk_overlap=150)
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore(persist_directory, collection_name)

    def ingest_pdf(self, file_path: str):
        documents = self.pdf_loader.load(file_path)
        if not documents:
            raise ValueError(f"No extractable text found in PDF: {file_path}")
        all_chunks = []
        for document in documents:
            document_id = document.document_id
            document.metadata.update({"document_id": document_id, "file_path": file_path})
            sections = self.section_extractor.extract_sections(document)
            chunks = self.chunker.chunk_document(document, sections)
            for chunk in chunks:
                chunk.embedding = self.embedding_model.embed_text(chunk.content)
            all_chunks.extend(chunks)
        self.vector_store.add_chunks(all_chunks)
        return {
            "document_id": document_id,
            "file_name": file_path,
            "documents": len(documents),
            "chunks": len(all_chunks),
            "vectors": self.vector_store.count(),
        }
