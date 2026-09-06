from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.section_extractor import SectionExtractor
from app.ingestion.chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vector_store.chroma_store import ChromaVectorStore


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    PDF
    -> Document
    -> Sections
    -> Chunks
    -> Embeddings
    -> Vector Store
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):

        self.pdf_loader = PDFLoader()

        self.section_extractor = SectionExtractor()

        self.chunker = TextChunker(
            chunk_size=800,
            chunk_overlap=150,
        )

        self.embedding_model = EmbeddingModel()

        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def ingest_pdf(
        self,
        file_path: str,
    ):

        # --------------------------------------------------
        # 1. Load PDF
        # --------------------------------------------------

        documents = self.pdf_loader.load(
            file_path
        )

        # --------------------------------------------------
        # 2. Extract sections
        # --------------------------------------------------

        all_chunks = []

        for document in documents:

            sections = self.section_extractor.extract_sections(
                document
            )

            # --------------------------------------------------
            # 3. Create chunks
            # --------------------------------------------------

            chunks = self.chunker.chunk_document(
                document,
                sections,
            )

            # --------------------------------------------------
            # 4. Generate embeddings
            # --------------------------------------------------

            for chunk in chunks:

                chunk.embedding = (
                    self.embedding_model.embed_text(
                        chunk.content
                    )
                )

            all_chunks.extend(chunks)

        # --------------------------------------------------
        # 5. Store in ChromaDB
        # --------------------------------------------------

        self.vector_store.add_chunks(
            all_chunks
        )

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "vectors": self.vector_store.count(),
        }