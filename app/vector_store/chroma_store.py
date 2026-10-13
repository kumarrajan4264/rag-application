from typing import Any

import chromadb

from app.models.document import DocumentChunk
from app.models.search import SearchResult
from app.vector_store.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "rag_documents"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[self._prepare_metadata(chunk.metadata) for chunk in chunks],
        )

    def search(self, query_vector: list[float], limit: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter
        result = self.collection.query(**kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchResult(
                content=document,
                metadata=metadata or {},
                distance=distance,
                score=1.0 / (1.0 + distance) if distance is not None else 0.0,
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def get_all_chunks(self) -> list[DocumentChunk]:
        data = self.collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        ids = data.get("ids") or []
        chunks = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            chunk_id = ids[index] if index < len(ids) else f"chunk-{index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", "unknown"),
                    content=document,
                    metadata=metadata,
                )
            )
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)

    @staticmethod
    def _prepare_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        prepared = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            else:
                prepared[key] = str(value)
        return prepared
