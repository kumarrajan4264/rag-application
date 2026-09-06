from pathlib import Path
from typing import Any
from app.models.search import SearchResult

import chromadb

from app.models.document import DocumentChunk
from app.vector_store.base import VectorStore


# class VectorSearchResult:
#     """
#     Database-independent vector search result.

#     Provides a common interface for the RAG pipeline
#     regardless of the underlying vector database.
#     """

#     def __init__(
#         self,
#         content: str,
#         metadata: dict[str, Any],
#         distance: float | None = None,
#     ):
#         self.payload = {
#             "content": content,
#             "metadata": metadata,
#         }

#         self.distance = distance

#         # Chroma returns distance while the existing
#         # RAG pipeline expects a score.
#         self.score = (
#             1.0 / (1.0 + distance)
#             if distance is not None
#             else 0.0
#         )


class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):

        self.persist_directory = Path(
            persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "Local RAG document collection"
                    )
                },
            )
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:

        if not chunks:
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:

            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} "
                    "does not contain an embedding."
                )

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            embeddings.append(chunk.embedding)

            metadata = self._prepare_metadata(
                chunk.metadata
            )

            metadata["document_id"] = (
                chunk.document_id
            )

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:

        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        results = self.collection.query(
            **query_kwargs
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        search_results = []

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            search_results.append(
                SearchResult(
                    content=document,
                    metadata=metadata,
                    distance=distance,
                    score=(
                        1.0 / (1.0 + distance)
                        if distance is not None
                        else 0.0
                    ),
                )
            )

        return search_results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(
            self.collection.name
        )

    def _prepare_metadata(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:

        clean_metadata = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)

        return clean_metadata