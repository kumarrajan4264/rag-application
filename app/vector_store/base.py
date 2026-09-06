from abc import ABC, abstractmethod
from typing import Any

from app.models.document import DocumentChunk


class VectorStore(ABC):
    """
    Abstract interface for vector databases.

    The RAG application depends on this interface,
    not on a specific vector database implementation.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        pass