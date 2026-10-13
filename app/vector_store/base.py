from abc import ABC, abstractmethod
from typing import Any

from app.models.document import DocumentChunk


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], limit: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[Any]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_collection(self) -> None: ...
