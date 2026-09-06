from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    source: str
    source_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class DocumentSection(BaseModel):
    section_id: str
    document_id: str
    title: str
    content: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)