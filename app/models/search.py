from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Database-independent representation of a
    retrieved document chunk.
    """

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    distance: float | None = None

    score: float | None = None