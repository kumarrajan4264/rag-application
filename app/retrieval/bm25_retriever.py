from rank_bm25 import BM25Okapi

from app.models.document import DocumentChunk
from app.models.search import SearchResult


class BM25Retriever:
    """
    Keyword-based document retrieval using BM25.

    The retriever is intentionally independent of any
    specific PDF or ingestion pipeline.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ):
        if not chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk."
            )

        self.chunks = chunks

        tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indexes:

            chunk = self.chunks[index]

            if metadata_filter:
                matches_filter = all(
                    chunk.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )

                if not matches_filter:
                    continue

            results.append(
                SearchResult(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(scores[index]),
                )
            )

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()