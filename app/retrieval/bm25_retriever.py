from rank_bm25 import BM25Okapi

from app.models.document import DocumentChunk
from app.models.search import SearchResult


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")
        self.chunks = chunks
        self.bm25 = BM25Okapi([self._tokenize(chunk.content) for chunk in chunks])

    def search(self, query: str, limit: int = 5, metadata_filter: dict | None = None) -> list[SearchResult]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for index in ranked_indexes:
            chunk = self.chunks[index]
            if metadata_filter and not all(chunk.metadata.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append(SearchResult(content=chunk.content, metadata=chunk.metadata, score=float(scores[index])))
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
