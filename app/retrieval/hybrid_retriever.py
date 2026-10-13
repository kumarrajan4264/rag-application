from app.models.search import SearchResult
from app.retrieval.bm25_retriever import BM25Retriever
from app.vector_store.chroma_store import ChromaVectorStore


class HybridRetriever:
    def __init__(self, vector_store: ChromaVectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(self, query_vector: list[float], query: str, limit: int = 5, rrf_k: int = 60, metadata_filter: dict | None = None) -> list[SearchResult]:
        dense_results = self.vector_store.search(query_vector, limit, metadata_filter)
        keyword_results = self.bm25_retriever.search(query, limit, metadata_filter)
        result_map = {}
        rrf_scores = {}
        for rank, result in enumerate(dense_results, start=1):
            key = self._result_key(result)
            result_map[key] = result
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        for rank, result in enumerate(keyword_results, start=1):
            key = self._result_key(result)
            result_map[key] = result
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        ranked_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [
            SearchResult(
                content=result_map[key].content,
                metadata=result_map[key].metadata,
                distance=result_map[key].distance,
                score=rrf_scores[key],
            )
            for key in ranked_keys[:limit]
        ]

    @staticmethod
    def _result_key(result: SearchResult) -> str:
        return str(result.metadata.get("chunk_id") or result.content)
