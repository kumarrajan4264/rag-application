from pathlib import Path

from sentence_transformers import CrossEncoder

from app.config.settings import RERANKER_MODEL
from app.models.search import SearchResult


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        model_path = Path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local reranker model not found: {model_path}. "
                "Copy the downloaded model into models/ms-marco-MiniLM-L-6-v2 "
                "or set RERANKER_MODEL to a local model directory."
            )

        self.model = CrossEncoder(
            str(model_path),
            local_files_only=True,
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, result.content) for result in results]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchResult(
                content=result.content,
                metadata=result.metadata,
                distance=result.distance,
                score=float(score),
            )
            for result, score in ranked[:top_k]
        ]
