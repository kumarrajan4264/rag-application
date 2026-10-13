from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        model_path = Path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local embedding model not found: {model_path}. "
                "Copy the downloaded model into models/all-MiniLM-L6-v2 "
                "or set EMBEDDING_MODEL to a local model directory."
            )

        self.model = SentenceTransformer(
            str(model_path),
            local_files_only=True,
        )

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()
