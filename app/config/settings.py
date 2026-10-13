import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIRECTORY = PROJECT_ROOT / "models"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(LOCAL_MODEL_DIRECTORY / "all-MiniLM-L6-v2"),
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    str(LOCAL_MODEL_DIRECTORY / "ms-marco-MiniLM-L-6-v2"),
)
