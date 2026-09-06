import os

from dotenv import load_dotenv


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")