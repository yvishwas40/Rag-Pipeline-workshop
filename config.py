"""Load `.env` and hold settings. Keys are read by other scripts."""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-workshop-demo").strip()
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1").strip()

# Model ids for google-genai (with or without "models/" prefix — we strip it in rag_pipeline).
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash").strip()

# Size of each embedding vector — must match your Pinecone index.
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))

# How we cut long text into smaller pieces before embedding.
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))
DOCUMENT_PATH = os.environ.get("DOCUMENT_PATH", "data/insurance.pdf").strip()
# How many text chunks we fetch for each question.
TOP_K = int(os.environ.get("TOP_K", "4"))


def check_keys():
    """Stop with a clear message if .env is missing keys."""
    if not GEMINI_API_KEY:
        raise SystemExit("Missing GEMINI_API_KEY in .env")
    if not PINECONE_API_KEY:
        raise SystemExit("Missing PINECONE_API_KEY in .env")
