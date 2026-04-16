"""Chunk documents, embed with Gemini, upsert into Pinecone. Supports .txt and .pdf."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

import config
import rag_pipeline
from utils import chunk_text, load_document

# Folder where ingest.py lives (so relative DOCUMENT_PATH like "data/file.pdf" works).
_PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # ----- Step 1: find the file from config -----
    path = Path(config.DOCUMENT_PATH).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path = path.resolve()

    if not path.is_file():
        raise SystemExit(f"DOCUMENT_PATH is not a file: {path}")
    if path.suffix.lower() not in (".pdf", ".txt"):
        raise SystemExit("DOCUMENT_PATH must be .pdf or .txt")

    print(f"Reading: {path.name}")
    text = load_document(str(path))
    if not text.strip():
        raise SystemExit("No text came out of this file (empty PDF?).")

    chunks = chunk_text(text, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS)
    if not chunks:
        raise SystemExit("No chunks were created.")
    print(f"Split into {len(chunks)} chunks.")

    # ----- Step 2: connect to Pinecone; create empty index if needed -----
    config.check_keys()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = set(pc.list_indexes().names())

    if config.PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index {config.PINECONE_INDEX_NAME!r}…")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while True:
            status = pc.describe_index(config.PINECONE_INDEX_NAME).status
            ready = getattr(status, "ready", None) if not isinstance(status, dict) else status.get("ready")
            if ready is True:
                break
            time.sleep(1)

    index = pc.Index(config.PINECONE_INDEX_NAME)

    # ----- Step 3: embed each chunk and upload -----
    print("Embedding and uploading…")
    batch = []
    for chunk in tqdm(chunks, desc="Chunks"):
        vec = rag_pipeline.embed(chunk, for_query=False)
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {"text": chunk, "source": path.name},
            }
        )

    step = 100
    for i in tqdm(range(0, len(batch), step), desc="Upload"):
        index.upsert(vectors=batch[i : i + step])

    print(f"Done. Indexed {len(batch)} chunks. Next: python main.py")


if __name__ == "__main__":
    main()
