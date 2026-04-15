"""
Helpers to read a file and split it into chunks.

Chunk = one small piece of text. We embed each chunk so search can find
the right paragraph later.
"""

from pathlib import Path


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def load_document(path: str) -> str:
    """Return the full text of a .txt or .pdf file."""
    ending = Path(path).suffix.lower()
    if ending == ".txt":
        return read_text_file(path)
    if ending == ".pdf":
        return read_pdf_text(path)
    raise ValueError("Use a .txt or .pdf file only.")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split long text into overlapping chunks (all plain strings)."""
    text = text.strip()
    if not text or chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return chunks
