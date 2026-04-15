"""
RAG in four steps:
1) Embed the user question
2) Search Pinecone for similar chunks
3) Put those chunks + question into a prompt
4) Call Gemini to answer
"""

from google import genai
from google.genai import types
from pinecone import Pinecone

import config


def clean_model_name(name: str) -> str:
    return name.strip().removeprefix("models/")


def embed(text: str, *, for_query: bool) -> list[float]:
    """Turn one string into a list of floats (the embedding vector)."""
    config.check_keys()
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    if for_query:
        task = "RETRIEVAL_QUERY"
    else:
        task = "RETRIEVAL_DOCUMENT"

    model = clean_model_name(config.GEMINI_EMBEDDING_MODEL)
    try:
        reply = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=config.EMBEDDING_DIMENSION,
            ),
        )
    except Exception:
        reply = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task),
        )

    if not reply.embeddings or not reply.embeddings[0].values:
        raise RuntimeError("Gemini did not return an embedding. Check the model name and API key.")

    vec = [float(x) for x in reply.embeddings[0].values]
    if len(vec) != config.EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Got {len(vec)} numbers from Gemini but EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}."
        )
    return vec


def open_index():
    """Connect to Pinecone. You must run ingest.py once before this works."""
    config.check_keys()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    names = pc.list_indexes().names()
    if config.PINECONE_INDEX_NAME not in names:
        raise SystemExit(f"No index named {config.PINECONE_INDEX_NAME!r}. Run: python ingest.py")
    return pc.Index(config.PINECONE_INDEX_NAME)


def search(question: str, index, top_k: int):
    """Find the top_k chunks most similar to the question."""
    vector = embed(question, for_query=True)
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return list(result.matches or [])


def get_chunk_texts(matches):
    """From Pinecone results, pull out the saved text of each chunk."""
    texts = []
    for m in matches:
        meta = m.metadata if isinstance(m.metadata, dict) else {}
        chunk = meta.get("text", "")
        if isinstance(chunk, str) and chunk.strip():
            texts.append(chunk.strip())
    return texts


def build_prompt(question: str, chunk_texts: list[str]) -> str:
    """One string we send to Gemini: context + question."""
    if chunk_texts:
        context = "\n\n".join(f"[{i}]\n{t}" for i, t in enumerate(chunk_texts, start=1))
    else:
        context = "(No matching chunks were found.)"

    return (
        "Use only the excerpts below. Answer in plain words only.\n"
        "If the answer is not there, say: I cannot find that in the document.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question.strip()}"
    )


def ask_gemini(prompt: str) -> str:
    """Send the prompt to Gemini and return the answer text."""
    config.check_keys()
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    model = clean_model_name(config.GEMINI_CHAT_MODEL)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except Exception as e:
        return f"Request failed: {e}"
    return (response.text or "(No reply.)").strip()
