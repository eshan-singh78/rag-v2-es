"""
Singleton embedding function backed by Ollama (local).
Thread-safe — safe to call from multiple workers.
"""
import threading
from langchain_ollama import OllamaEmbeddings
import config

_lock = threading.Lock()
_embedder: OllamaEmbeddings | None = None


def get_embedding_function() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = OllamaEmbeddings(model=config.embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, returns list of float vectors."""
    return get_embedding_function().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return get_embedding_function().embed_query(text)
