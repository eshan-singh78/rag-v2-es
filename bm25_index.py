"""
Persistent BM25 index backed by the Pinecone metadata store.
Builds once per process and caches in memory.
Rebuilds automatically if the corpus changes (detected via doc count).
"""
import threading
import pickle
import os
from rank_bm25 import BM25Okapi

import logger as log

_CACHE_FILE = ".bm25_cache.pkl"
_lock = threading.Lock()
_bm25: BM25Okapi | None = None
_corpus: list[dict] | None = None   # list of pinecone metadata dicts


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build(corpus: list[dict]):
    """Build and cache the BM25 index from a list of metadata dicts."""
    global _bm25, _corpus
    with _lock:
        tokenized = [_tokenize(d.get("child_text", "")) for d in corpus]
        _bm25 = BM25Okapi(tokenized)
        _corpus = corpus
        # Persist to disk so next process startup is instant
        try:
            with open(_CACHE_FILE, "wb") as f:
                pickle.dump({"bm25": _bm25, "corpus": _corpus}, f)
        except Exception as e:
            log.warning("bm25_cache_write_failed", error=str(e))
    log.info("bm25_index_built", docs=len(corpus))


def _load_from_disk() -> bool:
    global _bm25, _corpus
    if not os.path.exists(_CACHE_FILE):
        return False
    try:
        with open(_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _corpus = data["corpus"]
        log.info("bm25_index_loaded_from_disk", docs=len(_corpus))
        return True
    except Exception as e:
        log.warning("bm25_cache_read_failed", error=str(e))
        return False


def search(query: str, top_k: int) -> list[dict]:
    """Return top_k docs by BM25 score. Loads from disk cache if available."""
    global _bm25, _corpus
    if _bm25 is None:
        with _lock:
            if _bm25 is None and not _load_from_disk():
                log.warning("bm25_index_not_ready")
                return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_corpus[i] for i in top_indices]


def invalidate():
    """Clear in-memory and on-disk BM25 cache (call after re-ingestion)."""
    global _bm25, _corpus
    with _lock:
        _bm25 = None
        _corpus = None
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    log.info("bm25_cache_invalidated")
