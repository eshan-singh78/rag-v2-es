"""
Cross-encoder reranker — thread-safe singleton model load.
Uses ms-marco-MiniLM-L-6-v2: fast, local, no GPU required.
"""
import threading
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_lock = threading.Lock()
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(
    query: str,
    docs: list[dict],
    top_n: int = 5,
) -> list[tuple[dict, float]]:
    """
    Score each doc against the query.

    Args:
        docs: list of dicts with at least a 'parent_text' key (used for context)
              and optionally 'child_text' (used for scoring).
        top_n: how many to return.

    Returns:
        List of (doc, score) sorted descending by relevance.
    """
    if not docs:
        return []

    model = _get_model()
    # Score against child_text for precision; fall back to parent_text
    pairs = [(query, d.get("child_text") or d.get("parent_text", "")) for d in docs]
    scores = model.predict(pairs).tolist()

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
