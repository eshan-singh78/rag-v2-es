from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from tqdm import tqdm

# Lightweight cross-encoder — runs fully local, no GPU needed
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[tuple[Document, float]]:
    """Score each doc against the query and return top_n sorted by relevance."""
    if not docs:
        return []
    model = _get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = []
    for pair in tqdm(pairs, desc="Reranking", unit="chunk", leave=False):
        scores.append(model.predict([pair])[0])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
