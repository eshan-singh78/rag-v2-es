"""
RAG query pipeline:
  1. Hybrid retrieval — Qdrant ANN vector search + BM25 keyword search
  2. Parent-document expansion — swap child chunks for their parent context
  3. Cross-encoder reranking
  4. Score-gated answer generation (refuses if context is not relevant enough)
"""
import argparse
import threading
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import config
import cache
import logger as log
import bm25_index
from db import get_client, COLLECTION
from get_embedding_function import embed_query
from reranker import rerank

# ── Singleton LLM ─────────────────────────────────────────────────────────────
_llm_lock = threading.Lock()
_llm: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = OllamaLLM(model=config.llm_model)
    return _llm


# ── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information to answer that."
Do not make up facts. Do not reference the context explicitly.

Context:
{context}

---

Question: {question}

Answer concisely and accurately:""")


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 2000:
        raise ValueError("Query exceeds maximum length of 2000 characters.")
    return query


# ── Retrieval ─────────────────────────────────────────────────────────────────
def _vector_search(query: str, k: int) -> list[dict]:
    """ANN search in Qdrant, returns list of payload dicts."""
    client = get_client()
    vec = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=k,
        with_payload=True,
    ).points
    return [hit.payload for hit in results if hit.payload]


def _hybrid_retrieve(query: str) -> list[dict]:
    """Merge vector + BM25 candidates, deduplicated by child_id."""
    with log.timer("vector_retrieval"):
        vector_docs = _vector_search(query, config.vector_fetch_k)

    with log.timer("bm25_retrieval"):
        bm25_docs = bm25_index.search(query, top_k=config.bm25_top_k)

    seen: set[str] = set()
    merged: list[dict] = []
    for doc in vector_docs + bm25_docs:
        key = doc.get("child_id") or doc.get("child_text", "")[:64]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    log.info("hybrid_candidates",
             vector=len(vector_docs),
             bm25=len(bm25_docs),
             merged=len(merged))
    return merged


def _expand_to_parents(docs: list[dict]) -> list[dict]:
    """
    Deduplicate by parent_id so the LLM sees full parent chunks,
    not multiple overlapping child fragments from the same parent.
    """
    seen_parents: set[str] = set()
    expanded: list[dict] = []
    for doc in docs:
        pid = doc.get("parent_id", doc.get("child_id", ""))
        if pid not in seen_parents:
            seen_parents.add(pid)
            expanded.append(doc)
    return expanded


# ── Main query function ───────────────────────────────────────────────────────
def query_rag(query_text: str) -> str | None:
    try:
        query_text = _validate_query(query_text)
    except ValueError as e:
        log.warning("invalid_query", error=str(e))
        print(f"Invalid query: {e}")
        return None

    # Cache check
    cached = cache.get(query_text)
    if cached:
        log.info("cache_hit", query=query_text[:60])
        print(f"\n[cached] {cached['response']}")
        print(f"Sources: {cached['sources']}")
        return cached["response"]

    # Retrieval
    candidates = _hybrid_retrieve(query_text)
    if not candidates:
        log.warning("no_candidates", query=query_text[:60])
        print("No relevant context found.")
        return None

    # Rerank on child_text precision, then expand to parent context
    with log.timer("reranking", candidates=len(candidates)):
        ranked = rerank(query_text, candidates, top_n=config.rerank_top_n * 2)

    # Score threshold filter
    ranked = [(doc, score) for doc, score in ranked if score >= config.rerank_score_threshold]
    if not ranked:
        log.warning("rerank_below_threshold", threshold=config.rerank_score_threshold)
        print("No sufficiently relevant context found for this query.")
        return None

    log.info("rerank_scores", scores=[round(float(s), 3) for _, s in ranked])

    # Hard floor — refuse to answer if best chunk isn't relevant enough
    top_score = float(ranked[0][1])
    if top_score < config.min_rerank_score:
        response_text = "I don't have enough information to answer that."
        sources = [d.get("source") for d, _ in ranked]
        cache.set(query_text, {"response": response_text, "sources": sources})
        print(f"\n{response_text}")
        return response_text

    # Expand child hits to their parent chunks for richer context
    top_docs = [doc for doc, _ in ranked[:config.rerank_top_n]]
    expanded = _expand_to_parents(top_docs)

    # Build context from parent_text (wider window)
    context_text = "\n\n---\n\n".join(
        d.get("parent_text") or d.get("child_text", "") for d in expanded
    )

    # Generate answer
    prompt = _PROMPT.format(context=context_text, question=query_text)
    with log.timer("llm_inference", model=config.llm_model):
        response_text = _get_llm().invoke(prompt)

    sources = list({d.get("source", "unknown") for d in expanded})
    rerank_scores = [round(float(s), 3) for _, s in ranked[:config.rerank_top_n]]

    cache.set(query_text, {"response": response_text, "sources": sources})

    print(f"\nResponse: {response_text}")
    print(f"Sources:  {sources}")
    print(f"Rerank scores: {rerank_scores}")

    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
