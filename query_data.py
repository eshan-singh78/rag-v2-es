"""
RAG query pipeline:
  1. Hybrid retrieval — Qdrant ANN vector search + BM25 keyword search
  2. Parent-document expansion — swap child chunks for their parent context
  3. Cross-encoder reranking (skippable via config)
  4. Score-gated answer generation (refuses if context is not relevant enough)
  5. Coverage check — refuses if key query terms are absent from context
"""
import argparse
import re
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

_NO_CONTEXT_RESPONSE = "No relevant SEBI regulation found in documents."

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
You MUST ONLY answer using the context provided below.
If the answer is not clearly present in the context, say exactly:
"Information not found in SEBI documents."
DO NOT use prior knowledge. DO NOT guess. DO NOT infer beyond what is written.

Provide a thorough, well-structured answer. Include:
- Specific figures, thresholds, or conditions mentioned in the context
- Any relevant distinctions (e.g. individual vs non-individual, different entity types)
- The regulatory basis or rule name where present in the context
Use plain paragraphs or bullet points as appropriate. Do not truncate your answer.

Context:
{context}

---

Question: {question}

Answer:""")


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 2000:
        raise ValueError("Query exceeds maximum length of 2000 characters.")
    return query


# ── Coverage check ────────────────────────────────────────────────────────────
_STOPWORDS = {"the", "a", "an", "of", "in", "for", "and", "or", "is", "are",
              "to", "be", "what", "how", "why", "which", "does", "do", "can",
              "me", "my", "i", "you", "it", "its", "this", "that", "with"}


def _has_coverage(query: str, context: str) -> bool:
    """
    Return True if at least half of the meaningful query terms appear in context.
    Prevents the LLM from being called when context is topically unrelated.
    """
    tokens = re.findall(r'\b[a-z]{3,}\b', query.lower())
    key_terms = [t for t in tokens if t not in _STOPWORDS]
    if not key_terms:
        return True  # can't determine — let it through
    context_lower = context.lower()
    hits = sum(1 for t in key_terms if t in context_lower)
    coverage = hits / len(key_terms)
    log.info("coverage_check", key_terms=key_terms, hits=hits, coverage=round(coverage, 2))
    return coverage >= 0.5


# ── Retrieval ─────────────────────────────────────────────────────────────────
def _vector_search(query: str, k: int) -> list[dict]:
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
        print(_NO_CONTEXT_RESPONSE)
        return _NO_CONTEXT_RESPONSE

    # Rerank (or pass through if skip_reranker is set)
    if config.skip_reranker:
        ranked = [(doc, 1.0) for doc in candidates[:config.rerank_top_n * 2]]
        log.info("reranker_skipped")
    else:
        with log.timer("reranking", candidates=len(candidates)):
            ranked = rerank(query_text, candidates, top_n=config.rerank_top_n * 2)

    # Score threshold filter
    ranked = [(doc, score) for doc, score in ranked if score >= config.rerank_score_threshold]
    if not ranked:
        log.warning("rerank_below_threshold", threshold=config.rerank_score_threshold)
        print(_NO_CONTEXT_RESPONSE)
        return _NO_CONTEXT_RESPONSE

    # Debug: log top chunks and scores
    print(f"\n[DEBUG] Query: {query_text[:120]}")
    for i, (doc, score) in enumerate(ranked[:config.rerank_top_n]):
        chunk_preview = (doc.get("child_text") or "")[:300]
        print(f"[DEBUG] TOP CHUNK {i+1}: {chunk_preview}")
        print(f"[DEBUG] SCORE {i+1}: {round(float(score), 4)}")

    log.info("rerank_scores", scores=[round(float(s), 3) for _, s in ranked])

    # Hard floor — refuse if best chunk isn't relevant enough
    top_score = float(ranked[0][1])
    if top_score < config.min_rerank_score:
        log.warning("min_rerank_score_not_met", top_score=top_score,
                    threshold=config.min_rerank_score)
        cache.set(query_text, {"response": _NO_CONTEXT_RESPONSE, "sources": []})
        print(f"\n{_NO_CONTEXT_RESPONSE}")
        return _NO_CONTEXT_RESPONSE

    # Take top N, expand to parents
    top_docs = [doc for doc, _ in ranked[:config.rerank_top_n]]
    expanded = _expand_to_parents(top_docs)

    # Build context — truncate each parent chunk to ~1200 chars (enough for a full regulatory clause)
    _CHUNK_LIMIT = 1200
    context_parts = []
    for d in expanded:
        text = d.get("parent_text") or d.get("child_text", "")
        context_parts.append(text[:_CHUNK_LIMIT])
    context_text = "\n\n---\n\n".join(context_parts)

    # Coverage check — don't call LLM if context is topically unrelated
    if not _has_coverage(query_text, context_text):
        log.warning("coverage_check_failed", query=query_text[:60])
        cache.set(query_text, {"response": _NO_CONTEXT_RESPONSE, "sources": []})
        print(f"\n{_NO_CONTEXT_RESPONSE}")
        return _NO_CONTEXT_RESPONSE

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
