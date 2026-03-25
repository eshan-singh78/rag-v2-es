import argparse
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from tqdm import tqdm

from db import get_db
from reranker import rerank
import cache
import logger as log

import config

LLM_MODEL = config.llm_model
VECTOR_FETCH_K = config.vector_fetch_k
BM25_TOP_K = config.bm25_top_k
RERANK_TOP_N = config.rerank_top_n

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

---

Question: {question}

Answer concisely and accurately:"""


def _bm25_search(query: str, all_docs: list[Document], top_k: int) -> list[Document]:
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [all_docs[i] for i in top_indices]


def _hybrid_retrieve(query: str) -> list[Document]:
    db = get_db()

    steps = ["Vector retrieval", "BM25 retrieval", "Merging results"]
    with tqdm(steps, desc="Retrieving", unit="step", leave=False) as pbar:

        pbar.set_description("Vector retrieval")
        with log.timer("vector_retrieval", query=query[:60]):
            vector_docs = db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": VECTOR_FETCH_K, "fetch_k": 40, "lambda_mult": 0.7},
            ).invoke(query)
        pbar.update(1)

        pbar.set_description("BM25 retrieval")
        with log.timer("bm25_retrieval"):
            all_raw = db.similarity_search("", k=1000)
            bm25_docs = _bm25_search(query, all_raw, top_k=BM25_TOP_K)
        pbar.update(1)

        pbar.set_description("Merging results")
        seen: set[str] = set()
        merged: list[Document] = []
        for doc in vector_docs + bm25_docs:
            doc_id = doc.metadata.get("id")
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
        pbar.update(1)

    log.info("hybrid_candidates", vector=len(vector_docs), bm25=len(bm25_docs), merged=len(merged))
    return merged


def query_rag(query_text: str) -> str | None:
    cached = cache.get(query_text)
    if cached:
        log.info("cache_hit", query=query_text[:60])
        print(f"\n[cached] Response: {cached['response']}")
        print(f"Sources:  {cached['sources']}")
        return cached["response"]

    candidates = _hybrid_retrieve(query_text)
    if not candidates:
        log.warning("no_candidates", query=query_text[:60])
        print("⚠️  No relevant context found.")
        return None

    with log.timer("reranking", candidates=len(candidates)):
        ranked = rerank(query_text, candidates, top_n=RERANK_TOP_N)

    if not ranked:
        log.warning("rerank_empty")
        return None

    log.info("rerank_scores", scores=[round(float(s), 3) for _, s in ranked])

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in ranked])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    with log.timer("llm_inference", model=LLM_MODEL):
        model = OllamaLLM(model=LLM_MODEL)
        response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id") for doc, _ in ranked]
    rerank_scores = [round(float(s), 3) for _, s in ranked]

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
