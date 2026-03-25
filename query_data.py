import argparse
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function
from reranker import rerank
import cache
import logger as log

CHROMA_PATH = "chroma"
LLM_MODEL = "llama3.2:3b"
VECTOR_FETCH_K = 20       # candidates from vector search
BM25_TOP_K = 10           # candidates from BM25
RERANK_TOP_N = 5          # final chunks after reranking
RELEVANCE_THRESHOLD = 0.0 # reranker scores can be negative; keep top_n only

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


def _hybrid_retrieve(db: Chroma, query: str) -> list[Document]:
    # Vector retrieval (MMR for diversity)
    with log.timer("vector_retrieval", query=query[:60]):
        vector_docs = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": VECTOR_FETCH_K, "fetch_k": 40, "lambda_mult": 0.7},
        ).invoke(query)

    # BM25 over full corpus for keyword recall
    with log.timer("bm25_retrieval"):
        all_docs_raw = db.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs_raw["documents"], all_docs_raw["metadatas"])
        ]
        bm25_docs = _bm25_search(query, all_docs, top_k=BM25_TOP_K)

    # Merge and deduplicate by chunk id
    seen = set()
    merged = []
    for doc in vector_docs + bm25_docs:
        doc_id = doc.metadata.get("id")
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append(doc)

    log.info("hybrid_candidates", vector=len(vector_docs), bm25=len(bm25_docs), merged=len(merged))
    return merged


def query_rag(query_text: str) -> str | None:
    # Cache check
    cached = cache.get(query_text)
    if cached:
        log.info("cache_hit", query=query_text[:60])
        print(f"\n[cached] Response: {cached['response']}")
        print(f"Sources:  {cached['sources']}")
        return cached["response"]

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Hybrid retrieval
    candidates = _hybrid_retrieve(db, query_text)
    if not candidates:
        log.warning("no_candidates", query=query_text[:60])
        print("⚠️  No relevant context found.")
        return None

    # Cross-encoder reranking
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

    # Cache result
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
