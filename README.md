# RAG v3

Production-grade local RAG pipeline with parent-child chunking, Pinecone local vector store, hybrid retrieval, and cross-encoder reranking.

## Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `nomic-embed-text` via Ollama (768-dim, local) |
| LLM | `llama3.2:3b` via Ollama (local) |
| Vector DB | Pinecone local (Docker, gRPC) |
| Retrieval | Hybrid: Pinecone ANN + persistent BM25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Chunking | Parent-child (512 child / 1536 parent) |
| Cache | Disk-based query cache with TTL |

## How it works

1. PDFs are loaded, cleaned, and split into **parent chunks** (1536 chars) and **child chunks** (512 chars)
2. Child chunks are embedded and indexed in Pinecone for high-precision search
3. Parent text is stored as metadata alongside each child vector
4. At query time, hybrid retrieval runs Pinecone ANN + BM25 in parallel
5. Candidates are reranked by a cross-encoder; results below the score threshold are dropped
6. Top child hits are expanded back to their parent chunks for richer LLM context
7. The LLM answers strictly from context; refuses if no relevant context is found

## Setup

### Automated (Ubuntu Server)

```bash
bash setup.sh
```

### Manual

**1. Start Pinecone local**
```bash
docker run -d \
  --name pinecone-local \
  --restart unless-stopped \
  -p 5081:5081 \
  ghcr.io/pinecone-io/pinecone-local:latest
```

**2. Pull Ollama models**
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure**
```bash
cp config.toml.example config.toml
# Edit thresholds as needed
```

**5. Ingest**
```bash
python populate_database.py
# Wipe and re-ingest: python populate_database.py --reset
```

## Usage

```bash
python bot.py                                    # interactive
python query_data.py "your question here"        # single query
pytest test_rag.py -v                            # run evals
```

## Key config knobs

```toml
[retrieval]
rerank_score_threshold = 0.15   # drop chunks below this score
min_rerank_score = 0.20         # refuse to answer if best chunk is below this

[ingestion]
child_chunk_size = 512          # search precision
parent_chunk_size = 1536        # LLM context window
```

Tune `min_rerank_score` upward if you're getting hallucinations, downward if you're getting too many refusals.
