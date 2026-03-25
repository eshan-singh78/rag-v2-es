# RAG v2

A production-grade local RAG pipeline using Ollama, pgvector, hybrid search, and cross-encoder reranking.

## Stack

- Embeddings: `nomic-embed-text` via Ollama (local)
- LLM: `llama3.2:3b` via Ollama (local)
- Vector DB: PostgreSQL + pgvector (self-hosted)
- Retrieval: Hybrid BM25 + MMR vector search
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- Cache: Disk-based query cache with TTL

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Docker (for Postgres)

## Setup

### Ubuntu Server (automated)

```bash
bash setup.sh
```

Installs Docker, Ollama, Python, spins up pgvector, pulls models, creates venv, and ingests any PDFs in `data/`.

---

### Manual setup

### 1. Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 2. Start pgvector

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

Copy the example config and edit as needed:

```bash
cp config.toml.example config.toml
```

Edit `config.toml`:

```toml
[database]
url = "postgresql+psycopg://postgres:password@localhost:5432/ragdb"

[embedding]
model = "nomic-embed-text"

[llm]
model = "llama3.2:3b"

[retrieval]
vector_fetch_k = 20
bm25_top_k = 10
rerank_top_n = 5

[ingestion]
batch_size = 10
max_retries = 3
data_path = "data"

[cache]
ttl_seconds = 3600
```

### 5. Add your documents

Drop PDF files into the `data/` folder.

### 6. Ingest

```bash
python populate_database.py
```

To wipe and re-ingest from scratch:

```bash
python populate_database.py --reset
```

## Usage

### Interactive bot

```bash
python bot.py
```

Type your questions and get answers. Type `exit` or `quit` to stop.

### Single query (CLI)

```bash
python query_data.py "How much money does a player start with in Monopoly?"
```

### Run tests

```bash
pytest test_rag.py -v
```

## Project Structure

```
├── bot.py                  # Interactive chat bot
├── config.toml             # Your local config (gitignored)
├── config.toml.example     # Config template to commit
├── config.py               # Config loader
├── db.py                   # pgvector connection
├── get_embedding_function.py
├── populate_database.py    # Ingest PDFs into pgvector
├── query_data.py           # Core RAG query pipeline
├── reranker.py             # Cross-encoder reranking
├── cache.py                # Disk query cache
├── logger.py               # Structured JSON logging
├── test_rag.py             # LLM-as-judge eval tests
└── data/                   # Drop your PDFs here
```

## How it works

1. PDFs are loaded, cleaned, and split into 1000-char chunks with 150-char overlap
2. Each chunk is embedded with `nomic-embed-text` and stored in pgvector
3. At query time, hybrid retrieval runs BM25 (keyword) + MMR vector search in parallel
4. Results are merged, deduplicated, and reranked by a cross-encoder
5. Top 5 chunks are passed as context to the LLM
6. Response and sources are returned and cached for 1 hour
