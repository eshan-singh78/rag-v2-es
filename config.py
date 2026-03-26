import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config.toml"
_EXAMPLE_PATH = _ROOT / "config.toml.example"

_path = _CONFIG_PATH if _CONFIG_PATH.exists() else _EXAMPLE_PATH

if not _path.exists():
    raise FileNotFoundError(
        "No config.toml found. Copy config.toml.example to config.toml and update your settings."
    )

with open(_path, "rb") as f:
    _cfg = tomllib.load(f)


def get(section: str, key: str):
    return _cfg[section][key]


def get_optional(section: str, key: str, default=None):
    return _cfg.get(section, {}).get(key, default)


# ── Database (Qdrant) ─────────────────────────────────────────────────────────
db_host: str = _cfg["database"]["host"]
db_grpc_port: int = _cfg["database"]["grpc_port"]
db_rest_port: int = _cfg["database"]["rest_port"]
db_collection_name: str = _cfg["database"]["collection_name"]
db_dimension: int = _cfg["database"]["dimension"]
db_metric: str = _cfg["database"]["metric"]

# ── Embedding ─────────────────────────────────────────────────────────────────
embedding_model: str = _cfg["embedding"]["model"]

# ── LLM ───────────────────────────────────────────────────────────────────────
llm_model: str = _cfg["llm"]["model"]

# ── Retrieval ─────────────────────────────────────────────────────────────────
vector_fetch_k: int = _cfg["retrieval"]["vector_fetch_k"]
bm25_top_k: int = _cfg["retrieval"]["bm25_top_k"]
rerank_top_n: int = _cfg["retrieval"]["rerank_top_n"]
rerank_score_threshold: float = float(_cfg["retrieval"]["rerank_score_threshold"])
min_rerank_score: float = float(_cfg["retrieval"]["min_rerank_score"])
skip_reranker: bool = bool(_cfg["retrieval"].get("skip_reranker", False))

# ── Ingestion ─────────────────────────────────────────────────────────────────
child_chunk_size: int = _cfg["ingestion"]["child_chunk_size"]
child_chunk_overlap: int = _cfg["ingestion"]["child_chunk_overlap"]
parent_chunk_size: int = _cfg["ingestion"]["parent_chunk_size"]
parent_chunk_overlap: int = _cfg["ingestion"]["parent_chunk_overlap"]
batch_size: int = _cfg["ingestion"]["batch_size"]
max_retries: int = _cfg["ingestion"]["max_retries"]
data_path: str = _cfg["ingestion"]["data_path"]
ingest_workers: int = _cfg["ingestion"].get("ingest_workers", 4)

# ── Cache ─────────────────────────────────────────────────────────────────────
cache_ttl: int = _cfg["cache"]["ttl_seconds"]
