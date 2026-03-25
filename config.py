import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config.toml"
_EXAMPLE_PATH = _ROOT / "config.toml.example"

# Use config.toml if present, fall back to config.toml.example
_path = _CONFIG_PATH if _CONFIG_PATH.exists() else _EXAMPLE_PATH

if not _path.exists():
    raise FileNotFoundError(
        "No config.toml found. Copy config.toml.example to config.toml and update your settings."
    )

with open(_path, "rb") as f:
    _cfg = tomllib.load(f)


def get(section: str, key: str):
    return _cfg[section][key]


# Convenience accessors
database_url: str = _cfg["database"]["url"]
embedding_model: str = _cfg["embedding"]["model"]
llm_model: str = _cfg["llm"]["model"]
vector_fetch_k: int = _cfg["retrieval"]["vector_fetch_k"]
bm25_top_k: int = _cfg["retrieval"]["bm25_top_k"]
rerank_top_n: int = _cfg["retrieval"]["rerank_top_n"]
batch_size: int = _cfg["ingestion"]["batch_size"]
max_retries: int = _cfg["ingestion"]["max_retries"]
data_path: str = _cfg["ingestion"]["data_path"]
cache_ttl: int = _cfg["cache"]["ttl_seconds"]
