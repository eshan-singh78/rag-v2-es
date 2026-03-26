"""
Qdrant vector store — singleton client + collection management.
All modules import get_client() and COLLECTION from here.

Qdrant local runs in Docker on port 6333 (REST) / 6334 (gRPC).
We use the Python client in gRPC mode for speed.
"""
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config
import logger as log

_lock = threading.Lock()
_client: QdrantClient | None = None

COLLECTION = config.db_collection_name
DISTANCE_MAP = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}


def get_client() -> QdrantClient:
    """Singleton Qdrant client (gRPC for speed, falls back to REST)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(
                    host=config.db_host,
                    grpc_port=config.db_grpc_port,
                    prefer_grpc=True,
                )
                log.info("qdrant_client_init",
                         host=config.db_host, grpc_port=config.db_grpc_port)
                _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient):
    """Create the collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        distance = DISTANCE_MAP.get(config.db_metric, Distance.COSINE)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=config.db_dimension,
                distance=distance,
            ),
        )
        log.info("qdrant_collection_created", name=COLLECTION,
                 dim=config.db_dimension, metric=config.db_metric)
    else:
        log.info("qdrant_collection_ready", name=COLLECTION)


def delete_collection():
    """Drop the collection (used for --reset). Resets the singleton."""
    global _client
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        log.info("qdrant_collection_deleted", name=COLLECTION)
    # Re-create immediately so the rest of the pipeline can proceed
    _ensure_collection(client)
