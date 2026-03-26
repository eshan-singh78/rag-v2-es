"""
Pinecone local (Docker) vector store — singleton connection.
All modules import get_index() from here.
"""
import threading
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC

import config
import logger as log

_lock = threading.Lock()
_client: PineconeGRPC | None = None
_index = None


def get_client() -> PineconeGRPC:
    """Return a singleton Pinecone gRPC client pointed at the local Docker instance."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = PineconeGRPC(
                    api_key="local",          # Pinecone local ignores the key
                    host=f"http://{config.db_host}:{config.db_port}",
                )
                log.info("pinecone_client_init", host=config.db_host, port=config.db_port)
    return _client


def get_index():
    """Return a singleton handle to the configured Pinecone index."""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                pc = get_client()
                existing = [i.name for i in pc.list_indexes()]
                if config.db_index_name not in existing:
                    pc.create_index(
                        name=config.db_index_name,
                        dimension=config.db_dimension,
                        metric=config.db_metric,
                    )
                    log.info("pinecone_index_created", name=config.db_index_name)
                _index = pc.Index(config.db_index_name)
                log.info("pinecone_index_ready", name=config.db_index_name)
    return _index


def delete_index():
    """Drop and recreate the index (used for --reset)."""
    global _index
    pc = get_client()
    existing = [i.name for i in pc.list_indexes()]
    if config.db_index_name in existing:
        pc.delete_index(config.db_index_name)
        log.info("pinecone_index_deleted", name=config.db_index_name)
    with _lock:
        _index = None
