"""
Pinecone local (Docker) vector store — singleton connection.
All modules import get_index() from here.

Pinecone local quirks vs cloud:
  - Default port is 5080 (not 5081)
  - create_index() requires ServerlessSpec even locally
  - Index must be targeted via describe_index().host, not by name
  - GRPCClientConfig(secure=False) required — no TLS on local
"""
import threading
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec

import config
import logger as log

_lock = threading.Lock()
_client: PineconeGRPC | None = None
_index = None


def get_client() -> PineconeGRPC:
    """Singleton gRPC client pointed at the local Docker instance."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = PineconeGRPC(
                    api_key="pclocal",  # required field, value ignored by local
                    host=f"http://{config.db_host}:{config.db_port}",
                )
                log.info("pinecone_client_init",
                         host=config.db_host, port=config.db_port)
    return _client


def get_index():
    """
    Singleton index handle.
    Creates the index if it doesn't exist, then connects via the resolved
    index host with TLS disabled (required for Pinecone local).
    """
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                pc = get_client()

                if not pc.has_index(config.db_index_name):
                    pc.create_index(
                        name=config.db_index_name,
                        vector_type="dense",
                        dimension=config.db_dimension,
                        metric=config.db_metric,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        deletion_protection="disabled",
                    )
                    log.info("pinecone_index_created", name=config.db_index_name)

                # Resolve the per-index host, then open a no-TLS gRPC connection
                index_host = pc.describe_index(name=config.db_index_name).host
                _index = pc.Index(
                    host=index_host,
                    grpc_config=GRPCClientConfig(secure=False),
                )
                log.info("pinecone_index_ready",
                         name=config.db_index_name, host=index_host)
    return _index


def delete_index():
    """Drop the index (used for --reset). Clears singleton so it's recreated on next call."""
    global _index
    pc = get_client()
    if pc.has_index(config.db_index_name):
        pc.delete_index(config.db_index_name)
        log.info("pinecone_index_deleted", name=config.db_index_name)
    with _lock:
        _index = None
