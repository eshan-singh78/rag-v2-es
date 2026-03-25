"""
Central pgvector connection. All files import get_db() from here.
Configure connection in config.toml under [database] url.
"""
from langchain_postgres import PGVector
from get_embedding_function import get_embedding_function
import config

COLLECTION_NAME = "rag_documents"
DATABASE_URL = config.database_url


def get_db() -> PGVector:
    return PGVector(
        embeddings=get_embedding_function(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
