import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from db import get_db
import logger as log

import config

DATA_PATH = config.data_path
BATCH_SIZE = config.batch_size
MAX_RETRIES = config.max_retries

_PRINT_META = re.compile(r'\[.*?reprint.*?\]', re.IGNORECASE)
_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{3,}')
_SOFT_NEWLINE = re.compile(r'(?<![.!?:])\n(?!\n)')
_CONTROL_CHARS = re.compile(r'[^\x20-\x7E\n£€°]')


def clean_text(text: str) -> str:
    text = _PRINT_META.sub('', text)
    text = _CONTROL_CHARS.sub(' ', text)
    text = _SOFT_NEWLINE.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    return text.strip()


def load_documents() -> list[Document]:
    with log.timer("load_documents"):
        docs = PyPDFDirectoryLoader(DATA_PATH).load()
    log.info("documents_loaded", count=len(docs))
    return docs


def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned = []
    for doc in documents:
        text = clean_text(doc.page_content)
        if len(text) < 30:
            log.warning("page_skipped", source=doc.metadata.get("source"), page=doc.metadata.get("page"))
            continue
        doc.page_content = text
        cleaned.append(doc)
    log.info("documents_cleaned", count=len(cleaned))
    return cleaned


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    log.info("documents_split", chunks=len(chunks))
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    return chunks


def _ingest_batch(db: PGVector, batch: list[Document], batch_num: int):
    ids = [c.metadata["id"] for c in batch]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            db.add_documents(batch, ids=ids)
            log.info("batch_ingested", batch=batch_num, size=len(batch))
            return
        except Exception as e:
            log.warning("batch_retry", batch=batch_num, attempt=attempt, error=str(e))
            time.sleep(2 ** attempt)
    log.error("batch_failed", batch=batch_num)


def add_to_pgvector(chunks: list[Document]):
    db = get_db()
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Fetch existing IDs to avoid re-ingesting
    try:
        existing = db.get_by_ids([c.metadata["id"] for c in chunks_with_ids])
        existing_ids = {doc.id for doc in existing if doc.id}
    except Exception:
        existing_ids = set()

    log.info("existing_chunks", count=len(existing_ids))
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if not new_chunks:
        log.info("no_new_chunks")
        return

    log.info("ingesting", new_chunks=len(new_chunks))
    batches = [new_chunks[i:i + BATCH_SIZE] for i in range(0, len(new_chunks), BATCH_SIZE)]

    with log.timer("ingest_all"), ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_ingest_batch, db, batch, i): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            future.result()

    log.info("ingestion_complete", total=len(new_chunks))


def clear_database():
    db = get_db()
    db.delete_collection()
    log.info("database_cleared")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    with log.timer("pipeline_total"):
        documents = load_documents()
        documents = clean_documents(documents)
        chunks = split_documents(documents)
        add_to_pgvector(chunks)


if __name__ == "__main__":
    main()
