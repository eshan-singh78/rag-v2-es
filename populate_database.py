"""
Ingestion pipeline — parallel at every stage:
  - PDF loading:   ThreadPoolExecutor (I/O bound)
  - Clean+chunk:   ProcessPoolExecutor (CPU bound, one process per doc)
  - Embed+upsert:  ThreadPoolExecutor with pre-batched embedding calls
  - ID dedup:      Parallel fetch batches
  - BM25 rebuild:  Parallel metadata fetch
"""
import argparse
import hashlib
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

import config
import logger as log
import bm25_index
from db import get_index, delete_index
from get_embedding_function import embed_texts

DATA_PATH = config.data_path
BATCH_SIZE = config.batch_size
MAX_RETRIES = config.max_retries
INGEST_WORKERS = config.ingest_workers

# ── Text cleaning ─────────────────────────────────────────────────────────────
_PRINT_META = re.compile(r'\[.*?reprint.*?\]', re.IGNORECASE)
_CONTROL_CHARS = re.compile(r'[^\x20-\x7E\n£€°]')
_SOFT_NEWLINE = re.compile(r'(?<![.!?:])\n(?!\n)')
_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{3,}')


def clean_text(text: str) -> str:
    text = _PRINT_META.sub('', text)
    text = _CONTROL_CHARS.sub(' ', text)
    text = _SOFT_NEWLINE.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    return text.strip()


# ── Per-file worker (runs in a subprocess) ────────────────────────────────────
def _process_pdf(pdf_path: str) -> list[dict]:
    """
    Load, clean, and chunk a single PDF.
    Runs in a worker process — no shared state needed.
    Returns a flat list of chunk records.
    """
    import hashlib
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import config, re

    _PRINT_META = re.compile(r'\[.*?reprint.*?\]', re.IGNORECASE)
    _CONTROL_CHARS = re.compile(r'[^\x20-\x7E\n£€°]')
    _SOFT_NEWLINE = re.compile(r'(?<![.!?:])\n(?!\n)')
    _MULTI_SPACE = re.compile(r'[ \t]+')
    _MULTI_NEWLINE = re.compile(r'\n{3,}')

    def _clean(text: str) -> str:
        text = _PRINT_META.sub('', text)
        text = _CONTROL_CHARS.sub(' ', text)
        text = _SOFT_NEWLINE.sub(' ', text)
        text = _MULTI_SPACE.sub(' ', text)
        text = _MULTI_NEWLINE.sub('\n\n', text)
        return text.strip()

    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=config.parent_chunk_size,
        chunk_overlap=config.parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
    )

    try:
        pages = PyPDFLoader(pdf_path).load()
    except Exception:
        return []

    records: list[dict] = []
    for page in pages:
        text = _clean(page.page_content)
        if len(text) < 30:
            continue
        source = page.metadata.get("source", pdf_path)
        page_num = page.metadata.get("page", 0)

        for p_idx, parent_text in enumerate(parent_splitter.split_text(text)):
            parent_id = hashlib.sha256(
                f"{source}:{page_num}:{p_idx}:{parent_text[:80]}".encode()
            ).hexdigest()[:16]
            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
                records.append({
                    "child_id": f"{parent_id}:{c_idx}",
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "source": source,
                    "page": page_num,
                    "parent_id": parent_id,
                })
    return records


# ── Parallel load + chunk ─────────────────────────────────────────────────────
def load_and_chunk_parallel(data_path: str) -> list[dict]:
    """
    Discover all PDFs, then fan out to a ProcessPoolExecutor —
    each worker loads, cleans, and chunks one PDF independently.
    """
    pdf_files = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        log.warning("no_pdfs_found", path=data_path)
        return []

    log.info("pdfs_discovered", count=len(pdf_files))
    all_records: list[dict] = []

    # Use min(cpu_count, file_count) workers — no point spawning more than files
    workers = min(os.cpu_count() or 4, len(pdf_files), 16)
    with log.timer("load_and_chunk"):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_pdf, p): p for p in pdf_files}
            with tqdm(total=len(pdf_files), desc="Loading & chunking PDFs", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        records = future.result()
                        all_records.extend(records)
                    except Exception as e:
                        log.error("pdf_worker_error", file=futures[future], error=str(e))
                    pbar.update(1)

    log.info("chunks_built",
             files=len(pdf_files),
             parents=len({r["parent_id"] for r in all_records}),
             children=len(all_records))
    return all_records


# ── Parallel ID dedup ─────────────────────────────────────────────────────────
def _fetch_id_batch(index, ids: list[str]) -> set[str]:
    try:
        result = index.fetch(ids=ids)
        return set(result.vectors.keys())
    except Exception as e:
        log.warning("fetch_existing_failed", error=str(e))
        return set()


def fetch_existing_ids_parallel(index, all_ids: list[str]) -> set[str]:
    batches = [all_ids[i:i + 100] for i in range(0, len(all_ids), 100)]
    existing: set[str] = set()
    with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as executor:
        futures = [executor.submit(_fetch_id_batch, index, b) for b in batches]
        for future in as_completed(futures):
            existing.update(future.result())
    return existing


# ── Embed + upsert ────────────────────────────────────────────────────────────
def _ingest_batch(index, batch: list[dict], batch_num: int):
    """Embed an entire batch in one call, then upsert. Retries with backoff."""
    texts = [r["child_text"] for r in batch]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectors = embed_texts(texts)   # single batched embedding call
            index.upsert(vectors=[
                {
                    "id": r["child_id"],
                    "values": vec,
                    "metadata": {
                        "child_text": r["child_text"],
                        "parent_text": r["parent_text"],
                        "source": r["source"],
                        "page": r["page"],
                        "parent_id": r["parent_id"],
                    },
                }
                for r, vec in zip(batch, vectors)
            ])
            return
        except Exception as e:
            log.warning("batch_retry", batch=batch_num, attempt=attempt, error=str(e))
            time.sleep(2 ** attempt)
    log.error("batch_failed", batch=batch_num)


def add_to_pinecone(records: list[dict]):
    index = get_index()

    with log.timer("dedup_check"):
        all_ids = [r["child_id"] for r in records]
        existing_ids = fetch_existing_ids_parallel(index, all_ids)

    log.info("existing_chunks", count=len(existing_ids))
    new_records = [r for r in records if r["child_id"] not in existing_ids]

    if not new_records:
        print("No new documents to add.")
        return

    log.info("ingesting", new_chunks=len(new_records))
    batches = [new_records[i:i + BATCH_SIZE] for i in range(0, len(new_records), BATCH_SIZE)]
    progress = tqdm(total=len(new_records), desc="Embedding & upserting", unit="chunk")

    def _tracked(index, batch, num):
        _ingest_batch(index, batch, num)
        progress.update(len(batch))

    with log.timer("ingest_all"), ThreadPoolExecutor(max_workers=INGEST_WORKERS) as executor:
        futures = {executor.submit(_tracked, index, batch, i): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error("ingest_worker_error", error=str(e))

    progress.close()
    log.info("ingestion_complete", total=len(new_records))


# ── BM25 rebuild (parallel metadata fetch) ───────────────────────────────────
def _fetch_metadata_batch(index, ids: list[str]) -> list[dict]:
    try:
        result = index.fetch(ids=ids)
        return [v.metadata for v in result.vectors.values() if v.metadata]
    except Exception as e:
        log.warning("bm25_fetch_batch_failed", error=str(e))
        return []


def rebuild_bm25(index):
    log.info("rebuilding_bm25_index")
    all_ids: list[str] = []
    try:
        for id_batch in index.list():
            all_ids.extend(id_batch)
    except Exception as e:
        log.warning("bm25_list_ids_failed", error=str(e))
        return

    if not all_ids:
        return

    id_batches = [all_ids[i:i + 100] for i in range(0, len(all_ids), 100)]
    all_metadata: list[dict] = []

    with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as executor:
        futures = [executor.submit(_fetch_metadata_batch, index, b) for b in id_batches]
        for future in as_completed(futures):
            all_metadata.extend(future.result())

    log.info("bm25_corpus_fetched", docs=len(all_metadata))
    if all_metadata:
        bm25_index.invalidate()
        bm25_index.build(all_metadata)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe and re-ingest everything.")
    args = parser.parse_args()

    if args.reset:
        delete_index()

    with log.timer("pipeline_total"):
        records = load_and_chunk_parallel(DATA_PATH)
        add_to_pinecone(records)

    rebuild_bm25(get_index())


if __name__ == "__main__":
    main()
