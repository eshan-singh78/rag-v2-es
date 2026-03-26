"""
Ingestion pipeline — parallel at every stage:
  - PDF loading:   ProcessPoolExecutor (CPU bound, one process per file)
  - Embed+upsert:  ThreadPoolExecutor with batched embedding calls
  - ID dedup:      Qdrant scroll to fetch existing point IDs
  - BM25 rebuild:  Parallel scroll batches
"""
import argparse
import hashlib
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from qdrant_client.models import PointStruct
from tqdm import tqdm

import config
import logger as log
import bm25_index
from db import get_client, delete_collection, COLLECTION
from get_embedding_function import embed_texts

DATA_PATH = config.data_path
BATCH_SIZE = config.batch_size
MAX_RETRIES = config.max_retries
INGEST_WORKERS = config.ingest_workers


# ── Per-file worker (subprocess) ──────────────────────────────────────────────
def _process_pdf(pdf_path: str) -> list[dict]:
    """Load, clean, and chunk one PDF. Runs in a worker process."""
    import hashlib, re
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import config

    _PRINT_META = re.compile(r'\[.*?reprint.*?\]', re.IGNORECASE)
    _CONTROL_CHARS = re.compile(r'[^\x20-\x7E\n£€°]')
    _SOFT_NEWLINE = re.compile(r'(?<![.!?:])\n(?!\n)')
    _MULTI_SPACE = re.compile(r'[ \t]+')
    _MULTI_NEWLINE = re.compile(r'\n{3,}')

    # Noise patterns that indicate non-regulatory boilerplate
    _NOISE_PATTERNS = re.compile(
        r'\b(page|schedule|proforma|circular no|inserted vide)\b'
        r'|^table\b|^note:',
        re.IGNORECASE,
    )

    def _clean(text: str) -> str:
        text = _PRINT_META.sub('', text)
        text = _CONTROL_CHARS.sub(' ', text)
        text = _SOFT_NEWLINE.sub(' ', text)
        text = _MULTI_SPACE.sub(' ', text)
        text = _MULTI_NEWLINE.sub('\n\n', text)
        return text.strip()

    def is_useful_chunk(text: str) -> bool:
        """Return False for chunks that are too short, sparse, or boilerplate."""
        if len(text) < 150:
            return False
        if len(text.split()) < 20:
            return False
        if _NOISE_PATTERNS.search(text):
            return False
        return True

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
        if len(text) < 150:          # raised from 30 — filters headers/footers
            continue
        source = page.metadata.get("source", pdf_path)
        page_num = page.metadata.get("page", 0)

        for p_idx, parent_text in enumerate(parent_splitter.split_text(text)):
            parent_id = hashlib.sha256(
                f"{source}:{page_num}:{p_idx}:{parent_text[:80]}".encode()
            ).hexdigest()[:16]
            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
                if not is_useful_chunk(child_text):   # quality gate
                    continue
                # Qdrant point IDs must be unsigned integers or UUIDs.
                # We derive a stable uint64 from the sha256 of the child_id string.
                child_id_str = f"{parent_id}:{c_idx}"
                point_id = int(
                    hashlib.sha256(child_id_str.encode()).hexdigest()[:16], 16
                )
                records.append({
                    "point_id": point_id,
                    "child_id": child_id_str,
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "source": source,
                    "page": page_num,
                    "parent_id": parent_id,
                })
    return records


# ── Parallel load + chunk ─────────────────────────────────────────────────────
def load_and_chunk_parallel(data_path: str) -> list[dict]:
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
    workers = min(os.cpu_count() or 4, len(pdf_files), 16)

    with log.timer("load_and_chunk"):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_pdf, p): p for p in pdf_files}
            with tqdm(total=len(pdf_files), desc="Loading & chunking PDFs", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        all_records.extend(future.result())
                    except Exception as e:
                        log.error("pdf_worker_error", file=futures[future], error=str(e))
                    pbar.update(1)

    log.info("chunks_built",
             files=len(pdf_files),
             parents=len({r["parent_id"] for r in all_records}),
             children=len(all_records))
    return all_records


# ── ID dedup via Qdrant scroll ────────────────────────────────────────────────
def fetch_existing_ids(client) -> set[int]:
    """Scroll through all points and collect existing point IDs."""
    existing: set[int] = set()
    next_offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in results:
            existing.add(point.id)
        if next_offset is None:
            break
    return existing


# ── Embed + upsert ────────────────────────────────────────────────────────────
def _ingest_batch(client, batch: list[dict], batch_num: int):
    texts = [r["child_text"] for r in batch]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectors = embed_texts(texts)
            points = [
                PointStruct(
                    id=r["point_id"],
                    vector=vec,
                    payload={
                        "child_id": r["child_id"],
                        "child_text": r["child_text"],
                        "parent_text": r["parent_text"],
                        "source": r["source"],
                        "page": r["page"],
                        "parent_id": r["parent_id"],
                    },
                )
                for r, vec in zip(batch, vectors)
            ]
            client.upsert(collection_name=COLLECTION, points=points)
            return
        except Exception as e:
            log.warning("batch_retry", batch=batch_num, attempt=attempt, error=str(e))
            time.sleep(2 ** attempt)
    log.error("batch_failed", batch=batch_num)


def add_to_qdrant(records: list[dict]):
    client = get_client()

    with log.timer("dedup_check"):
        existing_ids = fetch_existing_ids(client)

    log.info("existing_chunks", count=len(existing_ids))
    new_records = [r for r in records if r["point_id"] not in existing_ids]

    if not new_records:
        print("No new documents to add.")
        return

    log.info("ingesting", new_chunks=len(new_records))
    batches = [new_records[i:i + BATCH_SIZE] for i in range(0, len(new_records), BATCH_SIZE)]
    progress = tqdm(total=len(new_records), desc="Embedding & upserting", unit="chunk")

    def _tracked(client, batch, num):
        _ingest_batch(client, batch, num)
        progress.update(len(batch))

    with log.timer("ingest_all"), ThreadPoolExecutor(max_workers=INGEST_WORKERS) as executor:
        futures = {executor.submit(_tracked, client, batch, i): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error("ingest_worker_error", error=str(e))

    progress.close()
    log.info("ingestion_complete", total=len(new_records))


# ── BM25 rebuild ──────────────────────────────────────────────────────────────
def rebuild_bm25(client):
    log.info("rebuilding_bm25_index")
    all_metadata: list[dict] = []
    next_offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            if point.payload:
                all_metadata.append(point.payload)
        if next_offset is None:
            break

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
        delete_collection()

    with log.timer("pipeline_total"):
        records = load_and_chunk_parallel(DATA_PATH)
        add_to_qdrant(records)

    rebuild_bm25(get_client())


if __name__ == "__main__":
    main()
