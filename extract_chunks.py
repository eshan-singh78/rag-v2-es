"""
extract_chunks.py
-----------------
Reads all PDFs in a folder, extracts text page by page, splits into chunks,
and saves everything to a single JSON file: chunks.json

Each record:
{
    "pdf":       "data/1289549364138.pdf",
    "page":      3,
    "chunk_idx": 1,
    "text":      "..."
}

Usage:
    python extract_chunks.py                  # uses ./data, outputs ./chunks.json
    python extract_chunks.py --data path/to/pdfs --out my_chunks.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 64
MIN_CHUNK_LEN = 150   # discard anything shorter (headers, footers, noise)

_CONTROL_CHARS = re.compile(r'[^\x20-\x7E\n£€°]')
_SOFT_NEWLINE  = re.compile(r'(?<![.!?:])\n(?!\n)')
_MULTI_SPACE   = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{3,}')


def clean(text: str) -> str:
    text = _CONTROL_CHARS.sub(' ', text)
    text = _SOFT_NEWLINE.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    return text.strip()


def extract(data_path: str, out_path: str):
    pdf_files = sorted(Path(data_path).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_path}")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}", end="  ", flush=True)
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        file_chunks = 0
        for page in pages:
            text = clean(page.page_content)
            if len(text) < MIN_CHUNK_LEN:
                continue
            page_num = page.metadata.get("page", 0)

            for idx, chunk in enumerate(splitter.split_text(text)):
                if len(chunk) < MIN_CHUNK_LEN:
                    continue
                all_chunks.append({
                    "pdf":       str(pdf_path),
                    "page":      page_num,
                    "chunk_idx": idx,
                    "text":      chunk,
                })
                file_chunks += 1

        print(f"→ {file_chunks} chunks")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_chunks)} total chunks saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data",       help="Folder containing PDFs")
    parser.add_argument("--out",  default="chunks.json", help="Output JSON file")
    args = parser.parse_args()
    extract(args.data, args.out)
