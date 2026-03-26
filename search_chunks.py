"""
search_chunks.py
----------------
Search for words or phrases across the extracted chunks.json file.
Prints every match with its PDF filename, page number, and a highlighted excerpt.

Usage:
    python search_chunks.py "net worth"
    python search_chunks.py "fee limit" --chunks my_chunks.json
    python search_chunks.py "circular" --case-sensitive
    python search_chunks.py "AUA" --pdf 1289549364138.pdf   # limit to one PDF
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"chunks.json not found at '{path}'.")
        print("Run extract_chunks.py first to generate it.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def highlight(text: str, pattern: re.Pattern, width: int = 200) -> str:
    """Return a short excerpt centred around the first match, with >>> markers."""
    m = pattern.search(text)
    if not m:
        return text[:width]
    start = max(0, m.start() - 80)
    end   = min(len(text), m.end() + 80)
    excerpt = text[start:end].replace("\n", " ")
    # wrap the matched portion
    highlighted = pattern.sub(lambda x: f">>>{x.group()}<<<", excerpt)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"


def search(query: str, chunks: list[dict], case_sensitive: bool, pdf_filter: str | None):
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(re.escape(query), flags)
    except re.error as e:
        print(f"Invalid search pattern: {e}")
        sys.exit(1)

    matches = []
    for chunk in chunks:
        if pdf_filter and pdf_filter.lower() not in chunk["pdf"].lower():
            continue
        if pattern.search(chunk["text"]):
            matches.append(chunk)

    if not matches:
        print(f'No matches found for "{query}"')
        return

    # Group by PDF for cleaner output
    by_pdf: dict[str, list[dict]] = {}
    for m in matches:
        by_pdf.setdefault(m["pdf"], []).append(m)

    print(f'\nFound {len(matches)} chunk(s) across {len(by_pdf)} PDF(s) for: "{query}"\n')
    print("=" * 70)

    for pdf_path, pdf_matches in sorted(by_pdf.items()):
        pdf_name = Path(pdf_path).name
        pages = sorted({m["page"] for m in pdf_matches})
        print(f"\nPDF:   {pdf_name}")
        print(f"Pages: {pages}  ({len(pdf_matches)} chunk(s))")
        print("-" * 70)
        for m in pdf_matches:
            excerpt = highlight(m["text"], pattern)
            print(f"  [page {m['page']}, chunk {m['chunk_idx']}]  {excerpt}")

    print("\n" + "=" * 70)
    print(f"Total matches: {len(matches)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query",                          help="Word or phrase to search for")
    parser.add_argument("--chunks",         default="chunks.json", help="Path to chunks.json")
    parser.add_argument("--case-sensitive", action="store_true",   help="Case-sensitive search")
    parser.add_argument("--pdf",            default=None,          help="Filter to a specific PDF filename")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    search(args.query, chunks, args.case_sensitive, args.pdf)
