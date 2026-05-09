"""
Match supervisor-supplied PDFs (with arbitrary filenames) to paper IDs
in Qdrant by extracting DOIs from their first page, then copy them into
data/pdfs/{paper_id}.pdf so the extraction pipeline can find them.

Usage:
    python match_supervisor_pdfs.py "C:/Users/MSI/Downloads/.../Missing_pdfs"
    python match_supervisor_pdfs.py "C:/Users/MSI/Downloads/.../Missing_pdfs" --dry-run
"""
import os
import re
import sys
import shutil
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from research_questions import QUESTIONS

# DOI regex — matches the standard 10.xxxx/yyyy format
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def extract_doi_from_pdf(path: str, max_pages: int = 2) -> str | None:
    """Return the first DOI found in the PDF's first few pages (or None)."""
    try:
        doc = fitz.open(path)
    except Exception:
        return None

    text = ""
    for i in range(min(max_pages, doc.page_count)):
        text += doc[i].get_text()
    doc.close()

    matches = DOI_RE.findall(text)
    if not matches:
        return None
    # Take the first match, strip trailing punctuation
    doi = matches[0].rstrip(".,;:")
    return doi


def find_paper_by_doi(client: QdrantClient, doi: str) -> tuple[str, str] | None:
    """Search every question collection for this DOI. Return (collection, paper_id) or None."""
    doi_norm = doi.lower().strip()
    for q in QUESTIONS:
        collection = settings.abstracts_collection + "_" + q["id"]
        try:
            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doi", match=MatchValue(value=doi_norm))]
                ),
                limit=1,
                with_payload=True,
            )
            if results:
                return collection, results[0].payload.get("paper_id", "")
        except Exception:
            continue
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Folder containing supervisor's PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy, just report matches")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Source folder not found: {args.source_dir}")
        sys.exit(1)

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    target_dir = settings.pdf_dir
    os.makedirs(target_dir, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(args.source_dir) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs in {args.source_dir}\n")

    matched, unmatched, skipped = [], [], []

    for fname in pdf_files:
        src = os.path.join(args.source_dir, fname)
        doi = extract_doi_from_pdf(src)
        if not doi:
            print(f"  ✗ {fname:<10} no DOI found in PDF")
            unmatched.append(fname)
            continue

        match = find_paper_by_doi(client, doi.lower())
        if not match:
            # Try with case-insensitive fallback (some DOIs use mixed case)
            match = find_paper_by_doi(client, doi)

        if not match:
            print(f"  ✗ {fname:<10} DOI {doi!r} not found in any Qdrant collection")
            unmatched.append(fname)
            continue

        collection, paper_id = match
        dest = os.path.join(target_dir, f"{paper_id}.pdf")

        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            # Already in place — skip
            print(f"  ↺ {fname:<10} → {paper_id}.pdf (already exists, skipping)")
            skipped.append(fname)
            continue

        if args.dry_run:
            print(f"  ✓ {fname:<10} → {paper_id}.pdf  [DRY RUN]")
        else:
            shutil.copy2(src, dest)
            print(f"  ✓ {fname:<10} → {paper_id}.pdf  copied")
        matched.append((fname, paper_id, collection))

    print(f"\n{'='*50}")
    print(f"Matched   : {len(matched)}")
    print(f"Skipped   : {len(skipped)}  (already in data/pdfs/)")
    print(f"Unmatched : {len(unmatched)}")
    if unmatched:
        print(f"\nUnmatched files (you may need to find DOIs manually):")
        for f in unmatched:
            print(f"  - {f}")
