"""
Phase 0b — Re-extract papers that previously fell back to abstract-only,
now that supervisor-supplied PDFs are available locally.

For every question collection:
  - find included papers where extraction_source == 'abstract'
  - if a local PDF exists at data/pdfs/{paper_id}.pdf
  - re-run extraction (this time it'll be fulltext via RAG)
  - results auto-saved to Qdrant

Usage:
    python re_extract_with_pdfs.py
    python re_extract_with_pdfs.py q1_technical    # only one question
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.extraction.extractor import FullTextExtractor
from src.indexer.indexer import QdrantIndexer
from src.models import ScreeningStatus
from config.settings import settings


def re_extract_for_question(qid: str) -> tuple[int, int, int]:
    """
    Returns (re_extracted_count, skipped_no_pdf, skipped_already_fulltext).
    """
    suffix = "_" + qid
    indexer = QdrantIndexer(collection_suffix=suffix)
    extractor = FullTextExtractor(indexer=indexer, collection_suffix=suffix)

    # Pull included papers that were extracted from abstract only
    points = indexer.client.scroll(
        collection_name=indexer.collection_name,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )[0]

    candidates = []
    for p in points:
        pl = p.payload
        if pl.get("screening_status") != ScreeningStatus.INCLUDED.value:
            continue
        if pl.get("extraction_source") != "abstract":
            continue
        candidates.append(pl)

    if not candidates:
        print(f"  No abstract-only papers in {qid}, nothing to do.")
        return 0, 0, 0

    re_extracted = 0
    skipped_no_pdf = 0

    for pl in candidates:
        paper_id = pl["paper_id"]
        pdf_path = os.path.join(settings.pdf_dir, f"{paper_id}.pdf")

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1024:
            skipped_no_pdf += 1
            continue

        # Build a Paper object
        paper = indexer._point_to_paper(type("P", (), {"payload": pl})())

        print(f"  Re-extracting: {paper_id}  ({paper.title[:60]})")
        result = extractor.extract_paper(paper)
        if result.extraction_source == "fulltext":
            re_extracted += 1
        else:
            print(f"    ⚠ ended up still 'abstract' — PDF may have failed to parse")

    return re_extracted, skipped_no_pdf, len(candidates)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        qids = args
    else:
        qids = [q["id"] for q in QUESTIONS]

    grand_re = 0
    grand_skip = 0
    grand_total = 0

    for qid in qids:
        print(f"\n{'='*60}\n{qid}\n{'='*60}")
        try:
            re_count, skip, total = re_extract_for_question(qid)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        grand_re += re_count
        grand_skip += skip
        grand_total += total
        print(f"  → re-extracted: {re_count}, no PDF available: {skip}, "
              f"total abstract-only candidates: {total}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Total re-extracted (fulltext): {grand_re}")
    print(f"  Skipped (no PDF locally)     : {grand_skip}")
    print(f"  Total candidates inspected   : {grand_total}")
