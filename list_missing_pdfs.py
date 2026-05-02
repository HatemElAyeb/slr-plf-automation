"""
List included papers whose PDFs could not be obtained, across all question
collections. Outputs a CSV at data/missing_pdfs.csv that can be sent to the
supervisor for institutional access lookup.

A paper counts as "missing PDF" if it was screened as INCLUDED but the
extraction had to fall back to abstract-only (extraction_source == "abstract"),
which means the PDF either was not available or could not be downloaded.

Run: python list_missing_pdfs.py
"""
import sys
import os
import csv
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from research_questions import QUESTIONS
from config.settings import settings

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "missing_pdfs.csv")


def get_missing_for_question(client: QdrantClient, qid: str) -> list[dict]:
    """Return payloads of included papers without a downloadable PDF."""
    collection = settings.abstracts_collection + "_" + qid
    try:
        results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="screening_status", match=MatchValue(value="included")),
                    FieldCondition(key="extraction_source", match=MatchValue(value="abstract")),
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in results]
    except Exception as e:
        # Collection doesn't exist yet for this question — skip silently
        return []


if __name__ == "__main__":
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    rows = []
    for q in QUESTIONS:
        qid = q["id"]
        missing = get_missing_for_question(client, qid)
        if not missing:
            continue
        print(f"  {qid}: {len(missing)} missing PDF(s)")
        for p in missing:
            rows.append({
                "question_id": qid,
                "question_category": q["category"],
                "title": p.get("title", ""),
                "authors": "; ".join(p.get("authors", [])[:3])
                           + (" et al." if len(p.get("authors", [])) > 3 else ""),
                "year": p.get("year", ""),
                "doi": p.get("doi", ""),
                "source": p.get("source", ""),
                "venue_name": p.get("venue_name", ""),
                "quartile": p.get("quartile", ""),
                "pdf_url_tried": p.get("pdf_url", ""),
            })

    if not rows:
        print("\nNo missing PDFs found. Either no questions have been run, "
              "or all included papers have full text available.")
        sys.exit(0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal: {len(rows)} missing PDFs across {len({r['question_id'] for r in rows})} question(s)")
    print(f"Saved to: {OUT_PATH}")
    print("\nYou can open this CSV in Excel and send it to your supervisor.")
