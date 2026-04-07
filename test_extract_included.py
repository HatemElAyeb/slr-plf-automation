"""
Runs full-text extraction on all included papers currently in Qdrant.
Run: python test_extract_included.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.extraction.extractor import FullTextExtractor
from src.models import ScreeningStatus

if __name__ == "__main__":
    extractor = FullTextExtractor()

    # Fetch included papers from Qdrant
    points = extractor.indexer.get_points_by_status(ScreeningStatus.INCLUDED)
    papers = [extractor.indexer._point_to_paper(p) for p in points]
    print(f"\n=== Found {len(papers)} included papers ===\n")

    success, skipped = 0, 0

    for paper in papers:
        print(f"Paper : {paper.title[:70]}")
        print(f"Source: {paper.source} | PDF: {paper.pdf_url or 'None'}")

        result = extractor.extract_paper(paper)

        if result is None:
            print("  SKIP — PDF not available\n")
            skipped += 1
            continue

        print(f"  Animal species : {result.animal_species}")
        print(f"  Sensor types   : {result.sensor_types}")
        print(f"  ML methods     : {result.ml_methods}")
        print(f"  Performance    : {result.performance_metrics}")
        print(f"  Dataset size   : {result.dataset_size}")
        print(f"  Key findings   : {result.key_findings[:120]}...")
        print()
        success += 1

    print("="*50)
    print(f"  Extracted : {success}")
    print(f"  Skipped   : {skipped} (no PDF)")
    print(f"  Total     : {len(papers)}")
