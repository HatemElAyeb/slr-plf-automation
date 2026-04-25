"""
Runs hybrid extraction on all included papers currently in Qdrant.
Tries full-text PDF first, falls back to abstract if PDF unavailable.
Run: python test_extract_included.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.extraction.extractor import FullTextExtractor
from src.models import ScreeningStatus

if __name__ == "__main__":
    extractor = FullTextExtractor()

    points = extractor.indexer.get_points_by_status(ScreeningStatus.INCLUDED)
    papers = [extractor.indexer._point_to_paper(p) for p in points]
    print(f"\n=== Found {len(papers)} included papers ===\n")

    fulltext, abstract_only = 0, 0

    for paper in papers:
        print(f"Paper : {paper.title[:70]}")
        print(f"Source: {paper.source}")

        result = extractor.extract_paper(paper)

        tag = "FULLTEXT" if result.extraction_source == "fulltext" else "ABSTRACT"
        if result.extraction_source == "fulltext":
            fulltext += 1
        else:
            abstract_only += 1

        print(f"  [{tag}] species={result.animal_species}")
        print(f"           sensors={result.sensor_types}")
        print(f"           methods={result.ml_methods}")
        print(f"           metrics={result.performance_metrics}")
        print(f"           dataset_size={result.dataset_size}")
        print(f"           findings={result.key_findings[:100]}...")
        print()

    print("="*50)
    print(f"  Extracted from full text : {fulltext}")
    print(f"  Extracted from abstract  : {abstract_only}")
    print(f"  Total                    : {len(papers)}")
