"""
Mini end-to-end pipeline test: collect → index → screen
5 papers per source (15 total after dedup)
Run: python test_pipeline_mini.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.collectors.collector import LiteratureCollector
from src.indexer.indexer import QdrantIndexer
from src.screening.screener import AbstractScreener
from src.models import ScreeningStatus

QUERY = "precision livestock farming sensor"

if __name__ == "__main__":
    # Step 1 — Collect
    print("\n=== Step 1: Collecting papers ===")
    collector = LiteratureCollector()
    papers = collector.collect(
        pubmed_query=QUERY,
        openalex_query=QUERY,
        arxiv_query="livestock monitoring sensor",
        arxiv_categories=["cs.AI", "cs.CV"],
        max_per_source=5,
    )
    print(f"Collected {len(papers)} unique papers\n")

    # Step 2 — Index
    print("=== Step 2: Indexing into Qdrant ===")
    indexer = QdrantIndexer()
    indexer.index_papers(papers)

    # Step 3 — Screen
    print("\n=== Step 3: Screening abstracts ===")
    screener = AbstractScreener(indexer=indexer)
    results = screener.screen_all(papers)

    # Summary
    included = [r for _, r in results if r.decision == ScreeningStatus.INCLUDED]
    excluded = [r for _, r in results if r.decision == ScreeningStatus.EXCLUDED]

    print(f"\n=== Results ===")
    print(f"  Total screened : {len(results)}")
    print(f"  Included       : {len(included)}")
    print(f"  Excluded       : {len(excluded)}")

    print("\n--- Included papers ---")
    for paper, result in results:
        if result.decision == ScreeningStatus.INCLUDED:
            print(f"  [{result.confidence:.2f}] {paper.title[:70]}")
            print(f"         {result.reason}")

    print("\n--- Excluded papers ---")
    for paper, result in results:
        if result.decision == ScreeningStatus.EXCLUDED:
            print(f"  [{result.confidence:.2f}] {paper.title[:70]}")
            print(f"         {result.reason}")
