"""
Smoke test for QdrantIndexer.
Run: python test_indexer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.collectors.collector import LiteratureCollector
from src.indexer.indexer import QdrantIndexer
from src.models import ScreeningStatus

if __name__ == "__main__":
    # 1. Collect a small batch of papers
    print("=== Step 1: Collecting papers ===")
    collector = LiteratureCollector()
    papers = collector.collect(
        pubmed_query="precision livestock farming sensor",
        openalex_query="precision livestock farming sensor",
        arxiv_query="livestock monitoring sensor",
        arxiv_categories=["cs.AI", "cs.CV"],
        max_per_source=5,
    )
    print(f"Collected {len(papers)} papers\n")

    # 2. Index into Qdrant
    print("=== Step 2: Indexing into Qdrant ===")
    indexer = QdrantIndexer()
    indexer.index_papers(papers)

    # 3. Verify count
    print("\n=== Step 3: Verifying ===")
    total = indexer.count()
    print(f"Total papers in Qdrant: {total}")
    assert total > 0, "No papers found in Qdrant after indexing!"

    # 4. Check pending papers are retrievable
    pending = indexer.get_papers_by_status(ScreeningStatus.PENDING)
    print(f"Papers with status=pending: {len(pending)}")
    assert len(pending) > 0

    # 5. Test screening update
    sample_id = papers[0].id
    indexer.update_screening(sample_id, "included", 0.95, "Relevant to PLF sensors")
    included = indexer.get_papers_by_status(ScreeningStatus.INCLUDED)
    print(f"Papers with status=included after update: {len(included)}")
    assert len(included) == 1

    print("\n========== RESULT ==========")
    print(f"  ✓  Indexer: PASS — {total} papers stored in Qdrant")
