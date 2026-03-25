"""
Quick smoke test for all three collectors.
Run from the project root: python test_collectors.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.collectors.pubmed import PubMedCollector
from src.collectors.openalex import OpenAlexCollector
from src.collectors.arxiv import ArXivCollector
from src.collectors.collector import LiteratureCollector

QUERY = "precision livestock farming sensor"
EMAIL = "test@example.com"


def test_pubmed():
    print("\n--- PubMed ---")
    collector = PubMedCollector(email=EMAIL)
    papers = collector.search(QUERY, max_results=5)
    assert len(papers) > 0, "PubMed returned 0 papers"
    p = papers[0]
    print(f"  First paper : {p.title[:80]}")
    print(f"  Year        : {p.year}")
    print(f"  Authors     : {p.authors[:2]}")
    print(f"  DOI         : {p.doi}")
    print(f"  Source      : {p.source}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_openalex():
    print("\n--- OpenAlex ---")
    collector = OpenAlexCollector(email=EMAIL)
    papers = collector.search(QUERY, max_results=5)
    assert len(papers) > 0, "OpenAlex returned 0 papers"
    p = papers[0]
    print(f"  First paper : {p.title[:80]}")
    print(f"  Year        : {p.year}")
    print(f"  PDF URL     : {p.pdf_url}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_arxiv():
    print("\n--- ArXiv ---")
    collector = ArXivCollector()
    papers = collector.search(
        query="livestock monitoring sensor",
        categories=["cs.AI", "cs.CV", "eess.SP"],
        max_results=5,
    )
    assert len(papers) > 0, "ArXiv returned 0 papers"
    p = papers[0]
    print(f"  First paper : {p.title[:80]}")
    print(f"  Year        : {p.year}")
    print(f"  PDF URL     : {p.pdf_url}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_deduplication():
    print("\n--- Deduplication ---")
    collector = LiteratureCollector()
    papers = collector.collect(
        pubmed_query=QUERY,
        openalex_query=QUERY,
        arxiv_query="livestock monitoring",
        arxiv_categories=["cs.AI", "cs.CV"],
        max_per_source=10,
    )
    assert len(papers) > 0
    ids = [p.id for p in papers]
    assert len(ids) == len(set(ids)), "Duplicate IDs found after deduplication!"
    print(f"  PASS — {len(papers)} unique papers across all sources")
    return papers


if __name__ == "__main__":
    results = {}
    tests = [
        ("PubMed", test_pubmed),
        ("OpenAlex", test_openalex),
        ("ArXiv", test_arxiv),
        ("Deduplication", test_deduplication),
    ]

    for name, fn in tests:
        try:
            fn()
            results[name] = "PASS"
        except Exception as e:
            print(f"  FAIL — {e}")
            results[name] = f"FAIL: {e}"

    print("\n========== RESULTS ==========")
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  {name}: {status}")
