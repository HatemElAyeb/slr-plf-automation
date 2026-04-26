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
from src.collectors.mdpi import MDPICollector
from src.collectors.springer import SpringerCollector
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
    print(f"  Venue       : {p.venue_name} ({p.venue_issn})")
    print(f"  Quartile    : {p.quartile}")
    print(f"  Conf rank   : {p.conference_rank}")
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
    print(f"  Venue       : {getattr(p, 'venue_name', None)} ({getattr(p, 'venue_issn', None)})")
    print(f"  Quartile    : {getattr(p, 'quartile', None)}")
    print(f"  Conf rank   : {getattr(p, 'conference_rank', None)}")
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
    print(f"  Venue       : {getattr(p, 'venue_name', None)} ({getattr(p, 'venue_issn', None)})")
    print(f"  Quartile    : {getattr(p, 'quartile', None)}")
    print(f"  Conf rank   : {getattr(p, 'conference_rank', None)}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_mdpi():
    print("\n--- MDPI ---")
    collector = MDPICollector()
    papers = collector.search(QUERY, max_results=5)
    assert len(papers) > 0, "MDPI returned 0 papers"
    p = papers[0]
    print(f"  First paper : {p.title[:80]}")
    print(f"  Year        : {p.year}")
    print(f"  DOI         : {p.doi}")
    print(f"  PDF URL     : {p.pdf_url}")
    print(f"  Venue       : {getattr(p, 'venue_name', None)} ({getattr(p, 'venue_issn', None)})")
    print(f"  Quartile    : {getattr(p, 'quartile', None)}")
    print(f"  Conf rank   : {getattr(p, 'conference_rank', None)}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_springer():
    print("\n--- Springer ---")
    collector = SpringerCollector()
    papers = collector.search(QUERY, max_results=5)
    assert len(papers) > 0, "Springer returned 0 papers"
    p = papers[0]
    print(f"  First paper : {p.title[:80]}")
    print(f"  Year        : {p.year}")
    print(f"  DOI         : {p.doi}")
    print(f"  PDF URL     : {p.pdf_url}")
    print(f"  Venue       : {getattr(p, 'venue_name', None)} ({getattr(p, 'venue_issn', None)})")
    print(f"  Quartile    : {getattr(p, 'quartile', None)}")
    print(f"  Conf rank   : {getattr(p, 'conference_rank', None)}")
    print(f"  Abstract    : {p.abstract[:120]}...")
    print(f"  PASS — {len(papers)} papers")
    return papers


def test_deduplication():
    print("\n--- Deduplication (all 5 sources) ---")
    collector = LiteratureCollector()
    papers = collector.collect(
        pubmed_query=QUERY,
        openalex_query=QUERY,
        arxiv_query="livestock monitoring",
        arxiv_categories=["cs.AI", "cs.CV"],
        max_per_source=5,
    )
    assert len(papers) > 0
    ids = [p.id for p in papers]
    assert len(ids) == len(set(ids)), "Duplicate IDs found after deduplication!"
    print(f"  PASS — {len(papers)} unique papers across all sources")
    return papers


if __name__ == "__main__":
    results = {}
    tests = [
        ("PubMed",        test_pubmed),
        ("OpenAlex",      test_openalex),
        ("ArXiv",         test_arxiv),
        ("MDPI",          test_mdpi),
        ("Springer",      test_springer),
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
