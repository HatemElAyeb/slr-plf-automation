"""
Smoke test for abstract screening module.
Run from project root: python test_screening.py
Requires: Ollama running with mistral pulled
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.models import Paper, ScreeningStatus
from src.screening.screener import AbstractScreener
from src.indexer.indexer import QdrantIndexer

# Two papers — one should be included, one excluded
TEST_PAPERS = [
    Paper(
        id="test_include_1",
        title="Accelerometer-based lameness detection in dairy cows",
        abstract=(
            "Lameness is a major welfare and economic issue in dairy farming. "
            "This study presents a tri-axial accelerometer attached to the leg of "
            "dairy cows to automatically detect lameness at early stages. "
            "Machine learning classifiers achieved 91% sensitivity and 87% specificity "
            "on a dataset of 120 cows over 6 months."
        ),
        source="test",
        year=2021,
    ),
    Paper(
        id="test_exclude_1",
        title="Economic analysis of organic wheat farming in Europe",
        abstract=(
            "This paper analyzes the economic viability of transitioning from "
            "conventional to organic wheat production across 5 European countries. "
            "Using regression analysis on farm income data from 2010-2020, we show "
            "that organic certification increases net income by 18% on average."
        ),
        source="test",
        year=2022,
    ),
]


def test_single_paper():
    print("\n--- Single paper screening ---")
    screener = AbstractScreener(indexer=None)

    # Test include
    result_include = screener.screen_paper(TEST_PAPERS[0])
    print(f"  Paper : {TEST_PAPERS[0].title[:60]}")
    print(f"  Decision   : {result_include.decision.value}")
    print(f"  Confidence : {result_include.confidence}")
    print(f"  Reason     : {result_include.reason}")
    assert result_include.decision == ScreeningStatus.INCLUDED, \
        f"Expected INCLUDED, got {result_include.decision}"

    # Test exclude
    result_exclude = screener.screen_paper(TEST_PAPERS[1])
    print(f"\n  Paper : {TEST_PAPERS[1].title[:60]}")
    print(f"  Decision   : {result_exclude.decision.value}")
    print(f"  Confidence : {result_exclude.confidence}")
    print(f"  Reason     : {result_exclude.reason}")
    assert result_exclude.decision == ScreeningStatus.EXCLUDED, \
        f"Expected EXCLUDED, got {result_exclude.decision}"

    print("\n  PASS — LLM correctly classified both papers")


def test_screen_all_with_qdrant():
    print("\n--- Screen all + Qdrant update ---")
    indexer = QdrantIndexer()

    # Index the test papers first
    indexer.index_papers(TEST_PAPERS)

    screener = AbstractScreener(indexer=indexer)
    results = screener.screen_all(TEST_PAPERS)

    assert len(results) == 2

    included = indexer.get_papers_by_status(ScreeningStatus.INCLUDED)
    excluded = indexer.get_papers_by_status(ScreeningStatus.EXCLUDED)

    print(f"  Included : {len(included)}")
    print(f"  Excluded : {len(excluded)}")
    print(f"  PASS — Qdrant updated with screening results")


if __name__ == "__main__":
    results = {}
    tests = [
        ("Single paper (no Qdrant)", test_single_paper),
        ("Screen all + Qdrant update", test_screen_all_with_qdrant),
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
