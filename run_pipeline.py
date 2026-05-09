"""
Run the full SLR pipeline for one or more research questions.

Usage:
  python run_pipeline.py q1_technical            # run a single question by id
  python run_pipeline.py all                     # run all questions
  python run_pipeline.py q1_technical q5_gaps    # run a subset

For each question:
  1. Build queries via LLM (Module 0 query builder)
  2. Build inclusion/exclusion criteria via LLM (Module 0 criteria builder)
  3. Collect papers from PubMed, OpenAlex, ArXiv, MDPI, Springer
  4. Index into Qdrant (collection suffixed with question id)
  5. Screen abstracts using question-specific criteria
  6. Extract structured data from included papers (PDF + abstract fallback)
  7. Save run config + counts to data/runs/{question_id}/config.json
"""
import sys
import os
import json
import datetime
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.query_builder import build_queries, build_criteria, format_criteria
from src.collectors.collector import LiteratureCollector
from src.indexer.indexer import QdrantIndexer
from src.screening.screener import AbstractScreener
from src.extraction.extractor import FullTextExtractor
from src.models import ScreeningStatus

MAX_PER_SOURCE = 200  # ~1000 papers per question before dedup, ~700 unique after


def _load_override(qid: str) -> dict:
    """
    If data/runs/{qid}/override.json exists, return its contents.
    Schema (any subset is allowed):
      {
        "queries":  { pubmed_query, openalex_query, arxiv_query,
                      mdpi_query, springer_query, arxiv_categories },
        "criteria": { include, exclude }
      }
    """
    path = os.path.join("data", "runs", qid, "override.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(question: dict):
    qid = question["id"]
    qtext = question["text"]
    suffix = "_" + qid

    print(f"\n{'#'*70}")
    print(f"# Running pipeline for {qid}")
    print(f"# Question: {qtext}")
    print('#'*70)

    override = _load_override(qid)
    if override:
        print(f"\n[OVERRIDE] Loaded data/runs/{qid}/override.json — "
              f"keys: {list(override.keys())}")

    # --- 1. Build queries (or use override) ---
    if "queries" in override:
        print("\n[1/6] Using override queries (skipping LLM query builder)...")
        queries = override["queries"]
    else:
        print("\n[1/6] Building queries via LLM...")
        queries = build_queries(qtext)
    print(f"  pubmed   : {queries['pubmed_query'][:80]}...")
    print(f"  arxiv    : {queries['arxiv_query'][:80]}...")
    print(f"  categories: {queries['arxiv_categories']}")

    # --- 2. Build criteria (or use override) ---
    if "criteria" in override:
        print("\n[2/6] Using override criteria (skipping LLM criteria builder)...")
        criteria_dict = override["criteria"]
    else:
        print("\n[2/6] Building inclusion/exclusion criteria via LLM...")
        criteria_dict = build_criteria(qtext)
    criteria_text = format_criteria(criteria_dict)
    print(f"  Criteria ({len(criteria_text)} chars)")

    # --- 3. Collect ---
    print("\n[3/6] Collecting papers from all 5 sources...")
    collector = LiteratureCollector()
    papers = collector.collect(
        pubmed_query=queries["pubmed_query"],
        openalex_query=queries["openalex_query"],
        arxiv_query=queries["arxiv_query"],
        mdpi_query=queries["mdpi_query"],
        springer_query=queries["springer_query"],
        arxiv_categories=queries["arxiv_categories"],
        max_per_source=MAX_PER_SOURCE,
    )
    print(f"  -> {len(papers)} unique papers after deduplication")

    if not papers:
        print(f"  No papers found, skipping {qid}")
        return None

    # --- 4. Index ---
    print("\n[4/6] Indexing into Qdrant...")
    indexer = QdrantIndexer(collection_suffix=suffix)
    indexer.index_papers(papers)

    # --- 5. Screen ---
    print("\n[5/6] Screening abstracts...")
    screener = AbstractScreener(indexer=indexer, criteria=criteria_text)
    results = screener.screen_all(papers)
    n_included = sum(1 for _, r in results if r.decision == ScreeningStatus.INCLUDED)
    n_excluded = sum(1 for _, r in results if r.decision == ScreeningStatus.EXCLUDED)
    print(f"  -> {n_included} included, {n_excluded} excluded")

    # --- 6. Extract ---
    print("\n[6/6] Extracting structured data from included papers...")
    extractor = FullTextExtractor(indexer=indexer, collection_suffix=suffix)
    extractions = extractor.extract_included()
    n_fulltext = sum(1 for e in extractions if e.extraction_source == "fulltext")
    n_abstract = sum(1 for e in extractions if e.extraction_source == "abstract")
    print(f"  -> {n_fulltext} from full text, {n_abstract} from abstract")

    # --- Save run config ---
    run_dir = os.path.join("data", "runs", qid)
    os.makedirs(run_dir, exist_ok=True)
    config = {
        "question_id": qid,
        "category": question.get("category"),
        "question": qtext,
        "timestamp": datetime.datetime.now().isoformat(),
        "max_per_source": MAX_PER_SOURCE,
        "queries": queries,
        "criteria": criteria_dict,
        "results": {
            "collected_after_dedup": len(papers),
            "included": n_included,
            "excluded": n_excluded,
            "extracted_fulltext": n_fulltext,
            "extracted_abstract": n_abstract,
        },
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  Run config saved: {config_path}")

    return config


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python run_pipeline.py <question_id|all> [...]")
        print("\nAvailable questions:")
        for q in QUESTIONS:
            print(f"  {q['id']:20} ({q['category']}) — {q['text'][:60]}...")
        sys.exit(1)

    if args == ["all"]:
        selected = QUESTIONS
    else:
        selected = [q for q in QUESTIONS if q["id"] in args]
        if not selected:
            print(f"No questions matched: {args}")
            sys.exit(1)

    for q in selected:
        run_one(q)

    print(f"\n{'='*70}")
    print(f"Pipeline complete for {len(selected)} question(s).")
    print('='*70)
