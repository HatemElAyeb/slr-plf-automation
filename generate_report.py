"""
Generate SLR reports.

Usage:
  python generate_report.py q1_technical                # one per-question report
  python generate_report.py all                         # all 6 + master report
  python generate_report.py master                      # only the master report
  python generate_report.py q1_technical q5_gaps        # a subset
  python generate_report.py all --figures-only          # only regen the PNG
                                                          Sankey figures (no LLM)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.synthesis import generate_report, generate_master_report
from src.synthesis.statistics import compute_statistics
from src.synthesis.figures import generate_figures_for_question


def _run_figures_only(selected: list[dict]):
    for q in selected:
        if not q.get("sankey_diagrams"):
            print(f"\n[{q['id']}] no sankey_diagrams configured, skipping")
            continue
        print(f"\n[{q['id']}] regenerating {len(q['sankey_diagrams'])} figure(s)...")
        try:
            stats = compute_statistics(q["id"])
        except Exception as e:
            print(f"  Could not load stats: {type(e).__name__}: {e}")
            continue
        figs = generate_figures_for_question(q, stats["included_papers"])
        for f in figs:
            ok = "✓" if f.get("rel_path") else "✗"
            print(f"  {ok} {f['title']} -> {f.get('rel_path') or 'FAILED'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    figures_only = "--figures-only" in args
    if figures_only:
        args = [a for a in args if a != "--figures-only"]

    if not args:
        print("Usage: python generate_report.py <question_id|all|master> [--figures-only]")
        print("\nAvailable questions:")
        for q in QUESTIONS:
            print(f"  {q['id']:20} ({q['category']}) — {q['text'][:60]}...")
        print(f"  {'master':20} (project-wide intro + methodology + cross-question synthesis)")
        sys.exit(1)

    do_master = False
    if args == ["all"]:
        selected = QUESTIONS
        do_master = True
    elif args == ["master"]:
        selected = []
        do_master = True
    else:
        selected = [q for q in QUESTIONS if q["id"] in args]
        if "master" in args:
            do_master = True
        if not selected and not do_master:
            print(f"No questions matched: {args}")
            sys.exit(1)

    if figures_only:
        # Only regenerate Sankey PNGs. No LLM calls. No master report (it has no figures).
        if not selected:
            selected = QUESTIONS
        _run_figures_only(selected)
        print("\nDone (figures only).")
        sys.exit(0)

    for q in selected:
        try:
            print(f"\n{'='*70}\nGenerating per-question report for {q['id']}\n{'='*70}")
            generate_report(q["id"], q["text"], question=q)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

    if do_master:
        try:
            print(f"\n{'='*70}\nGenerating master report\n{'='*70}")
            generate_master_report()
        except Exception as e:
            print(f"  Master report FAILED: {type(e).__name__}: {e}")

    print("\nDone.")
