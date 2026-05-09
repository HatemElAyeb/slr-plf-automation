"""
Generate SLR reports.

Usage:
  python generate_report.py q1_technical            # one per-question report
  python generate_report.py all                     # all 6 + master report
  python generate_report.py master                  # only the master report
  python generate_report.py q1_technical q5_gaps    # a subset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.synthesis import generate_report, generate_master_report


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python generate_report.py <question_id|all|master> [...]")
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

    for q in selected:
        try:
            print(f"\n{'='*70}\nGenerating per-question report for {q['id']}\n{'='*70}")
            generate_report(q["id"], q["text"])
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
