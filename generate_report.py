"""
Generate the SLR report for one or more research questions.

Usage:
  python generate_report.py q1_technical            # one report
  python generate_report.py all                     # all 6
  python generate_report.py q1_technical q5_gaps    # subset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.synthesis import generate_report


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python generate_report.py <question_id|all> [...]")
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
        try:
            print(f"\n{'='*70}\nGenerating report for {q['id']}\n{'='*70}")
            generate_report(q["id"], q["text"])
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

    print("\nDone.")
