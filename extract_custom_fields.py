"""
Run question-specific custom field extraction across all (or selected) questions.

Usage:
  python extract_custom_fields.py            # all 6 questions
  python extract_custom_fields.py q5_gaps    # one question
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from research_questions import QUESTIONS
from src.extraction.custom_extractor import extract_custom_for_question


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        selected = [q for q in QUESTIONS if q["id"] in args]
    else:
        selected = QUESTIONS

    if not selected:
        print(f"No questions matched: {args}")
        sys.exit(1)

    total = 0
    for q in selected:
        print(f"\n=== {q['id']} ({q['category']}) ===")
        try:
            total += extract_custom_for_question(q)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\nTotal papers processed: {total}")
