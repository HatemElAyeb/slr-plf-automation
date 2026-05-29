"""
Smoke test for Module 0 — query builder.
Run: python test_query_builder.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.query_builder import build_queries

QUESTIONS = [
    "What deep learning methods are used for dairy cattle lameness detection?",
    "Which sensors are used for poultry health monitoring?",
    "How is computer vision applied to livestock behavior recognition?",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print('='*70)
        queries = build_queries(q)
        print(json.dumps(queries, indent=2))
