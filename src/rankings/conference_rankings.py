"""
Lookup CORE conference rank (A*, A, B, C) by acronym.
Loads core.csv once at import time and caches an {acronym: rank} dict.
"""
import csv
import os

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "rankings", "core.csv",
)

# CORE CSV has no header — column positions are fixed:
#   0: ID, 1: Title, 2: Acronym, 3: Source, 4: Rank, 5: ..., 6+: FoR codes
COL_ACRONYM = 2
COL_RANK = 4

VALID_RANKS = {"A*", "A", "B", "C", "Australasian B", "Australasian C"}

_ACRONYM_TO_RANK: dict[str, str] = {}


def normalize_acronym(acronym: str) -> str:
    """Uppercase and strip whitespace. Used for case-insensitive lookups."""
    if not acronym:
        return ""
    return acronym.strip().upper()


def _load():
    if _ACRONYM_TO_RANK:
        return
    if not os.path.exists(CSV_PATH):
        print(f"[Rankings] CORE CSV not found at {CSV_PATH} — conference ranks disabled")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= COL_RANK:
                continue
            acronym = normalize_acronym(row[COL_ACRONYM])
            rank = row[COL_RANK].strip()
            if acronym and rank in VALID_RANKS:
                _ACRONYM_TO_RANK[acronym] = rank

    print(f"[Rankings] Loaded {len(_ACRONYM_TO_RANK)} conference acronym -> rank mappings")


def lookup_conference_rank(acronym: str) -> str | None:
    """Return 'A*' / 'A' / 'B' / 'C' / 'Australasian B|C' or None if not found."""
    _load()
    if not acronym:
        return None
    return _ACRONYM_TO_RANK.get(normalize_acronym(acronym))
