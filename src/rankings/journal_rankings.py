"""
Lookup journal quartile (Q1-Q4) by ISSN using Scimago Journal Rank data.
Loads scimago.csv once at import time and caches a {issn: quartile} dict.
"""
import csv
import os

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "rankings", "scimago.csv",
)

_ISSN_TO_QUARTILE: dict[str, str] = {}


def normalize_issn(issn: str) -> str:
    """Strip dashes and whitespace, uppercase. '1542-4863' -> '15424863'."""
    if not issn:
        return ""
    return issn.replace("-", "").replace(" ", "").upper().strip()


def _load():
    """Parse scimago.csv into {issn: quartile} once."""
    if _ISSN_TO_QUARTILE:
        return
    if not os.path.exists(CSV_PATH):
        print(f"[Rankings] Scimago CSV not found at {CSV_PATH} — quartiles disabled")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            quartile = (row.get("SJR Best Quartile") or "").strip()
            if not quartile or quartile == "-":
                continue
            issn_field = row.get("Issn", "")
            # Field can be a single ISSN or multiple comma-separated
            for issn in issn_field.split(","):
                norm = normalize_issn(issn)
                if norm and norm not in _ISSN_TO_QUARTILE:
                    _ISSN_TO_QUARTILE[norm] = quartile

    print(f"[Rankings] Loaded {len(_ISSN_TO_QUARTILE)} journal ISSN -> quartile mappings")


def lookup_quartile(issn: str) -> str | None:
    """Return 'Q1' / 'Q2' / 'Q3' / 'Q4' or None if not found."""
    _load()
    if not issn:
        return None
    return _ISSN_TO_QUARTILE.get(normalize_issn(issn))
