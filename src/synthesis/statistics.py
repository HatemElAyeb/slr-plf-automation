"""
Compute SLR statistics for a question's Qdrant collection.
No LLM calls — pure Python aggregation over stored payloads.
Applies normalization (data/normalization_map.json) and a year cutoff.
"""
import json
import os
from collections import Counter
from src.indexer.indexer import QdrantIndexer
from src.models import ScreeningStatus


YEAR_CUTOFF = 2015
NORMALIZATION_PATH = os.path.join("data", "normalization_map.json")
NON_LIVESTOCK_TAG = "_non_livestock_"

# Reverse-lookup tables loaded once
_NORMALIZATION: dict[str, dict[str, str]] = {}


def _load_normalization() -> dict[str, dict[str, str]]:
    """Load and invert normalization_map.json into {category: {variant: canonical}}."""
    global _NORMALIZATION
    if _NORMALIZATION:
        return _NORMALIZATION
    if not os.path.exists(NORMALIZATION_PATH):
        return {}
    with open(NORMALIZATION_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for category, groups in raw.items():
        lookup: dict[str, str] = {}
        for canonical, variants in groups.items():
            for v in variants:
                lookup[v.lower().strip()] = canonical.lower().strip()
        out[category] = lookup
    _NORMALIZATION = out
    return out


def _normalize(value: str, category: str) -> str | None:
    """
    Map a raw value to its canonical form.
    Returns None if the canonical name is _non_livestock_ (so the caller drops it).
    Falls back to the raw value if no mapping exists.
    """
    if not value:
        return None
    lookup = _load_normalization().get(category, {})
    canonical = lookup.get(value.lower().strip(), value.lower().strip())
    if canonical == NON_LIVESTOCK_TAG:
        return None
    return canonical


def compute_statistics(question_id: str) -> dict:
    """
    Aggregate stats across the 'plf_abstracts_{question_id}' collection.
    Returns a dict with PRISMA counts, source/quartile/year distributions,
    and frequency counts of normalized extracted fields.
    """
    suffix = "_" + question_id
    ix = QdrantIndexer(collection_suffix=suffix)

    points = ix.client.scroll(
        collection_name=ix.collection_name,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )[0]
    payloads = [p.payload for p in points]

    total = len(payloads)
    included_all = [p for p in payloads if p.get("screening_status") == ScreeningStatus.INCLUDED.value]
    excluded = [p for p in payloads if p.get("screening_status") == ScreeningStatus.EXCLUDED.value]
    pending  = [p for p in payloads if p.get("screening_status") == ScreeningStatus.PENDING.value]

    # Year filter: drop papers with year < 2015 from included corpus
    included = [p for p in included_all if (p.get("year") or 0) >= YEAR_CUTOFF]
    dropped_old = len(included_all) - len(included)

    # Distinguish genuine exclusions from rate-limit failures
    failed_screening = [p for p in excluded if "LLM error" in (p.get("screening_reason") or "")]

    # Extraction tracking: only papers with at least one field filled
    extracted = []
    for p in included:
        if p.get("extraction_source") and (
            p.get("animal_species") or p.get("sensor_types") or p.get("ml_methods")
            or p.get("key_findings") or p.get("dataset_size")
        ):
            extracted.append(p)
    extracted_fulltext = [p for p in extracted if p.get("extraction_source") == "fulltext"]
    extracted_abstract = [p for p in extracted if p.get("extraction_source") == "abstract"]

    # PRISMA counts
    prisma = {
        "identified":            total,
        "screened":              total - len(pending),
        "included":              len(included),
        "included_pre_year_filter": len(included_all),
        "dropped_pre_2015":      dropped_old,
        "excluded":              len(excluded),
        "excluded_genuine":      len(excluded) - len(failed_screening),
        "screening_failed":      len(failed_screening),
        "extracted_total":       len(extracted),
        "extracted_fulltext":    len(extracted_fulltext),
        "extracted_abstract":    len(extracted_abstract),
        "missing_pdfs":          len(extracted_abstract),
    }

    source_dist  = Counter(p.get("source", "unknown") for p in included)
    quartile_dist = Counter(p.get("quartile") or "unranked" for p in included)
    conf_rank_dist = Counter(
        p.get("conference_rank") or "unranked"
        for p in included if p.get("is_conference")
    )
    venue_type = Counter(
        "conference" if p.get("is_conference") else "journal"
        for p in included
    )
    year_dist = Counter(p.get("year") for p in included if p.get("year"))

    # Normalized frequency counters
    species_counter = Counter()
    sensor_counter = Counter()
    method_counter = Counter()
    for p in extracted:
        for s in p.get("animal_species") or []:
            n = _normalize(s, "animal_species")
            if n:
                species_counter[n] += 1
        for s in p.get("sensor_types") or []:
            n = _normalize(s, "sensor_types")
            if n:
                sensor_counter[n] += 1
        for m in p.get("ml_methods") or []:
            n = _normalize(m, "ml_methods")
            if n:
                method_counter[n] += 1

    return {
        "prisma": prisma,
        "source_distribution":      dict(source_dist.most_common()),
        "quartile_distribution":    dict(quartile_dist.most_common()),
        "conference_rank_distribution": dict(conf_rank_dist.most_common()),
        "venue_type_split":         dict(venue_type.most_common()),
        "year_distribution":        dict(sorted(year_dist.items())),
        "top_animal_species":       dict(species_counter.most_common(15)),
        "top_sensor_types":         dict(sensor_counter.most_common(15)),
        "top_ml_methods":           dict(method_counter.most_common(15)),
        "included_papers":          included,
        "extracted_papers":         extracted,
    }
