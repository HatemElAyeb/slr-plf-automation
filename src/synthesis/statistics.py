"""
Compute SLR statistics for a question's Qdrant collection.
No LLM calls — pure Python aggregation over stored payloads.
"""
from collections import Counter
from src.indexer.indexer import QdrantIndexer
from src.models import ScreeningStatus


def compute_statistics(question_id: str) -> dict:
    """
    Aggregate stats across the 'plf_abstracts_{question_id}' collection.
    Returns a dict with PRISMA counts, source/quartile/year distributions,
    and frequency counts of extracted fields.
    """
    suffix = "_" + question_id
    ix = QdrantIndexer(collection_suffix=suffix)

    # Pull every paper's payload once
    points = ix.client.scroll(
        collection_name=ix.collection_name,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )[0]
    payloads = [p.payload for p in points]

    total = len(payloads)
    included = [p for p in payloads if p.get("screening_status") == ScreeningStatus.INCLUDED.value]
    excluded = [p for p in payloads if p.get("screening_status") == ScreeningStatus.EXCLUDED.value]
    pending  = [p for p in payloads if p.get("screening_status") == ScreeningStatus.PENDING.value]

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
        "identified":            total,                  # after dedup at collection time
        "screened":              total - len(pending),
        "included":              len(included),
        "excluded":              len(excluded),
        "excluded_genuine":      len(excluded) - len(failed_screening),
        "screening_failed":      len(failed_screening),
        "extracted_total":       len(extracted),
        "extracted_fulltext":    len(extracted_fulltext),
        "extracted_abstract":    len(extracted_abstract),
        "missing_pdfs":          len(extracted_abstract),
    }

    # Source distribution (across included)
    source_dist = Counter(p.get("source", "unknown") for p in included)

    # Quartile distribution (journals only — conferences have None)
    quartile_dist = Counter(p.get("quartile") or "unranked" for p in included)

    # Conference rank distribution
    conf_rank_dist = Counter(
        p.get("conference_rank") or "unranked"
        for p in included if p.get("is_conference")
    )

    # Conference vs journal split
    venue_type = Counter(
        "conference" if p.get("is_conference") else "journal"
        for p in included
    )

    # Year distribution
    year_dist = Counter(p.get("year") for p in included if p.get("year"))

    # Top extracted fields (frequency counts across extracted papers)
    species_counter = Counter()
    sensor_counter = Counter()
    method_counter = Counter()
    for p in extracted:
        for s in p.get("animal_species") or []:
            species_counter[s.lower().strip()] += 1
        for s in p.get("sensor_types") or []:
            sensor_counter[s.lower().strip()] += 1
        for m in p.get("ml_methods") or []:
            method_counter[m.lower().strip()] += 1

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
        "included_papers":          included,        # full payloads for the report
        "extracted_papers":         extracted,
    }
