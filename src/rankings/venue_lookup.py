"""
Fallback venue lookup via CrossRef API.
Used when an OpenAlex paper is missing venue info but has a DOI.
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

CROSSREF_WORK_URL = "https://api.crossref.org/works/{doi}"

# Cache to avoid duplicate lookups for the same DOI
_VENUE_CACHE: dict[str, tuple[str | None, str | None]] = {}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _fetch(doi: str, email: str) -> dict | None:
    try:
        resp = requests.get(
            CROSSREF_WORK_URL.format(doi=doi),
            params={"mailto": email},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("message")
    except Exception:
        return None


def lookup_venue_via_crossref(doi: str, email: str) -> tuple[str | None, str | None]:
    """
    Return (venue_name, venue_issn) for a DOI by querying CrossRef.
    Returns (None, None) if not found.
    """
    if not doi:
        return (None, None)
    if doi in _VENUE_CACHE:
        return _VENUE_CACHE[doi]

    data = _fetch(doi, email)
    if not data:
        _VENUE_CACHE[doi] = (None, None)
        return (None, None)

    container = data.get("container-title", [])
    venue_name = container[0] if container else None

    issns = data.get("ISSN", [])
    venue_issn = issns[0] if issns else None

    result = (venue_name, venue_issn)
    _VENUE_CACHE[doi] = result
    return result
