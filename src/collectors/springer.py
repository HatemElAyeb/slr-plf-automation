import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper
from src.rankings import lookup_quartile, lookup_conference_rank
from config.settings import settings

META_URL = "https://api.springernature.com/meta/v2/json"


class SpringerCollector:
    """
    Collector for Springer papers via the Springer Nature Meta API.

    Requires SPRINGER_META_API_KEY in .env (free at dev.springernature.com).
    Returns metadata + abstracts of Springer papers (open and closed access).
    PDFs are paywalled — use Unpaywall fallback in Module 4 for free OA copies.
    """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _fetch(self, query: str, rows: int, start: int) -> dict:
        params = {
            "q": query,
            "p": rows,
            "s": start,
            "api_key": settings.springer_meta_api_key,
        }
        resp = requests.get(META_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _parse_records(self, records: list) -> list[Paper]:
        papers = []
        for rec in records:
            try:
                doi = rec.get("doi", "")
                title = (rec.get("title") or "").strip()
                abstract = (rec.get("abstract") or "").strip()

                if not title or not abstract or not doi:
                    continue

                authors = [
                    c.get("creator", "")
                    for c in rec.get("creators", [])
                    if c.get("creator")
                ]

                year = None
                date = rec.get("publicationDate", "")
                if date and len(date) >= 4 and date[:4].isdigit():
                    year = int(date[:4])

                # Springer URL — may be paywalled
                pdf_url = None
                for url in rec.get("url", []):
                    if url.get("format") == "pdf":
                        pdf_url = url.get("value")
                        break
                if not pdf_url and doi:
                    pdf_url = f"https://link.springer.com/content/pdf/{doi}.pdf"

                paper_id = f"springer_{doi.replace('/', '_')}" if doi else f"springer_{title[:40]}"

                # Venue info (Springer publishes both journals and conference proceedings)
                venue_name = rec.get("publicationName")
                venue_issn = rec.get("issn") or rec.get("electronicIssn")
                content_type = (rec.get("contentType") or "").lower()
                is_conference = "conference" in content_type or "proceedings" in content_type

                conference_acronym = None
                quartile = None
                conference_rank = None

                if is_conference and venue_name:
                    import re
                    paren_match = re.search(r"\(([A-Z][A-Z0-9\-]{1,15})\)", venue_name)
                    if paren_match:
                        conference_acronym = paren_match.group(1)
                        conference_rank = lookup_conference_rank(conference_acronym)
                elif venue_issn:
                    quartile = lookup_quartile(venue_issn)

                papers.append(
                    Paper(
                        id=paper_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        doi=doi or None,
                        source="springer",
                        pdf_url=pdf_url,
                        venue_name=venue_name,
                        venue_issn=venue_issn,
                        is_conference=is_conference,
                        conference_acronym=conference_acronym,
                        quartile=quartile,
                        conference_rank=conference_rank,
                    )
                )
            except Exception:
                continue
        return papers

    def search(self, query: str, max_results: int = 200) -> list[Paper]:
        if not settings.springer_meta_api_key:
            print("[Springer] No API key configured — skipping")
            return []

        print(f"[Springer] Searching: {query[:80]}...")
        papers = []
        page_size = 50

        for start in range(1, max_results + 1, page_size):
            data = self._fetch(query, rows=min(page_size, max_results - len(papers)), start=start)
            records = data.get("records", [])
            if not records:
                break
            papers.extend(self._parse_records(records))

        print(f"[Springer] Retrieved {len(papers)} papers")
        return papers
