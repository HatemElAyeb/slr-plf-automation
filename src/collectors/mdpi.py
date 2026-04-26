import re
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper
from src.rankings import lookup_quartile
from config.settings import settings

CROSSREF_URL = "https://api.crossref.org/works"
MDPI_MEMBER_ID = "1968"


class MDPICollector:
    """
    Collector for MDPI papers via the CrossRef API.

    MDPI has no public API, but all MDPI papers are indexed in CrossRef.
    Filtering by member ID 1968 returns only MDPI publications.
    All MDPI papers are open access — PDF download will always work.
    """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _fetch(self, query: str, rows: int) -> dict:
        params = {
            "query": query,
            "filter": f"member:{MDPI_MEMBER_ID},type:journal-article",
            "rows": rows,
            "select": "DOI,title,abstract,author,published,container-title,link,ISSN",
            "mailto": settings.openalex_email,
        }
        resp = requests.get(CROSSREF_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _strip_jats(self, text: str) -> str:
        """Remove JATS XML tags from CrossRef abstracts."""
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    def _parse_items(self, items: list) -> list[Paper]:
        papers = []
        for item in items:
            try:
                doi = item.get("DOI", "")
                if not doi:
                    continue

                title_list = item.get("title", [])
                title = title_list[0].strip() if title_list else ""
                abstract = self._strip_jats(item.get("abstract", ""))

                if not title or not abstract:
                    continue

                authors = []
                for a in item.get("author", []):
                    name = f"{a.get('family', '')} {a.get('given', '')}".strip()
                    if name:
                        authors.append(name)

                year = None
                pub = item.get("published", {}).get("date-parts", [[None]])
                if pub and pub[0] and pub[0][0]:
                    year = int(pub[0][0])

                pdf_url = None
                for link in item.get("link", []):
                    if link.get("content-type") == "application/pdf":
                        pdf_url = link.get("URL")
                        break
                if not pdf_url:
                    pdf_url = f"https://www.mdpi.com/{doi}/pdf"

                # Journal info
                container = item.get("container-title", [])
                venue_name = container[0] if container else None
                issns = item.get("ISSN", [])
                venue_issn = issns[0] if issns else None
                quartile = lookup_quartile(venue_issn) if venue_issn else None

                papers.append(
                    Paper(
                        id=f"mdpi_{doi.replace('/', '_')}",
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        doi=doi,
                        source="mdpi",
                        pdf_url=pdf_url,
                        venue_name=venue_name,
                        venue_issn=venue_issn,
                        quartile=quartile,
                    )
                )
            except Exception:
                continue
        return papers

    def search(self, query: str, max_results: int = 200) -> list[Paper]:
        print(f"[MDPI] Searching: {query[:80]}...")
        data = self._fetch(query, rows=max_results)
        items = data.get("message", {}).get("items", [])
        papers = self._parse_items(items)
        print(f"[MDPI] Retrieved {len(papers)} papers")
        return papers
