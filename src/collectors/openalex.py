import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper
from src.rankings import lookup_quartile, lookup_conference_rank, lookup_venue_via_crossref
from config.settings import settings


class OpenAlexCollector:
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, email: str):
        # Using email puts you in the "polite pool" — faster responses
        self.headers = {"User-Agent": f"mailto:{email}"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _fetch_page(self, params: dict) -> dict:
        resp = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _parse_work(self, work: dict) -> Paper | None:
        title = (work.get("title") or "").strip()
        if not title:
            return None

        # Abstract is stored as inverted index — reconstruct it
        abstract = ""
        inv_index = work.get("abstract_inverted_index")
        if inv_index:
            positions = {}
            for word, pos_list in inv_index.items():
                for pos in pos_list:
                    positions[pos] = word
            abstract = " ".join(positions[k] for k in sorted(positions))

        if not abstract:
            return None

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]

        year = work.get("publication_year")
        doi = work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None
        if not doi:
            return None

        # Best open-access PDF
        pdf_url = None
        oa = work.get("open_access", {})
        if oa.get("is_oa"):
            pdf_url = oa.get("oa_url")

        openalex_id = work.get("id", "").split("/")[-1]

        # Venue info — primary_location.source contains journal or conference info
        venue_name = None
        venue_issn = None
        is_conference = False
        conference_acronym = None
        quartile = None
        conference_rank = None

        primary_loc = work.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        if source:
            venue_name = source.get("display_name")
            issn_l = source.get("issn_l")
            issns = source.get("issn") or []
            venue_issn = issn_l or (issns[0] if issns else None)
            source_type = (source.get("type") or "").lower()
            is_conference = source_type == "conference"

        # Fallback: if OpenAlex has no venue but we have a DOI, ask CrossRef
        if not venue_name and doi:
            cr_name, cr_issn = lookup_venue_via_crossref(doi, settings.openalex_email)
            venue_name = venue_name or cr_name
            venue_issn = venue_issn or cr_issn

        # Detect conference from venue name patterns (when OpenAlex didn't tag it)
        if venue_name and not is_conference:
            name_l = venue_name.lower()
            if any(kw in name_l for kw in ["conference", "workshop", "symposium",
                                           "proceedings", "ieee sensors"]):
                is_conference = True

        if is_conference and venue_name:
            # Try to extract acronym from venue name (often in parentheses or all-caps)
            import re
            paren_match = re.search(r"\(([A-Z][A-Z0-9\-]{1,15})\)", venue_name)
            if paren_match:
                conference_acronym = paren_match.group(1)
                conference_rank = lookup_conference_rank(conference_acronym)
        elif venue_issn:
            quartile = lookup_quartile(venue_issn)

        return Paper(
            id=f"openalex_{openalex_id}",
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            doi=doi,
            source="openalex",
            pdf_url=pdf_url,
            venue_name=venue_name,
            venue_issn=venue_issn,
            is_conference=is_conference,
            conference_acronym=conference_acronym,
            quartile=quartile,
            conference_rank=conference_rank,
        )

    def search(self, query: str, max_results: int = 500) -> list[Paper]:
        print(f"[OpenAlex] Searching: {query[:80]}...")
        papers = []
        per_page = 50
        cursor = "*"

        while len(papers) < max_results:
            params = {
                "search": query,
                "per-page": per_page,
                "cursor": cursor,
                "filter": "type:article|proceedings-article",  # journal articles + conference papers
                "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,open_access,primary_location",
            }

            data = self._fetch_page(params)
            results = data.get("results", [])
            if not results:
                break

            for work in results:
                paper = self._parse_work(work)
                if paper:
                    papers.append(paper)
                if len(papers) >= max_results:
                    break

            # Pagination
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

        print(f"[OpenAlex] Retrieved {len(papers)} papers")
        return papers
