import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper


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

        # Best open-access PDF
        pdf_url = None
        oa = work.get("open_access", {})
        if oa.get("is_oa"):
            pdf_url = oa.get("oa_url")

        openalex_id = work.get("id", "").split("/")[-1]

        return Paper(
            id=f"openalex_{openalex_id}",
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            doi=doi,
            source="openalex",
            pdf_url=pdf_url,
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
                "filter": "type:article",  # journal articles only
                "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,open_access",
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
