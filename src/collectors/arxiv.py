import requests
import time
from xml.etree import ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

ARXIV_NS = "http://www.w3.org/2005/Atom"


class ArXivCollector:
    BASE_URL = "http://export.arxiv.org/api/query"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def _fetch(self, params: dict) -> str:
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
            arxiv_id_url = entry.findtext(f"{{{ARXIV_NS}}}id", "")
            arxiv_id = arxiv_id_url.split("/abs/")[-1].replace("/", "_")

            title = (entry.findtext(f"{{{ARXIV_NS}}}title") or "").strip().replace("\n", " ")
            abstract = (entry.findtext(f"{{{ARXIV_NS}}}summary") or "").strip().replace("\n", " ")

            if not title or not abstract:
                continue

            authors = [
                a.findtext(f"{{{ARXIV_NS}}}name", "")
                for a in entry.findall(f"{{{ARXIV_NS}}}author")
            ]

            published = entry.findtext(f"{{{ARXIV_NS}}}published", "")
            year = int(published[:4]) if published else None

            pdf_url = None
            for link in entry.findall(f"{{{ARXIV_NS}}}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            # ArXiv auto-registers DOIs in the format 10.48550/arXiv.<id>
            # Strip version suffix (v1, v2...) for DOI registration
            base_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id
            doi = f"10.48550/arXiv.{base_id}"

            papers.append(
                Paper(
                    id=f"arxiv_{arxiv_id}",
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    doi=doi,
                    source="arxiv",
                    pdf_url=pdf_url,
                )
            )

        return papers

    def search(
        self,
        query: str,
        categories: list[str] | None = None,
        max_results: int = 200,
    ) -> list[Paper]:
        """
        categories: e.g. ["cs.AI", "cs.CV", "eess.SP"]
        """
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        print(f"[ArXiv] Searching: {full_query[:80]}...")
        papers = []
        batch_size = 100

        for start in range(0, max_results, batch_size):
            params = {
                "search_query": full_query,
                "start": start,
                "max_results": min(batch_size, max_results - start),
            }
            xml_text = self._fetch(params)
            batch = self._parse_feed(xml_text)
            if not batch:
                break
            papers.extend(batch)
            time.sleep(3)  # ArXiv rate limit: 1 req/3s

        print(f"[ArXiv] Retrieved {len(papers)} papers")
        return papers
