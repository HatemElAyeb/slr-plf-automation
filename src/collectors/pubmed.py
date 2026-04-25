import requests
import time
from xml.etree import ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper


class PubMedCollector:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str):
        self.email = email

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Return list of PMIDs for a query."""
        resp = requests.get(
            f"{self.BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": self.email,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["esearchresult"]["idlist"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _efetch(self, pmids: list[str]) -> list[Paper]:
        """Fetch abstracts + metadata for a batch of PMIDs."""
        resp = requests.get(
            f"{self.BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": self.email,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return self._parse_xml(resp.text)

    def _parse_xml(self, xml_text: str) -> list[Paper]:
        papers = []
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID", "")
                title = article.findtext(".//ArticleTitle", "").strip()
                abstract = " ".join(
                    t.text or ""
                    for t in article.findall(".//AbstractText")
                ).strip()

                if not title or not abstract:
                    continue

                # Authors
                authors = [
                    f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip()
                    for a in article.findall(".//Author")
                ]

                # Year
                year_text = article.findtext(".//PubDate/Year")
                year = int(year_text) if year_text and year_text.isdigit() else None

                # DOI — required, skip paper if missing
                doi = None
                for id_el in article.findall(".//ArticleId"):
                    if id_el.get("IdType") == "doi":
                        doi = id_el.text
                        break
                if not doi:
                    continue

                papers.append(
                    Paper(
                        id=f"pubmed_{pmid}",
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        doi=doi,
                        source="pubmed",
                        pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    )
                )
            except Exception:
                continue

        return papers

    def search(self, query: str, max_results: int = 500) -> list[Paper]:
        print(f"[PubMed] Searching: {query[:80]}...")
        pmids = self._esearch(query, max_results)
        if not pmids:
            return []

        papers = []
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            papers.extend(self._efetch(batch))
            time.sleep(0.4)  # NCBI rate limit: max 3 req/s without API key

        print(f"[PubMed] Retrieved {len(papers)} papers")
        return papers
