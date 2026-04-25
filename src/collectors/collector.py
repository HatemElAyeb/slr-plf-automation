from .pubmed import PubMedCollector
from .openalex import OpenAlexCollector
from .arxiv import ArXivCollector
from .mdpi import MDPICollector
from .springer import SpringerCollector
from src.models import Paper
from config.settings import settings


class LiteratureCollector:
    def __init__(self):
        self.pubmed = PubMedCollector(email=settings.pubmed_email)
        self.openalex = OpenAlexCollector(email=settings.openalex_email)
        self.arxiv = ArXivCollector()
        self.mdpi = MDPICollector()
        self.springer = SpringerCollector()

    def collect(
        self,
        pubmed_query: str,
        openalex_query: str,
        arxiv_query: str,
        mdpi_query: str | None = None,
        springer_query: str | None = None,
        arxiv_categories: list[str] | None = None,
        max_per_source: int = 300,
    ) -> list[Paper]:
        all_papers: list[Paper] = []

        all_papers.extend(self.pubmed.search(pubmed_query, max_per_source))
        all_papers.extend(self.openalex.search(openalex_query, max_per_source))
        all_papers.extend(
            self.arxiv.search(arxiv_query, arxiv_categories, max_per_source)
        )
        all_papers.extend(self.mdpi.search(mdpi_query or openalex_query, max_per_source))
        all_papers.extend(self.springer.search(springer_query or openalex_query, max_per_source))

        deduplicated = self._deduplicate(all_papers)
        print(f"\n[Collector] Total after deduplication: {len(deduplicated)} papers")
        return deduplicated

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Deduplicate by DOI first, then by normalised title."""
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for p in papers:
            if p.doi:
                doi_key = p.doi.lower().strip()
                if doi_key in seen_dois:
                    continue
                seen_dois.add(doi_key)

            title_key = "".join(c for c in p.title.lower() if c.isalnum())
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            unique.append(p)

        return unique
