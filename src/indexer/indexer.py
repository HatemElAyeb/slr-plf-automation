import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.models import Paper, ScreeningStatus
from config.settings import settings


class QdrantIndexer:
    def __init__(self, collection_suffix: str = ""):
        """
        collection_suffix: optional suffix for namespacing collections per question.
            "" -> "plf_abstracts" (default, backward compatible)
            "_q1" -> "plf_abstracts_q1"
        """
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.abstracts_collection + collection_suffix
        print(f"[Indexer] Loading embedding model: {settings.embedding_model}")
        self.embedder = SentenceTransformer(settings.embedding_model)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the abstracts collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Indexer] Created collection: {self.collection_name}")
        else:
            print(f"[Indexer] Collection already exists: {self.collection_name}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def index_papers(self, papers: list[Paper], batch_size: int = 64) -> int:
        """Embed and upsert papers into Qdrant. Returns number of papers indexed."""
        if not papers:
            return 0

        indexed = 0
        for i in tqdm(range(0, len(papers), batch_size), desc="Indexing"):
            batch = papers[i : i + batch_size]

            texts = [f"{p.title}. {p.abstract}" for p in batch]
            vectors = self._embed(texts)

            points = [
                PointStruct(
                    id=int(hashlib.md5(p.id.encode()).hexdigest(), 16) % (2**63),
                    vector=vector,
                    payload={
                        "paper_id": p.id,
                        "title": p.title,
                        "abstract": p.abstract,
                        "authors": p.authors,
                        "year": p.year,
                        "doi": p.doi,
                        "source": p.source,
                        "pdf_url": p.pdf_url,
                        "venue_name": p.venue_name,
                        "venue_issn": p.venue_issn,
                        "is_conference": p.is_conference,
                        "conference_acronym": p.conference_acronym,
                        "quartile": p.quartile,
                        "conference_rank": p.conference_rank,
                        "screening_status": ScreeningStatus.PENDING.value,
                        "screening_confidence": None,
                        "screening_reason": None,
                    },
                )
                for p, vector in zip(batch, vectors)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            indexed += len(batch)

        print(f"[Indexer] Indexed {indexed} papers into '{self.collection_name}'")
        return indexed

    def update_screening(self, paper_id: str, status: str, confidence: float, reason: str):
        """Update the screening result for a paper by paper_id."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=1,
        )[0]

        if not results:
            return

        point_id = results[0].id
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "screening_status": status,
                "screening_confidence": confidence,
                "screening_reason": reason,
            },
            points=[point_id],
        )

    def get_papers_by_status(self, status: ScreeningStatus) -> list[dict]:
        """Retrieve all papers with a given screening status."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="screening_status",
                        match=MatchValue(value=status.value),
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in results]

    def _point_to_paper(self, point) -> Paper:
        """Convert a Qdrant point back into a Paper object."""
        p = point.payload if hasattr(point, "payload") else point
        return Paper(
            id=p["paper_id"],
            title=p["title"],
            abstract=p["abstract"],
            authors=p.get("authors", []),
            year=p.get("year"),
            doi=p.get("doi"),
            source=p.get("source", ""),
            pdf_url=p.get("pdf_url"),
            venue_name=p.get("venue_name"),
            venue_issn=p.get("venue_issn"),
            is_conference=p.get("is_conference", False),
            conference_acronym=p.get("conference_acronym"),
            quartile=p.get("quartile"),
            conference_rank=p.get("conference_rank"),
            screening_status=ScreeningStatus(p.get("screening_status", "pending")),
        )

    def get_points_by_status(self, status: ScreeningStatus):
        """Return raw Qdrant points for a given screening status."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="screening_status",
                        match=MatchValue(value=status.value),
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return results

    def update_extraction(self, paper_id: str, extraction):
        """Persist extraction results to the paper's payload in Qdrant."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=1,
        )[0]

        if not results:
            return

        point_id = results[0].id
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "extraction_source": extraction.extraction_source,
                "animal_species": extraction.animal_species,
                "sensor_types": extraction.sensor_types,
                "ml_methods": extraction.ml_methods,
                "performance_metrics": extraction.performance_metrics,
                "dataset_size": extraction.dataset_size,
                "key_findings": extraction.key_findings,
            },
            points=[point_id],
        )

    def get_extracted_papers(self) -> list[dict]:
        """Return all papers that have extraction data (for Module 5 synthesis)."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="screening_status",
                        match=MatchValue(value=ScreeningStatus.INCLUDED.value),
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        # Only return papers that actually have extraction data
        return [r.payload for r in results if r.payload.get("extraction_source")]

    def count(self) -> int:
        return self.client.count(self.collection_name).count
