import json
import re
import hashlib

import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.models import Paper, ExtractionResult, ScreeningStatus
from src.indexer.indexer import QdrantIndexer
from src.extraction.pdf_downloader import download_pdf
from src.llm import get_llm
from config.settings import settings

CHUNK_SIZE = 1000       # words per chunk
CHUNK_OVERLAP = 200     # overlapping words between chunks
TOP_K_CHUNKS = 8        # chunks to retrieve per extraction query

EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are extracting structured data from a scientific paper about Precision Livestock Farming.

Based on the excerpts below, extract the following fields:
- animal_species: list of animal species studied (e.g. ["dairy cattle", "pigs"])
- sensor_types: list of sensors or technologies used (e.g. ["accelerometer", "RFID", "camera"])
- ml_methods: list of AI/ML techniques used (e.g. ["SVM", "LSTM", "Random Forest"])
- performance_metrics: dict of metric name to value (e.g. {{"accuracy": "91%", "F1": "0.88"}})
- dataset_size: number of animals or samples as a string (e.g. "120 cows", "5000 samples")
- key_findings: 2-3 sentence summary of the main results

Paper excerpts:
{context}

Return a valid JSON object with exactly these keys. If a field cannot be determined from the text, use an empty list or empty string. Do NOT guess or invent values — only extract what is explicitly stated in the excerpts.
""")


def _is_references_chunk(chunk: str) -> bool:
    """Return True if chunk is mostly a references/bibliography section."""
    lines = chunk.strip().splitlines()
    if not lines:
        return False
    # Count lines that look like citations
    citation_lines = sum(
        1 for l in lines
        if l.strip().startswith("[") or
           (len(l.strip()) > 0 and l.strip()[0].isdigit() and ". " in l[:10])
    )
    return citation_lines / max(len(lines), 1) > 0.4


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    # Only use the first 80% of the document (skip references at the end)
    words = words[:int(len(words) * 0.8)]
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip() and not _is_references_chunk(chunk):
            chunks.append(chunk)
    return chunks


def _extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


class FullTextExtractor:
    def __init__(self, indexer: QdrantIndexer | None = None):
        self.indexer = indexer or QdrantIndexer()
        self.embedder = self.indexer.embedder

        self.ft_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._ensure_fulltext_collection()

        self.llm = get_llm(temperature=0, json_mode=True)
        self.chain = EXTRACTION_PROMPT | self.llm | StrOutputParser()

    def _ensure_fulltext_collection(self):
        existing = [c.name for c in self.ft_client.get_collections().collections]
        if settings.fulltext_collection not in existing:
            self.ft_client.create_collection(
                collection_name=settings.fulltext_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Extractor] Created collection: {settings.fulltext_collection}")

    def _index_chunks(self, paper_id: str, chunks: list[str]):
        vectors = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        points = [
            PointStruct(
                id=int(hashlib.md5(f"{paper_id}_chunk_{i}".encode()).hexdigest(), 16) % (2**63),
                vector=vector,
                payload={"paper_id": paper_id, "text": chunk, "chunk_index": i},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.ft_client.upsert(collection_name=settings.fulltext_collection, points=points)

    def _retrieve_chunks(self, paper_id: str, query: str) -> list[str]:
        query_vector = self.embedder.encode([query], show_progress_bar=False)[0].tolist()
        results = self.ft_client.search(
            collection_name=settings.fulltext_collection,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=TOP_K_CHUNKS,
            with_payload=True,
        )
        return [r.payload["text"] for r in results]

    def _parse_extraction(self, raw: str, paper_id: str, source: str = "fulltext") -> ExtractionResult:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group() if match else raw)
            return ExtractionResult(
                paper_id=paper_id,
                animal_species=data.get("animal_species", []),
                sensor_types=data.get("sensor_types", []),
                ml_methods=data.get("ml_methods", []),
                performance_metrics=data.get("performance_metrics", {}),
                dataset_size=data.get("dataset_size", ""),
                key_findings=data.get("key_findings", ""),
                extraction_source=source,
            )
        except Exception:
            return ExtractionResult(paper_id=paper_id, extraction_source=source)

    def _extract_from_fulltext(self, paper: Paper, pdf_path: str) -> ExtractionResult | None:
        """Run RAG extraction on the PDF text. Returns None if parsing fails."""
        try:
            text = _extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  [PDF] Failed to parse {paper.id}: {e}")
            return None
        if not text.strip():
            return None

        chunks = _chunk_text(text)
        if not chunks:
            return None

        self._index_chunks(paper.id, chunks)

        query = "sensors used, animal species, machine learning methods, performance metrics, dataset size, key findings"
        context_chunks = self._retrieve_chunks(paper.id, query)
        context = "\n\n---\n\n".join(context_chunks)[:6000]

        raw = self.chain.invoke({"context": context})
        return self._parse_extraction(raw, paper.id, source="fulltext")

    def _extract_from_abstract(self, paper: Paper) -> ExtractionResult:
        """Fallback extraction using only the title + abstract."""
        context = f"TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}"
        raw = self.chain.invoke({"context": context})
        return self._parse_extraction(raw, paper.id, source="abstract")

    def extract_paper(self, paper: Paper) -> ExtractionResult:
        """
        Hybrid extraction:
          1. Try to download and extract from full PDF
          2. If PDF unavailable or parsing fails, fall back to abstract-only
        Always returns an ExtractionResult.
        """
        pdf_path = download_pdf(paper)
        if pdf_path:
            result = self._extract_from_fulltext(paper, pdf_path)
            if result is not None:
                return result
            print(f"  [Fallback] Full-text extraction failed for {paper.id} — using abstract")

        return self._extract_from_abstract(paper)

    def extract_included(self) -> list[ExtractionResult]:
        """Extract data from all included papers in Qdrant."""
        points = self.indexer.get_points_by_status(ScreeningStatus.INCLUDED)
        papers = [self.indexer._point_to_paper(p) for p in points]
        print(f"[Extractor] Found {len(papers)} included papers to extract")

        results = []
        for paper in tqdm(papers, desc="Extracting"):
            result = self.extract_paper(paper)
            if result:
                results.append(result)
        return results
