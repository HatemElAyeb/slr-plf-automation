import hashlib

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.models import Paper, ExtractionResult, ScreeningStatus
from src.indexer.indexer import QdrantIndexer
from src.extraction.pdf_downloader import download_pdf
from src.llm import get_llm
from config.settings import settings


class _ExtractionSchema(BaseModel):
    """Schema for the LLM's structured output. Does NOT include paper_id or extraction_source —
    those are filled in by us, not the LLM."""
    animal_species: list[str] = Field(
        default_factory=list,
        description="Animal species studied, e.g. ['dairy cattle', 'pigs']",
    )
    sensor_types: list[str] = Field(
        default_factory=list,
        description="Sensors or technologies used, e.g. ['accelerometer', 'RFID', 'camera']",
    )
    ml_methods: list[str] = Field(
        default_factory=list,
        description="AI/ML techniques used, e.g. ['SVM', 'LSTM', 'Random Forest']",
    )
    performance_metrics: list[str] = Field(
        default_factory=list,
        description=(
            "List of performance metrics as 'name: value' strings, e.g. "
            "['accuracy: 91%', 'F1: 0.88', 'sensitivity: 83.94%']. "
            "If a metric appears with multiple values (e.g. for different conditions), "
            "include each as a SEPARATE item with a descriptive name like "
            "'accuracy on day 0: 79%' and 'accuracy 2 days prior: 64%'. "
            "Only include numbers that have a clear metric name in the text. "
            "Skip raw numbers without an associated metric label."
        ),
    )
    dataset_size: str = Field(
        default="",
        description="Number of animals or samples as a string, e.g. '120 cows'",
    )
    key_findings: str = Field(
        default="",
        description="2-3 sentence summary of the main results",
    )

CHUNK_SIZE = 1000       # words per chunk
CHUNK_OVERLAP = 200     # overlapping words between chunks
TOP_K_CHUNKS = 8        # chunks to retrieve per extraction query

EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are extracting structured data from a scientific paper about Precision Livestock Farming.

Paper excerpts:
{context}

Extract the requested fields from the excerpts above. Only include information that is explicitly stated in the text — do NOT guess or invent values. If a field cannot be determined, leave it empty.

{format_instructions}
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
    def __init__(
        self,
        indexer: QdrantIndexer | None = None,
        collection_suffix: str = "",
    ):
        self.indexer = indexer or QdrantIndexer(collection_suffix=collection_suffix)
        self.embedder = self.indexer.embedder
        self.fulltext_collection = settings.fulltext_collection + collection_suffix

        self.ft_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._ensure_fulltext_collection()

        # JSON mode + Pydantic parser. Avoids Groq's flaky tool-calling validation
        # while still getting full schema validation.
        self.llm = get_llm(temperature=0, json_mode=True)
        self.parser = PydanticOutputParser(pydantic_object=_ExtractionSchema)
        prompt = EXTRACTION_PROMPT.partial(
            format_instructions=self.parser.get_format_instructions(),
        )
        self.chain = prompt | self.llm | self.parser

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _invoke_chain(self, context: str) -> "_ExtractionSchema":
        """LLM call wrapped with retry — handles transient network errors during long runs."""
        return self.chain.invoke({"context": context})

    def _ensure_fulltext_collection(self):
        existing = [c.name for c in self.ft_client.get_collections().collections]
        if self.fulltext_collection not in existing:
            self.ft_client.create_collection(
                collection_name=self.fulltext_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Extractor] Created collection: {self.fulltext_collection}")

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
        self.ft_client.upsert(collection_name=self.fulltext_collection, points=points)

    def _retrieve_chunks(self, paper_id: str, query: str) -> list[str]:
        query_vector = self.embedder.encode([query], show_progress_bar=False)[0].tolist()
        results = self.ft_client.search(
            collection_name=self.fulltext_collection,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=TOP_K_CHUNKS,
            with_payload=True,
        )
        return [r.payload["text"] for r in results]

    def _to_result(self, schema: _ExtractionSchema, paper_id: str, source: str) -> ExtractionResult:
        """Convert the LLM-validated schema into our internal ExtractionResult."""
        return ExtractionResult(
            paper_id=paper_id,
            animal_species=schema.animal_species,
            sensor_types=schema.sensor_types,
            ml_methods=schema.ml_methods,
            performance_metrics=schema.performance_metrics,
            dataset_size=schema.dataset_size,
            key_findings=schema.key_findings,
            extraction_source=source,
        )

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

        try:
            schema = self._invoke_chain(context)
        except Exception as e:
            print(f"  [LLM] Full-text extraction failed for {paper.id} after retries: {type(e).__name__}")
            return None
        return self._to_result(schema, paper.id, source="fulltext")

    def _extract_from_abstract(self, paper: Paper) -> ExtractionResult:
        """Fallback extraction using only the title + abstract."""
        context = f"TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}"
        try:
            schema = self._invoke_chain(context)
        except Exception as e:
            print(f"  [LLM] Abstract extraction failed for {paper.id} after retries: {type(e).__name__}")
            return ExtractionResult(paper_id=paper.id, extraction_source="abstract")
        return self._to_result(schema, paper.id, source="abstract")
        return self._parse_extraction(raw, paper.id, source="abstract")

    def extract_paper(self, paper: Paper) -> ExtractionResult:
        """
        Hybrid extraction:
          1. Try to download and extract from full PDF
          2. If PDF unavailable or parsing fails, fall back to abstract-only
        Always returns an ExtractionResult and persists it to Qdrant.
        """
        pdf_path = download_pdf(paper)
        result = None
        if pdf_path:
            result = self._extract_from_fulltext(paper, pdf_path)
            if result is None:
                print(f"  [Fallback] Full-text extraction failed for {paper.id} — using abstract")

        if result is None:
            result = self._extract_from_abstract(paper)

        # Persist to Qdrant so Module 5 can query
        self.indexer.update_extraction(paper.id, result)
        return result

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
