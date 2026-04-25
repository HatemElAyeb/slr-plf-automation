from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ScreeningStatus(str, Enum):
    PENDING = "pending"
    INCLUDED = "included"
    EXCLUDED = "excluded"


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    source: str  # pubmed | openalex | arxiv | crossref
    pdf_url: Optional[str] = None
    screening_status: ScreeningStatus = ScreeningStatus.PENDING
    screening_confidence: Optional[float] = None
    screening_reason: Optional[str] = None


class ScreeningResult(BaseModel):
    decision: ScreeningStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ExtractionResult(BaseModel):
    paper_id: str
    animal_species: list[str] = Field(default_factory=list)
    sensor_types: list[str] = Field(default_factory=list)
    ml_methods: list[str] = Field(default_factory=list)
    performance_metrics: dict[str, str] = Field(default_factory=dict)
    dataset_size: Optional[str] = None
    key_findings: str = ""
    extraction_source: str = "fulltext"  # "fulltext" | "abstract"


class SLRState(BaseModel):
    research_question: str
    search_queries: dict = Field(default_factory=dict)
    collected_papers: list[Paper] = Field(default_factory=list)
    screened_papers: list[Paper] = Field(default_factory=list)
    included_papers: list[Paper] = Field(default_factory=list)
    extracted_data: list[ExtractionResult] = Field(default_factory=list)
    final_report: str = ""
