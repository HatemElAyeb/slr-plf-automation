from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    abstracts_collection: str = "plf_abstracts"
    fulltext_collection: str = "plf_full_text"

    # LLM provider: "ollama" or "groq"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    screening_model: str = os.getenv("SCREENING_MODEL", "llama3.1:8b")
    synthesis_model: str = os.getenv("SYNTHESIS_MODEL", "llama3.1:8b")

    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_screening_model: str = os.getenv("GROQ_SCREENING_MODEL", "llama-3.1-8b-instant")
    groq_synthesis_model: str = os.getenv("GROQ_SYNTHESIS_MODEL", "llama-3.1-8b-instant")

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # Literature APIs
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "user@example.com")
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "user@example.com")
    springer_meta_api_key: str = os.getenv("SPRINGER_META_API_KEY", "")
    springer_oa_api_key: str = os.getenv("SPRINGER_OA_API_KEY", "")

    # Storage
    pdf_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
    raw_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

    # Screening
    screening_batch_size: int = 20
    screening_confidence_threshold: float = 0.5


settings = Settings()
