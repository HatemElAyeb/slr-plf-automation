# slr-plf-automation
Automated Systematic Literature Review (SLR) pipeline for Precision Livestock Farming (PLF) using LLMs, LangChain, LangGraph, and Qdrant.

---

## What is this project?

A Systematic Literature Review is a rigorous method for synthesizing all published research on a specific question. Traditionally it takes 6–18 months for a human team. This project automates the entire pipeline using LLMs, applied to **Precision Livestock Farming** — a domain that uses sensors, IoT, and AI to monitor animal health and behavior in real time.

---

## Pipeline

```
Research Question
       │
       ▼
┌─────────────────┐
│   Module 1      │  Collect papers from PubMed, OpenAlex, ArXiv
│   Collection    │  Deduplicate across sources
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Module 2      │  Embed title + abstract with BAAI/bge-large-en-v1.5
│   Indexing      │  Store vectors + metadata in Qdrant
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Module 3      │  LLM reads each abstract
│   Screening     │  Decides: include / exclude + confidence + reason
└────────┬────────┘
         │ (included papers only)
         ▼
┌─────────────────┐
│   Module 4      │  Download PDFs of included papers
│   Extraction    │  Chunk + embed full text (RAG)
│                 │  LLM extracts: species, sensors, methods, metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Module 5      │  LLM synthesizes all extracted data
│   Synthesis     │  Generates structured academic report
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Module 6      │  Streamlit web interface
│   UI            │  Run pipeline, review results, export report
└─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + LangChain |
| LLM | Ollama (Llama 3.1:8b) — runs locally, free |
| Vector DB | Qdrant (Docker) |
| Embeddings | BAAI/bge-large-en-v1.5 (local, free) |
| PDF Parsing | PyMuPDF / PyPDFLoader |
| Literature APIs | PubMed, OpenAlex, ArXiv — all free |
| Frontend | Streamlit |

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <repo-url>
cd slr-plf-automation
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your email for PubMed and OpenAlex
```

### 3. Start Qdrant
```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 4. Install Ollama and pull model
```bash
# Download Ollama from https://ollama.com
ollama pull llama3.1:8b
```

---

## Project Structure

```
slr-plf-automation/
├── config/
│   └── settings.py          # All configuration (Qdrant, Ollama, models)
├── src/
│   ├── models.py             # Pydantic data models (Paper, SLRState, ...)
│   ├── collectors/           # Module 1 — literature collection
│   │   ├── pubmed.py
│   │   ├── openalex.py
│   │   ├── arxiv.py
│   │   └── collector.py      # LiteratureCollector (main entry point)
│   ├── indexer/              # Module 2 — Qdrant embedding + storage
│   │   └── indexer.py
│   ├── screening/            # Module 3 — abstract screening (LLM)
│   ├── extraction/           # Module 4 — full-text RAG extraction
│   └── synthesis/            # Module 5 — report generation
├── data/
│   ├── pdfs/                 # Downloaded PDFs
│   └── raw/                  # Raw API responses
├── test_collectors.py
├── test_indexer.py
├── requirements.txt
└── .env
```
