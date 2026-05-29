# slr-plf-automation

An automated **Systematic Literature Review (SLR)** pipeline for
**Precision Livestock Farming (PLF)**, powered by LLMs and
Retrieval-Augmented Generation (RAG). End-to-end: from a
natural-language research question to a structured per-question report
and a Streamlit exploration interface.

---

## What is this project?

A Systematic Literature Review is a rigorous method for synthesising
all published research relevant to a specific question. Manually
performed it takes a team of two or more reviewers between 6 and 18
months and consumes upwards of a thousand person-hours. This project
automates the full PRISMA pipeline using LLMs for query generation,
abstract screening, and structured extraction, and RAG for full-text
mining.

The pipeline was developed and applied to **six research questions on
Precision Livestock Farming** covering technical methods, outcomes,
gaps, and future directions.

---

## Pipeline

```
Research Question
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 0 — LLM Query & Criteria Builder                      │
│   Generates boolean queries + inclusion/exclusion criteria   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 1 — Multi-Source Literature Collection                │
│   PubMed · OpenAlex · ArXiv · MDPI · Springer Nature         │
│   Deduplicate by DOI / normalised title                      │
│ Module 1.5 — Venue ranking (Scimago Q1-Q4 / CORE A*-C)       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 2 — Qdrant Indexing                                   │
│   Embed title+abstract with BAAI/bge-large-en-v1.5           │
│   One Qdrant collection per research question                │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 3 — Abstract Screening                                │
│   LLM applies the question-specific criteria                 │
│   Persists decision + confidence + reason                    │
└──────────────────────────────┬───────────────────────────────┘
                               │ (included papers only)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 4 — Full-Text RAG Extraction                          │
│   PDF acquisition: Unpaywall → publisher URL → local cache   │
│   Chunk + embed; LLM extracts a structured record            │
│   Falls back to abstract-only when no PDF is obtainable      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 5 — Synthesis & Reporting                             │
│   Normalised statistics + Sankey diagrams                    │
│   Master report + 6 per-question Markdown reports            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Module 6 — Streamlit Interface (app.py)                      │
│   Browse papers · run pipeline · view reports & diagrams     │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (default) | OpenAI `gpt-4o-mini` — also pluggable to Ollama or Groq via `LLM_PROVIDER` |
| Orchestration | LangChain |
| Vector DB | Qdrant (Docker) |
| Embeddings | BAAI/bge-large-en-v1.5 (local, free) |
| PDF parsing | PyMuPDF |
| OA fallback | Unpaywall API |
| Literature APIs | PubMed (E-utilities), OpenAlex, ArXiv, MDPI (via CrossRef), Springer Meta |
| Visualisation | Plotly Sankey + kaleido (PNG export) |
| Frontend | Streamlit |

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/HatemElAyeb/slr-plf-automation.git
cd slr-plf-automation
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file at the repo root:
```env
# LLM provider — one of: openai | groq | ollama
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Polite-pool contact emails (required by PubMed and OpenAlex)
PUBMED_EMAIL=you@example.com
OPENALEX_EMAIL=you@example.com

# Springer Nature Meta API key (free: dev.springernature.com)
SPRINGER_META_API_KEY=

# Qdrant (defaults work if you run the container as below)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant
```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 4. (Optional) Local LLM via Ollama
If you set `LLM_PROVIDER=ollama` instead of `openai`:
```bash
# install Ollama from https://ollama.com
ollama pull llama3.1:8b
```

---

## Running the pipeline

### Full end-to-end run on all six research questions
```bash
python run_pipeline.py
```
The six questions are declared in [`research_questions.py`](research_questions.py).
Each one produces:
- `data/runs/{qid}/config.json` — its generated queries and criteria
- `data/runs/{qid}/report.md` — its synthesis report
- `data/runs/{qid}/figures/` — its Sankey diagrams
- a per-question Qdrant collection (`plf_abstracts_{qid}`)

Plus a project-wide `data/runs/master_report.md` aggregating everything.

### Streamlit UI
```bash
streamlit run app.py
```
Pages: Home · Run pipeline · Browse papers · Statistics · Flow diagrams ·
Report · Missing PDFs.

### One-shot helpers (already-run, kept for reproducibility)
| Script | Purpose |
|---|---|
| [`build_normalization_map.py`](build_normalization_map.py) | LLM proposes canonical groupings for sensor/method/species variants; output edited by hand into `data/normalization_map.json`. |
| [`extract_custom_fields.py`](extract_custom_fields.py) | Second-pass extractor that fills the per-question `custom_fields` declared in `research_questions.py`. |
| [`generate_report.py`](generate_report.py) | Re-renders the master + per-question Markdown reports and Sankey figures. |
| [`match_supervisor_pdfs.py`](match_supervisor_pdfs.py) | Matches manually-collected PDFs to existing records by DOI. |
| [`re_extract_with_pdfs.py`](re_extract_with_pdfs.py) | Re-runs Module 4 on abstract-only papers once new PDFs become available. |
| [`list_missing_pdfs.py`](list_missing_pdfs.py) | Lists included papers whose PDF could not be obtained. |

---

## Project Structure

```
slr-plf-automation/
├── config/
│   └── settings.py             # All configuration (Qdrant, LLM provider, APIs)
├── src/
│   ├── models.py               # Pydantic models (Paper, ScreeningResult, ...)
│   ├── llm.py                  # LLM provider abstraction (OpenAI/Ollama/Groq)
│   ├── query_builder/          # Module 0 — LLM query & criteria builder
│   ├── collectors/             # Module 1 — five literature collectors
│   │   ├── pubmed.py
│   │   ├── openalex.py
│   │   ├── arxiv.py
│   │   ├── mdpi.py
│   │   └── springer.py
│   ├── rankings.py             # Module 1.5 — Scimago + CORE lookups
│   ├── indexer/                # Module 2 — Qdrant embedding & storage
│   ├── screening/              # Module 3 — abstract screening
│   ├── extraction/             # Module 4 — RAG full-text extraction
│   │   ├── extractor.py
│   │   ├── pdf_downloader.py   # Unpaywall + publisher + local-cache fallback
│   │   └── custom_extractor.py # per-question custom fields
│   └── synthesis/              # Module 5 — statistics, figures, reports
│       ├── statistics.py
│       ├── figures.py
│       ├── report.py
│       └── master_report.py
├── data/
│   ├── pdfs/                   # Downloaded / locally cached PDFs (gitignored)
│   ├── runs/{qid}/             # Per-question outputs (gitignored)
│   │   ├── config.json
│   │   ├── report.md
│   │   └── figures/*.png
│   ├── normalization_map.json  # Curated synonym → canonical mappings
│   ├── scimagojr.csv           # Scimago Journal Rank dump
│   └── core_conferences.csv    # CORE conference ranking dump
├── tests/                      # Sanity-check scripts (one per module)
├── research_questions.py       # The six research questions + custom fields
├── run_pipeline.py             # End-to-end orchestrator
├── app.py                      # Streamlit interface
├── requirements.txt
└── README.md
```

