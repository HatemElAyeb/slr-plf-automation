"""
Question-specific field extractor.

Runs a SECOND extraction pass over each included paper, extracting fields
defined in research_questions.py per-question 'custom_fields'. Results are
stored as a `custom_fields` dict in the paper's Qdrant payload (additive —
the standard extraction fields are kept).
"""
import json
import re
import os

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm import get_llm
from src.indexer.indexer import QdrantIndexer
from src.models import Paper, ScreeningStatus
from src.extraction.pdf_downloader import download_pdf
from config.settings import settings

import fitz  # PyMuPDF


PROMPT = ChatPromptTemplate.from_template("""
You are extracting question-specific structured data from a scientific
paper about Precision Livestock Farming.

PAPER TITLE: {title}

PAPER CONTEXT (excerpts):
{context}

Extract the following fields. Each field is a list of short strings.
Only include items explicitly stated in the text — do NOT guess.
If a field cannot be determined, return an empty list [].

Fields:
{fields_spec}

Return ONLY a valid JSON object whose keys are exactly the field names
above and whose values are arrays of strings. No preamble, no explanation,
no code fences.
""")


def _format_fields_spec(custom_fields: dict[str, str]) -> str:
    return "\n".join(f"  - {name}: {desc}" for name, desc in custom_fields.items())


def _build_context(paper: Paper) -> str:
    """Use full PDF text if available, else fall back to title + abstract."""
    pdf_path = os.path.join(settings.pdf_dir, f"{paper.id}.pdf")
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
        try:
            doc = fitz.open(pdf_path)
            pages = "\n".join(doc[i].get_text() for i in range(min(doc.page_count, 8)))
            doc.close()
            return pages[:8000]
        except Exception:
            pass
    return f"TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}"


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
def _invoke(chain, **kwargs) -> dict:
    raw = chain.invoke(kwargs)
    return _parse_json(raw)


def extract_custom_for_question(question: dict) -> int:
    """
    Run custom extraction for all included papers in a question.
    Returns the number of papers successfully processed.
    """
    custom_fields = question.get("custom_fields") or {}
    if not custom_fields:
        print(f"  [{question['id']}] no custom_fields defined, skipping")
        return 0

    qid = question["id"]
    suffix = "_" + qid
    indexer = QdrantIndexer(collection_suffix=suffix)

    points = indexer.client.scroll(
        collection_name=indexer.collection_name,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )[0]

    included = [p for p in points if p.payload.get("screening_status") == ScreeningStatus.INCLUDED.value]
    if not included:
        print(f"  [{qid}] no included papers")
        return 0

    print(f"  [{qid}] {len(included)} included papers, fields: {list(custom_fields.keys())}")

    llm = get_llm(temperature=0, json_mode=True)
    prompt = PROMPT.partial(fields_spec=_format_fields_spec(custom_fields))
    chain = prompt | llm | StrOutputParser()

    processed = 0
    for point in included:
        pl = point.payload
        paper = indexer._point_to_paper(point)
        # Skip if already processed (existing custom_fields covers all expected keys)
        existing = pl.get("custom_fields") or {}
        if all(k in existing for k in custom_fields.keys()):
            continue

        context = _build_context(paper)
        try:
            data = _invoke(chain, title=paper.title, context=context)
        except Exception as e:
            print(f"    ✗ {paper.id}: {type(e).__name__}")
            continue

        # Filter to only the requested keys, ensure list[str]
        clean: dict[str, list[str]] = {}
        for k in custom_fields.keys():
            v = data.get(k, [])
            if isinstance(v, str):
                v = [v] if v.strip() else []
            elif isinstance(v, list):
                v = [str(x).strip() for x in v if str(x).strip()]
            else:
                v = []
            clean[k] = v

        # Persist to Qdrant
        indexer.client.set_payload(
            collection_name=indexer.collection_name,
            payload={"custom_fields": clean},
            points=[point.id],
        )
        processed += 1
        print(f"    ✓ {paper.id}: {sum(len(v) for v in clean.values())} items extracted")

    print(f"  [{qid}] processed {processed}/{len(included)} papers")
    return processed
