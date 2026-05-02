"""
Module 0 — LLM-based query builder.

Takes a natural-language research question and uses the LLM to generate
optimized boolean queries for each literature API (PubMed, ArXiv, OpenAlex,
MDPI, Springer) plus suggested ArXiv categories.
"""
import json
import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_llm


QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research librarian helping construct boolean search queries
for a Systematic Literature Review (SLR).

Given a natural-language research question, identify the key concepts and
expand each one with synonyms. Combine them with boolean operators to produce
optimized queries for several academic APIs.

RESEARCH QUESTION:
{question}

GUIDANCE:
- Identify 2-4 main concepts (e.g. animal type, technology, method, outcome)
- For each concept, list 2-5 synonyms or related terms
- Combine concepts with AND, synonyms with OR
- Use parentheses to group properly
- For PubMed and ArXiv: full boolean syntax (AND, OR uppercase)
- For OpenAlex and MDPI: simpler search — use just the most important keywords as a phrase, no boolean operators
- For Springer: boolean syntax supported
- For ArXiv categories: pick relevant subject codes from {{cs.AI, cs.CV, cs.LG, eess.SP, eess.IV, cs.RO}}

EXAMPLE INPUT:
"What deep learning methods are used for dairy cattle lameness detection?"

EXAMPLE OUTPUT:
{{
  "pubmed_query": "(dairy cattle OR dairy cow* OR bovine) AND (lameness OR gait) AND (\\"deep learning\\" OR CNN OR \\"neural network*\\")",
  "arxiv_query": "(cattle OR cow OR bovine) AND (lameness OR gait) AND (\\"deep learning\\" OR CNN OR LSTM)",
  "openalex_query": "dairy cattle lameness detection deep learning",
  "mdpi_query": "cattle lameness deep learning",
  "springer_query": "(cattle OR cow) AND lameness AND \\"deep learning\\"",
  "arxiv_categories": ["cs.AI", "cs.CV", "eess.SP"]
}}

Now generate optimized queries for the research question above. Return ONLY a
valid JSON object with the keys: pubmed_query, arxiv_query, openalex_query,
mdpi_query, springer_query, arxiv_categories.
""")


def _parse_response(raw: str) -> dict:
    """Extract the JSON object from the LLM response."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group() if match else raw)
    except Exception as e:
        raise ValueError(f"Failed to parse query builder response: {e}\nRaw: {raw[:500]}")


def build_queries(question: str) -> dict:
    """
    Build optimized API queries from a natural-language research question.

    Returns a dict with:
      - pubmed_query, arxiv_query, openalex_query, mdpi_query, springer_query
      - arxiv_categories (list of category codes)
    """
    llm = get_llm(temperature=0, json_mode=True)
    chain = QUERY_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question})
    queries = _parse_response(raw)

    # Validate expected keys
    expected = {"pubmed_query", "arxiv_query", "openalex_query",
                "mdpi_query", "springer_query", "arxiv_categories"}
    missing = expected - set(queries.keys())
    if missing:
        raise ValueError(f"Query builder response missing keys: {missing}")

    return queries


CRITERIA_PROMPT = ChatPromptTemplate.from_template("""
You are an expert systematic literature review methodologist working on a
review about Precision Livestock Farming (PLF).

Your task is to write inclusion/exclusion criteria for a SPECIFIC research
question. The criteria must be tightly aligned with what the question asks.

RESEARCH QUESTION:
{question}

GUIDELINES:
- The criteria must explicitly link to the question. If the question is about
  "Explainable AI", criteria must require papers that evaluate interpretability.
  If the question is about "barriers to adoption", criteria must allow surveys,
  qualitative studies, and economic analyses (NOT only sensor experiments).
- For technical questions about sensors/AI/methods: focus on primary research,
  exclude reviews/surveys.
- For questions about gaps, barriers, adoption, or future directions: ALLOW
  reviews, surveys, position papers, and qualitative studies.
- For questions about outcomes/welfare/sustainability: require quantitative
  metrics or measurable outcomes.
- Always require the paper to be about FARM LIVESTOCK (cattle, swine, poultry,
  sheep, goats) — NOT pets, wildlife, insects, or aquatic animals.
- Always require publication year >= 2015.
- Animal subject scope can be narrowed if the question targets a specific
  species (e.g. "dairy cattle" → only cattle).

Output a JSON object with TWO string fields:
  - "include": bullet-pointed list of inclusion criteria
  - "exclude": bullet-pointed list of exclusion criteria

EXAMPLE for "What sensor and AI combinations are most prevalent in cattle livestock farming?":
{{
  "include": "- Animals: cattle, dairy cows, beef cattle (NOT other livestock species)\\n- Technology: sensors AND AI/ML methods applied together\\n- Study type: original primary research with experiments or implementations\\n- Goal: monitoring health, behavior, productivity, or welfare of cattle\\n- Year: 2015 or later",
  "exclude": "- Literature reviews, surveys, meta-analyses\\n- Studies on non-cattle livestock or wildlife\\n- Papers without both a sensor AND an AI/ML component\\n- Theoretical-only papers without applied results"
}}

Now generate criteria for the research question above. Return ONLY a valid JSON object.
""")


def build_criteria(question: str) -> dict:
    """
    Build question-specific inclusion/exclusion criteria via LLM.

    Returns a dict with "include" and "exclude" string fields.
    """
    llm = get_llm(temperature=0, json_mode=True)
    chain = CRITERIA_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question})
    data = _parse_response(raw)

    if "include" not in data or "exclude" not in data:
        raise ValueError(f"Criteria builder missing keys. Got: {list(data.keys())}")

    return data


def format_criteria(criteria: dict) -> str:
    """Format criteria dict into the string format expected by AbstractScreener."""
    return (
        "INCLUDE the paper ONLY if it meets ALL of the following:\n"
        f"{criteria['include']}\n\n"
        "EXCLUDE the paper if ANY of the following apply:\n"
        f"{criteria['exclude']}"
    )
