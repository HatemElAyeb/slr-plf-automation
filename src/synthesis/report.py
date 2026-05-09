"""
Generate a Markdown SLR report for a research question.
Combines auto-computed statistics with LLM-generated narrative sections.
"""
import os
import json
import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm import get_llm
from src.synthesis.statistics import compute_statistics
from src.synthesis.figures import generate_figures_for_question


def _format_dist(d: dict, total: int | None = None, hide_if_single: bool = False) -> str:
    """Format a distribution dict as a Markdown table.
    hide_if_single: if True, return None when there's 0 or 1 entries (so caller can skip the section)."""
    if hide_if_single and len(d) <= 1:
        return None
    if not d:
        return "_(no data)_\n"
    total = total or sum(d.values())
    if total == 0:
        return "_(no data)_\n"
    lines = ["| Value | Count | % |", "|---|---:|---:|"]
    for k, v in d.items():
        pct = (v / total) * 100
        lines.append(f"| {k} | {v} | {pct:.1f}% |")
    return "\n".join(lines) + "\n"


def _format_prisma(p: dict) -> str:
    """ASCII-art PRISMA flow diagram."""
    return (
        "```\n"
        f"┌─ IDENTIFICATION ─────────────────────────┐\n"
        f"│ Records collected (after dedup): {p['identified']:>5}    │\n"
        f"└──────────────────────────────────────────┘\n"
        f"                    │\n"
        f"                    ▼\n"
        f"┌─ SCREENING ──────────────────────────────┐\n"
        f"│ Records screened:                {p['screened']:>5}    │──► Excluded: {p['excluded_genuine']:>4}\n"
        f"│ Screening failed (LLM error):    {p['screening_failed']:>5}    │\n"
        f"└──────────────────────────────────────────┘\n"
        f"                    │\n"
        f"                    ▼\n"
        f"┌─ INCLUDED ───────────────────────────────┐\n"
        f"│ Studies included:                {p['included']:>5}    │\n"
        f"└──────────────────────────────────────────┘\n"
        f"                    │\n"
        f"                    ▼\n"
        f"┌─ EXTRACTED ──────────────────────────────┐\n"
        f"│ From full PDF:                   {p['extracted_fulltext']:>5}    │\n"
        f"│ From abstract only (no PDF):     {p['extracted_abstract']:>5}    │\n"
        f"│ Total extracted:                 {p['extracted_total']:>5}    │\n"
        f"└──────────────────────────────────────────┘\n"
        "```\n"
    )


SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are writing a section of a Systematic Literature Review (SLR) report on
Precision Livestock Farming (PLF).

RESEARCH QUESTION:
{question}

CORPUS STATISTICS (use these numbers — do NOT invent counts):
{stats_summary}

INCLUDED PAPERS DATA (JSON, with extracted fields):
{papers_data}

TASK: Write the **{section}** section of the SLR report.

STRICT REQUIREMENTS:
- Use the EXACT counts and percentages from CORPUS STATISTICS above. NEVER make up numbers.
- Every quantitative claim must be backed by a number from the statistics.
- Cite papers inline using [FirstAuthorLastName Year] using authors/year from the data.
- Style: academic, neutral. Use Markdown but do NOT add a section heading
  (the heading is added separately).
- Length: {length}.

{section_guidance}

Return only the Markdown content of the section — no preamble, no heading, no code fences.
""")

SECTION_GUIDANCE = {
    "Introduction": (
        "Write a QUESTION-SPECIFIC introduction. Do NOT describe Precision Livestock "
        "Farming in general — that's covered in the master report. Instead, focus "
        "narrowly on:\n"
        "  (a) Why THIS specific question matters (the gap it addresses).\n"
        "  (b) Sub-topics or terms that this question encompasses.\n"
        "  (c) Brief preview of corpus size and what kind of papers were included.\n"
        "Keep it tight: 1-2 paragraphs. Do NOT repeat the methodology pipeline."
    ),
    "Results": (
        "Synthesize findings ACROSS the corpus. Structure:\n"
        "  (a) State the most prevalent sensor types using the EXACT counts from "
        "      'top_sensor_types' — e.g. 'Cameras dominate the corpus (X papers, Y%)'.\n"
        "  (b) State the most prevalent ML methods using 'top_ml_methods' counts.\n"
        "  (c) State the animal species distribution using 'top_animal_species' counts.\n"
        "  (d) Discuss notable performance metrics from the included papers.\n"
        "  (e) Compare approaches between papers when relevant, with citations.\n"
        "4-6 paragraphs."
    ),
    "Discussion": (
        "Directly answer the research question using quantitative evidence:\n"
        "  (a) Quote the dominant sensor+AI combinations with counts/percentages.\n"
        "  (b) Identify CONCRETE research gaps based on what the data shows is "
        "      under-represented (e.g. 'only N papers used species X', "
        "      'only M papers reported metric Y').\n"
        "  (c) Discuss methodological limitations of the included studies "
        "      (e.g. small datasets, single-farm studies).\n"
        "  (d) Note limitations of the review itself "
        "      (e.g. PDFs unavailable for K papers, language restriction, year cutoff).\n"
        "  (e) Briefly mention practical implications for the field.\n"
        "3-4 paragraphs."
    ),
    "Conclusion": (
        "Directly answer the research question in the first sentence, citing top "
        "sensor+AI combinations with counts. Then list 2-3 concrete future research "
        "directions grounded in the gaps you identified. "
        "Use the actual corpus size from CORPUS STATISTICS (not made-up numbers). "
        "1-2 paragraphs."
    ),
}


def _format_stats_for_llm(stats: dict) -> str:
    """Compact text summary of the statistics, injected into the prompt."""
    p = stats["prisma"]
    lines = [
        f"PRISMA counts:",
        f"  - Records identified (after dedup): {p['identified']}",
        f"  - Records screened: {p['screened']}",
        f"  - Records included: {p['included']}",
        f"  - Records excluded: {p['excluded']}",
        f"  - Extraction succeeded: {p['extracted_total']} "
        f"({p['extracted_fulltext']} from full text, {p['extracted_abstract']} from abstract)",
        f"  - Missing PDFs: {p['missing_pdfs']}",
        "",
        f"Source distribution (across {p['included']} included papers):",
    ]
    for k, v in stats["source_distribution"].items():
        pct = (v / p["included"]) * 100 if p["included"] else 0
        lines.append(f"  - {k}: {v} ({pct:.1f}%)")

    lines.append("")
    lines.append(f"Venue quartile (journals among included):")
    for k, v in stats["quartile_distribution"].items():
        pct = (v / p["included"]) * 100 if p["included"] else 0
        lines.append(f"  - {k}: {v} ({pct:.1f}%)")

    lines.append("")
    lines.append(f"Top animal species (across {p['extracted_total']} extracted papers):")
    for k, v in stats["top_animal_species"].items():
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append(f"Top sensor types (across {p['extracted_total']} extracted papers):")
    for k, v in stats["top_sensor_types"].items():
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append(f"Top ML methods (across {p['extracted_total']} extracted papers):")
    for k, v in stats["top_ml_methods"].items():
        lines.append(f"  - {k}: {v}")

    # Question-specific custom fields
    custom = stats.get("custom_fields") or {}
    for fname, dist in custom.items():
        if not dist:
            continue
        lines.append("")
        lines.append(f"Top {fname.replace('_', ' ')} (question-specific):")
        for k, v in dist.items():
            lines.append(f"  - {k}: {v}")

    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
def _generate_section(chain, question: str, section: str, papers_data: list,
                       length: str, stats_summary: str) -> str:
    return chain.invoke({
        "question":     question,
        "section":      section,
        "stats_summary": stats_summary,
        "papers_data":  json.dumps(papers_data, indent=2, default=str)[:15000],
        "length":       length,
        "section_guidance": SECTION_GUIDANCE.get(section, ""),
    })


def _papers_summary_for_llm(papers: list[dict]) -> list[dict]:
    """Strip down full payloads to just what the LLM needs to synthesize."""
    out = []
    for p in papers:
        out.append({
            "authors": p.get("authors", [])[:2] + (["et al."] if len(p.get("authors", [])) > 2 else []),
            "year": p.get("year"),
            "title": p.get("title"),
            "venue": p.get("venue_name"),
            "quartile": p.get("quartile"),
            "conference_rank": p.get("conference_rank"),
            "animal_species": p.get("animal_species") or [],
            "sensor_types": p.get("sensor_types") or [],
            "ml_methods": p.get("ml_methods") or [],
            "performance_metrics": p.get("performance_metrics") or [],
            "key_findings": p.get("key_findings") or "",
        })
    return out


def generate_report(
    question_id: str,
    question_text: str,
    output_path: str | None = None,
    question: dict | None = None,
) -> str:
    """
    Generate the full Markdown SLR report for a question.
    Saves it to output_path (default: data/runs/{qid}/report.md) and returns the text.

    `question` is the full dict from research_questions.py — needed for
    sankey_diagrams configuration. Falls back to looking it up by id.
    """
    if question is None:
        from research_questions import QUESTIONS
        question = next((q for q in QUESTIONS if q["id"] == question_id), None) or {
            "id": question_id, "text": question_text
        }

    print(f"[Report] Computing statistics for {question_id}...")
    stats = compute_statistics(question_id)

    extracted_summary = _papers_summary_for_llm(stats["extracted_papers"])
    stats_summary = _format_stats_for_llm(stats)

    # --- Generate Sankey figures ---
    print(f"[Report] Generating Sankey figures...")
    try:
        figures = generate_figures_for_question(question, stats["included_papers"])
    except Exception as e:
        print(f"  [Figures] failed: {type(e).__name__}: {e}")
        figures = []

    print(f"[Report] Generating LLM narrative sections...")
    llm = get_llm(temperature=0.3, json_mode=False)
    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()

    sections = {}
    for section, length in [
        ("Introduction", "300-500 words"),
        ("Results", "600-1000 words"),
        ("Discussion", "400-700 words"),
        ("Conclusion", "150-300 words"),
    ]:
        print(f"  - {section}...")
        try:
            sections[section] = _generate_section(
                chain, question_text, section, extracted_summary, length, stats_summary
            )
        except Exception as e:
            sections[section] = f"_(Error generating {section}: {type(e).__name__})_"

    # Build final markdown
    p = stats["prisma"]
    md = []
    md.append(f"# Systematic Literature Review — {question_id}\n")
    md.append(f"**Research question:** {question_text}\n")
    md.append(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    md.append(
        "\n> **Note:** This is a per-question report. For the project-wide "
        "introduction, methodology, and cross-question synthesis, see "
        "`master_report.md`.\n"
    )
    md.append("\n---\n\n## 1. Introduction\n")
    md.append(sections["Introduction"] + "\n")

    md.append("\n## 2. PRISMA Flow Diagram\n")
    md.append(_format_prisma(p))

    md.append("\n## 3. Corpus Statistics\n")

    def _maybe_section(title: str, dist: dict, hide_if_single: bool = True):
        """Append section + table only if there's enough data to be informative."""
        formatted = _format_dist(dist, hide_if_single=hide_if_single)
        if formatted is None:
            return  # single-row distribution, skip the whole section
        md.append(f"\n### {title}\n")
        md.append(formatted)

    _maybe_section("3.1 Source distribution (included papers)", stats["source_distribution"])
    _maybe_section("3.2 Venue quartile (journals)",             stats["quartile_distribution"])
    _maybe_section("3.3 Conference vs journal",                 stats["venue_type_split"])
    _maybe_section("3.4 Conference rank (CORE)",                stats["conference_rank_distribution"])
    _maybe_section("3.5 Year distribution",                     stats["year_distribution"], hide_if_single=False)

    md.append("\n## 4. Most-frequent extracted fields\n")
    md.append("\n### 4.1 Animal species\n")
    md.append(_format_dist(stats["top_animal_species"]))
    md.append("\n### 4.2 Sensor types\n")
    md.append(_format_dist(stats["top_sensor_types"]))
    md.append("\n### 4.3 ML methods\n")
    md.append(_format_dist(stats["top_ml_methods"]))

    # Question-specific custom field distributions
    custom = stats.get("custom_fields") or {}
    if custom:
        md.append("\n## 4.bis Question-specific extracted fields\n")
        for i, (fname, dist) in enumerate(custom.items(), start=1):
            pretty = fname.replace("_", " ").title()
            md.append(f"\n### 4.bis.{i} {pretty}\n")
            md.append(_format_dist(dist) or "_(no data)_\n")

    # Sankey / flow diagrams
    if figures:
        md.append("\n## 4.ter Flow diagrams\n")
        for f in figures:
            md.append(f"\n### {f['title']}\n")
            stages_str = " → ".join(s.replace("_", " ").title() for s in f["stages"])
            md.append(f"_Stages:_ {stages_str}\n")
            if f.get("rel_path"):
                md.append(f"\n![{f['title']}]({f['rel_path']})\n")
            else:
                md.append("_(figure could not be rendered)_\n")

    md.append("\n## 5. Results\n")
    md.append(sections["Results"] + "\n")

    md.append("\n## 6. Discussion\n")
    md.append(sections["Discussion"] + "\n")

    md.append("\n## 7. Conclusion\n")
    md.append(sections["Conclusion"] + "\n")

    md.append("\n## 8. Included Papers\n")
    for paper in stats["included_papers"]:
        authors = "; ".join((paper.get("authors") or [])[:3])
        if len(paper.get("authors") or []) > 3:
            authors += " et al."
        venue = paper.get("venue_name") or "—"
        q = paper.get("quartile") or paper.get("conference_rank") or "—"
        doi = paper.get("doi") or "—"
        md.append(
            f"- **{paper.get('title', 'Untitled')}** ({paper.get('year', '?')})  \n"
            f"  {authors}. *{venue}* [{q}]. DOI: {doi}\n"
        )

    text = "\n".join(md)

    # Save to disk
    if output_path is None:
        output_path = os.path.join("data", "runs", question_id, "report.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Report] Saved: {output_path}")
    return text
