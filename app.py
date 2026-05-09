"""
PFA — Automated SLR for Precision Livestock Farming.
Streamlit dashboard wrapping the entire pipeline.

Run: streamlit run app.py
"""
import os
import sys
import json
import subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from research_questions import QUESTIONS
from src.indexer.indexer import QdrantIndexer
from src.synthesis.statistics import compute_statistics
from config.settings import settings


st.set_page_config(
    page_title="PFA — Automated SLR",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar navigation ----------
st.sidebar.title("🐄 PFA — Automated SLR")
st.sidebar.markdown("**Precision Livestock Farming**")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "▶️ Run pipeline",
        "📚 Browse papers",
        "📊 Statistics",
        "🌊 Flow diagrams",
        "📄 Report",
        "📤 Missing PDFs",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: `{settings.llm_provider}`")


# ---------- Helpers ----------
@st.cache_data(ttl=30)
def list_completed_runs() -> list[str]:
    runs_dir = os.path.join("data", "runs")
    if not os.path.isdir(runs_dir):
        return []
    return sorted(d for d in os.listdir(runs_dir)
                  if os.path.isfile(os.path.join(runs_dir, d, "config.json")))


def load_run_config(qid: str) -> dict | None:
    path = os.path.join("data", "runs", qid, "config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=30)
def get_stats(qid: str) -> dict:
    return compute_statistics(qid)


@st.cache_data(ttl=30)
def get_papers_df(qid: str) -> pd.DataFrame:
    suffix = "_" + qid
    ix = QdrantIndexer(collection_suffix=suffix)
    points, _ = ix.client.scroll(
        collection_name=ix.collection_name,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )
    rows = []
    for p in points:
        pl = p.payload
        rows.append({
            "title":           pl.get("title", ""),
            "authors":         "; ".join((pl.get("authors") or [])[:3]),
            "year":            pl.get("year"),
            "source":          pl.get("source", ""),
            "venue":           pl.get("venue_name", ""),
            "quartile":        pl.get("quartile") or pl.get("conference_rank") or "",
            "is_conference":   pl.get("is_conference", False),
            "doi":             pl.get("doi", ""),
            "screening_status": pl.get("screening_status", ""),
            "extraction_source": pl.get("extraction_source", ""),
            "animal_species":  pl.get("animal_species") or [],
            "sensor_types":    pl.get("sensor_types") or [],
            "ml_methods":      pl.get("ml_methods") or [],
            "performance_metrics": pl.get("performance_metrics") or [],
            "key_findings":    pl.get("key_findings", ""),
        })
    return pd.DataFrame(rows)


# =============================================================
# PAGE: OVERVIEW
# =============================================================
if page == "🏠 Overview":
    st.title("PFA — Automated Systematic Literature Review")
    st.markdown(
        "An end-to-end pipeline that automates SLRs for "
        "**Precision Livestock Farming** using LLMs, vector search, and RAG."
    )

    st.markdown("### Pipeline modules")
    cols = st.columns(3)
    cols[0].success("**Module 0** — LLM query + criteria builder ✓")
    cols[1].success("**Module 1** — Literature collection (5 sources) ✓")
    cols[2].success("**Module 1.5** — Scimago/CORE rankings ✓")
    cols = st.columns(3)
    cols[0].success("**Module 2** — Qdrant indexing ✓")
    cols[1].success("**Module 3** — Abstract screening ✓")
    cols[2].success("**Module 4** — RAG extraction ✓")
    cols = st.columns(3)
    cols[0].success("**Module 5** — Synthesis + report ✓")
    cols[1].success("**Module 6** — Streamlit UI (this) ✓")
    cols[2].info("**Module 1.6** — IEEE Xplore (pending)")

    st.markdown("### Research questions")
    for q in QUESTIONS:
        # Try to compute live stats from Qdrant (always current).
        # Fall back to config.json only if the collection doesn't exist yet.
        live_stats = None
        try:
            live_stats = get_stats(q["id"])
        except Exception:
            pass
        config = load_run_config(q["id"])
        status = "✅ run completed" if (live_stats or config) else "⏳ not yet run"
        with st.expander(f"**{q['id']}** ({q['category']}) — {status}"):
            st.write(q["text"])
            c = st.columns(4)
            if live_stats:
                p = live_stats["prisma"]
                c[0].metric("Collected", p["identified"])
                c[1].metric("Included", p["included"])
                c[2].metric("Extracted (FT)", p["extracted_fulltext"])
                c[3].metric("Extracted (abs)", p["extracted_abstract"])
            elif config:
                r = config.get("results", {})
                c[0].metric("Collected", r.get("collected_after_dedup", "—"))
                c[1].metric("Included", r.get("included", "—"))
                c[2].metric("Extracted (FT)", r.get("extracted_fulltext", "—"))
                c[3].metric("Extracted (abs)", r.get("extracted_abstract", "—"))


# =============================================================
# PAGE: RUN PIPELINE
# =============================================================
elif page == "▶️ Run pipeline":
    st.title("Run pipeline")
    st.markdown("Trigger a fresh pipeline run for a research question. "
                "This calls `run_pipeline.py` as a subprocess.")

    qids = [q["id"] for q in QUESTIONS]
    selected_id = st.selectbox(
        "Research question",
        qids,
        format_func=lambda i: f"{i} — {next(q['text'][:70] for q in QUESTIONS if q['id'] == i)}",
    )

    custom = st.text_area(
        "Or enter a custom research question (overrides the dropdown)",
        placeholder="e.g. What are the main challenges of adopting computer vision in pig farming?",
    )

    max_per_source = st.slider("Max papers per source", 10, 300, 50, step=10)

    st.warning(
        "⚠️ Running the pipeline can take from a few minutes (small max) to "
        "several hours (large max). Output streams below in real time."
    )

    if st.button("▶ Run pipeline", type="primary"):
        # Update MAX_PER_SOURCE temporarily
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if custom.strip():
            st.info("Custom question mode is currently not supported via UI — "
                    "edit `research_questions.py` to add it permanently.")
        else:
            cmd = [sys.executable, "run_pipeline.py", selected_id]
            st.code(" ".join(cmd), language="bash")
            log_box = st.empty()
            log_lines: list[str] = []
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, bufsize=1, universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ) as proc:
                for line in proc.stdout or []:
                    log_lines.append(line.rstrip())
                    log_box.code("\n".join(log_lines[-200:]), language="text")
            if proc.returncode == 0:
                st.success("Pipeline completed.")
            else:
                st.error(f"Pipeline failed (exit code {proc.returncode}).")


# =============================================================
# PAGE: BROWSE PAPERS
# =============================================================
elif page == "📚 Browse papers":
    st.title("Browse papers")
    runs = list_completed_runs()
    if not runs:
        st.info("No runs completed yet. Run a question first.")
        st.stop()

    qid = st.selectbox("Question", runs)
    df = get_papers_df(qid)

    if df.empty:
        st.warning("No papers in this collection.")
        st.stop()

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    status_filter = c1.multiselect("Screening status", df["screening_status"].dropna().unique().tolist(),
                                    default=["included"] if "included" in df["screening_status"].values else [])
    source_filter = c2.multiselect("Source", df["source"].unique().tolist())
    quartile_filter = c3.multiselect("Quartile / Rank", sorted(df["quartile"].dropna().unique().tolist()))
    text_filter = c4.text_input("Search title/authors")

    filt = df.copy()
    if status_filter:
        filt = filt[filt["screening_status"].isin(status_filter)]
    if source_filter:
        filt = filt[filt["source"].isin(source_filter)]
    if quartile_filter:
        filt = filt[filt["quartile"].isin(quartile_filter)]
    if text_filter:
        s = text_filter.lower()
        filt = filt[filt["title"].str.lower().str.contains(s, na=False) |
                    filt["authors"].str.lower().str.contains(s, na=False)]

    st.caption(f"Showing {len(filt)} of {len(df)} papers")
    # Reset filtered index so the visible row numbers match what you type below
    filt = filt.reset_index(drop=True)
    filt.insert(0, "row", filt.index)
    display_cols = ["row", "year", "title", "authors", "source", "venue", "quartile",
                    "screening_status", "extraction_source"]
    st.dataframe(filt[display_cols], use_container_width=True, height=500, hide_index=True)

    # Drill-down
    st.markdown("### Paper detail")
    if not filt.empty:
        idx = st.number_input(
            "Row to inspect (the 'row' column in the table above)",
            min_value=0,
            max_value=len(filt) - 1,
            value=0,
        )
        row = filt.iloc[int(idx)]
        st.markdown(f"#### {row['title']}")
        st.write(f"**Authors:** {row['authors']}  \n"
                 f"**Year:** {row['year']}  \n"
                 f"**Venue:** {row['venue']} ({row['quartile']})  \n"
                 f"**DOI:** {row['doi']}  \n"
                 f"**Source:** {row['source']}  \n"
                 f"**Status:** {row['screening_status']} / extracted from: {row['extraction_source'] or '—'}")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Animal species:**")
        c1.write(row["animal_species"] or "—")
        c2.markdown("**Sensor types:**")
        c2.write(row["sensor_types"] or "—")
        c3.markdown("**ML methods:**")
        c3.write(row["ml_methods"] or "—")
        st.markdown("**Performance metrics:**")
        st.write(row["performance_metrics"] or "—")
        st.markdown("**Key findings:**")
        st.write(row["key_findings"] or "—")


# =============================================================
# PAGE: STATISTICS
# =============================================================
elif page == "📊 Statistics":
    st.title("Statistics")
    runs = list_completed_runs()
    if not runs:
        st.info("No runs completed yet.")
        st.stop()

    qid = st.selectbox("Question", runs)
    stats = get_stats(qid)
    p = stats["prisma"]

    st.markdown("### PRISMA flow")
    cols = st.columns(5)
    cols[0].metric("Identified",   p["identified"])
    cols[1].metric("Screened",     p["screened"])
    cols[2].metric("Included",     p["included"])
    cols[3].metric("Excluded",     p["excluded"])
    cols[4].metric("Extracted",    p["extracted_total"])

    cols = st.columns(2)
    if stats["source_distribution"]:
        cols[0].plotly_chart(
            px.pie(values=list(stats["source_distribution"].values()),
                   names=list(stats["source_distribution"].keys()),
                   title="Source distribution"),
            use_container_width=True,
        )
    if stats["quartile_distribution"]:
        cols[1].plotly_chart(
            px.bar(x=list(stats["quartile_distribution"].keys()),
                   y=list(stats["quartile_distribution"].values()),
                   title="Quartile distribution",
                   labels={"x": "Quartile", "y": "Count"}),
            use_container_width=True,
        )

    cols = st.columns(2)
    if stats["top_sensor_types"]:
        cols[0].plotly_chart(
            px.bar(x=list(stats["top_sensor_types"].values()),
                   y=list(stats["top_sensor_types"].keys()),
                   orientation="h",
                   title="Top sensor types",
                   labels={"x": "Count", "y": "Sensor"}),
            use_container_width=True,
        )
    if stats["top_ml_methods"]:
        cols[1].plotly_chart(
            px.bar(x=list(stats["top_ml_methods"].values()),
                   y=list(stats["top_ml_methods"].keys()),
                   orientation="h",
                   title="Top ML methods",
                   labels={"x": "Count", "y": "Method"}),
            use_container_width=True,
        )

    if stats["top_animal_species"]:
        st.plotly_chart(
            px.bar(x=list(stats["top_animal_species"].values()),
                   y=list(stats["top_animal_species"].keys()),
                   orientation="h",
                   title="Top animal species",
                   labels={"x": "Count", "y": "Species"}),
            use_container_width=True,
        )

    if stats["year_distribution"]:
        st.plotly_chart(
            px.bar(x=list(stats["year_distribution"].keys()),
                   y=list(stats["year_distribution"].values()),
                   title="Publications by year",
                   labels={"x": "Year", "y": "Count"}),
            use_container_width=True,
        )

    # Question-specific custom field distributions
    custom = stats.get("custom_fields") or {}
    if custom:
        st.markdown("---")
        st.subheader("Question-specific extracted fields")
        # Render two charts per row when there are several custom fields
        items = [(name, dist) for name, dist in custom.items() if dist]
        for i in range(0, len(items), 2):
            row = items[i : i + 2]
            row_cols = st.columns(len(row))
            for j, (fname, dist) in enumerate(row):
                pretty = fname.replace("_", " ").title()
                row_cols[j].plotly_chart(
                    px.bar(x=list(dist.values()),
                           y=list(dist.keys()),
                           orientation="h",
                           title=f"Top {pretty}",
                           labels={"x": "Count", "y": pretty}),
                    use_container_width=True,
                )


# =============================================================
# PAGE: FLOW DIAGRAMS
# =============================================================
elif page == "🌊 Flow diagrams":
    st.title("Flow diagrams")
    st.markdown("Sankey diagrams showing how papers flow between extracted "
                "dimensions (animal × sensor × ML method, etc.). Configured "
                "per question in `research_questions.py`.")

    runs = list_completed_runs()
    if not runs:
        st.info("No runs completed yet.")
        st.stop()

    # Filter to only questions that have sankey_diagrams declared
    diag_questions = [q for q in QUESTIONS
                      if q["id"] in runs and (q.get("sankey_diagrams") or [])]
    if not diag_questions:
        st.info("No questions have Sankey diagrams configured yet.")
        st.stop()

    qid = st.selectbox(
        "Question",
        [q["id"] for q in diag_questions],
        format_func=lambda i: f"{i} — {next(q['text'][:70] for q in diag_questions if q['id'] == i)}",
    )
    question = next(q for q in diag_questions if q["id"] == qid)

    from src.synthesis.figures import compute_sankey_data, make_sankey_figure, MIN_FLOW_FOR_REPORT
    stats = get_stats(qid)
    papers = stats["included_papers"]

    if not papers:
        st.warning("No included papers for this question.")
        st.stop()

    show_sparse = st.checkbox(
        f"Show sparse diagrams (total flow < {MIN_FLOW_FOR_REPORT})",
        value=False,
        help="Sparse diagrams aren't included in the markdown report. "
             "Toggle on to inspect them here anyway.",
    )

    rendered_any = False
    for spec in question["sankey_diagrams"]:
        try:
            data = compute_sankey_data(
                papers,
                spec["stages"],
                max_per_stage=spec.get("max_per_stage", 8),
            )
        except Exception as e:
            st.error(f"{spec['title']} — could not compute: {type(e).__name__}: {e}")
            continue

        total_flow = sum(data["value"]) if data["value"] else 0
        if total_flow < MIN_FLOW_FOR_REPORT and not show_sparse:
            continue

        rendered_any = True
        st.markdown(f"### {spec['title']}")
        stages_str = " → ".join(s.replace("_", " ").title() for s in spec["stages"])
        st.caption(f"Stages: {stages_str} · total flow: {total_flow}")
        if total_flow < MIN_FLOW_FOR_REPORT:
            st.warning("This diagram is sparse — not included in the markdown report.")
        try:
            fig = make_sankey_figure(data, spec["title"], spec["stages"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render: {type(e).__name__}: {e}")

    if not rendered_any:
        st.info(f"All diagrams for this question are below the flow threshold "
                f"({MIN_FLOW_FOR_REPORT}). Toggle the checkbox above to inspect them.")


# =============================================================
# PAGE: REPORT
# =============================================================
elif page == "📄 Report":
    st.title("Report viewer")
    runs = list_completed_runs()
    master_exists = os.path.exists(os.path.join("data", "runs", "master_report.md"))

    if not runs and not master_exists:
        st.info("No runs completed yet.")
        st.stop()

    # Build the dropdown: master report first (if available), then per-question
    options = []
    if master_exists:
        options.append("master_report")
    options += runs

    selection = st.selectbox(
        "Report",
        options,
        format_func=lambda o: "🌐 Master report (project-wide)" if o == "master_report"
                              else f"📄 {o}",
    )

    if selection == "master_report":
        report_path = os.path.join("data", "runs", "master_report.md")
        download_name = "master_report.md"
    else:
        report_path = os.path.join("data", "runs", selection, "report.md")
        download_name = f"{selection}_report.md"

    if not os.path.exists(report_path):
        st.warning("Report not generated yet.")
        if selection != "master_report" and st.button("Generate report now"):
            with st.spinner("Generating report..."):
                from src.synthesis import generate_report
                question_text = next((q["text"] for q in QUESTIONS if q["id"] == selection), selection)
                generate_report(selection, question_text)
            st.rerun()
    else:
        with open(report_path, encoding="utf-8") as f:
            text = f.read()
        st.download_button(
            "⬇️ Download report.md", data=text,
            file_name=download_name, mime="text/markdown",
        )

        # Inline relative figure paths as base64 data URLs so Streamlit
        # can display them (it doesn't resolve paths relative to the .md).
        import base64, re
        report_dir = os.path.dirname(report_path)

        def _inline(match: re.Match) -> str:
            alt, rel = match.group(1), match.group(2)
            # Accept both / and \ separators; resolve relative to the report dir
            rel_norm = rel.replace("\\", "/")
            abs_path = os.path.join(report_dir, *rel_norm.split("/"))
            if not os.path.exists(abs_path):
                return f"_(missing figure: {rel})_"
            with open(abs_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            return f"![{alt}](data:image/png;base64,{b64})"

        # Match figures referenced with either / or \ as separator
        text = re.sub(r"!\[([^\]]*)\]\((figures[/\\][^)]+\.png)\)", _inline, text)
        st.markdown(text)


# =============================================================
# PAGE: MISSING PDFS
# =============================================================
elif page == "📤 Missing PDFs":
    st.title("Missing PDFs report")
    st.markdown("Papers that were screened as **included** but the PDF couldn't be downloaded "
                "(extraction fell back to abstract). Send this list to your supervisor for "
                "institutional access lookup.")
    runs = list_completed_runs()
    if not runs:
        st.info("No runs completed yet.")
        st.stop()

    rows = []
    for qid in runs:
        df = get_papers_df(qid)
        missing = df[
            (df["screening_status"] == "included") &
            (df["extraction_source"] == "abstract")
        ]
        for _, r in missing.iterrows():
            rows.append({
                "question_id":  qid,
                "title":        r["title"],
                "authors":      r["authors"],
                "year":         r["year"],
                "doi":          r["doi"],
                "source":       r["source"],
                "venue":        r["venue"],
                "quartile":     r["quartile"],
            })

    if not rows:
        st.success("All included papers have full-text extractions — no missing PDFs.")
        st.stop()

    out = pd.DataFrame(rows)
    st.caption(f"{len(out)} missing PDFs across {out['question_id'].nunique()} question(s)")
    st.dataframe(out, use_container_width=True, height=600)
    st.download_button(
        "⬇️ Download CSV",
        data=out.to_csv(index=False),
        file_name="missing_pdfs.csv",
        mime="text/csv",
    )
