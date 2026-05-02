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
        "📄 Report",
        "🔍 Compare questions",
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
        config = load_run_config(q["id"])
        status = "✅ run completed" if config else "⏳ not yet run"
        with st.expander(f"**{q['id']}** ({q['category']}) — {status}"):
            st.write(q["text"])
            if config:
                r = config.get("results", {})
                c = st.columns(4)
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
    display_cols = ["year", "title", "authors", "source", "venue", "quartile",
                    "screening_status", "extraction_source"]
    st.dataframe(filt[display_cols], use_container_width=True, height=500)

    # Drill-down
    st.markdown("### Paper detail")
    if not filt.empty:
        idx = st.number_input("Row index to inspect (from filtered table above)", 0, len(filt) - 1, 0)
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


# =============================================================
# PAGE: REPORT
# =============================================================
elif page == "📄 Report":
    st.title("Report viewer")
    runs = list_completed_runs()
    if not runs:
        st.info("No runs completed yet.")
        st.stop()

    qid = st.selectbox("Question", runs)
    report_path = os.path.join("data", "runs", qid, "report.md")

    if not os.path.exists(report_path):
        st.warning("Report not generated yet for this question.")
        if st.button("Generate report now"):
            with st.spinner("Generating report..."):
                from src.synthesis import generate_report
                question_text = next((q["text"] for q in QUESTIONS if q["id"] == qid), qid)
                generate_report(qid, question_text)
            st.rerun()
    else:
        with open(report_path, encoding="utf-8") as f:
            text = f.read()
        st.download_button(
            "⬇️ Download report.md", data=text,
            file_name=f"{qid}_report.md", mime="text/markdown",
        )
        st.markdown(text)


# =============================================================
# PAGE: COMPARE QUESTIONS
# =============================================================
elif page == "🔍 Compare questions":
    st.title("Compare questions")
    runs = list_completed_runs()
    if len(runs) < 2:
        st.info("Need at least 2 completed runs to compare.")
        st.stop()

    selected = st.multiselect("Pick questions", runs, default=runs)
    if not selected:
        st.stop()

    rows = []
    for qid in selected:
        s = get_stats(qid)
        p = s["prisma"]
        rows.append({
            "Question":  qid,
            "Identified": p["identified"],
            "Included":   p["included"],
            "Excluded":   p["excluded"],
            "Extracted":  p["extracted_total"],
            "Q1 papers":  s["quartile_distribution"].get("Q1", 0),
            "Top sensor": next(iter(s["top_sensor_types"]), "—"),
            "Top method": next(iter(s["top_ml_methods"]), "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Stacked sensor bars across questions
    st.markdown("### Sensor types per question")
    bars = []
    for qid in selected:
        s = get_stats(qid)
        for sensor, count in s["top_sensor_types"].items():
            bars.append({"question": qid, "sensor": sensor, "count": count})
    if bars:
        st.plotly_chart(
            px.bar(pd.DataFrame(bars), x="question", y="count", color="sensor",
                   title="Sensor types per question"),
            use_container_width=True,
        )


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
