"""
Generate Sankey / alluvial flow diagrams for a question.

Each diagram shows how papers flow between consecutive categorical
dimensions (e.g. Animal -> Sensor -> ML method). Stages can be any of
the standard extraction fields (animal_species, sensor_types,
ml_methods) or per-question custom fields (defined in
research_questions.py).
"""
import os
import re
from collections import Counter, defaultdict

import plotly.graph_objects as go

from src.synthesis.statistics import _normalize, NON_LIVESTOCK_TAG


# Fields that go through the curated normalization map
_NORMALIZED_FIELDS = {"animal_species", "sensor_types", "ml_methods"}


def _values_for_paper(paper: dict, field: str) -> list[str]:
    """
    Return the (normalized, lowercased) list of values for `field` on a paper.

    - Standard fields are read directly off the payload.
    - Custom fields are read from payload['custom_fields'][field].
    - Standard fields go through _normalize() to consolidate variants.
    - Empty / non-livestock entries are dropped.
    """
    if field in _NORMALIZED_FIELDS:
        raw = paper.get(field) or []
    else:
        raw = (paper.get("custom_fields") or {}).get(field) or []

    out: list[str] = []
    for v in raw:
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if not s:
            continue
        if field in _NORMALIZED_FIELDS:
            n = _normalize(s, field)
            if n is None:
                continue  # _non_livestock_, drop
            out.append(n)
        else:
            out.append(s)
    return out


def _topn_per_stage(papers: list[dict], stages: list[str], max_per_stage: int) -> list[set[str]]:
    """For each stage, return the set of top-N most frequent values across the corpus."""
    keep: list[set[str]] = []
    for field in stages:
        c = Counter()
        for p in papers:
            for v in _values_for_paper(p, field):
                c[v] += 1
        keep.append({k for k, _ in c.most_common(max_per_stage)})
    return keep


def compute_sankey_data(
    papers: list[dict],
    stages: list[str],
    max_per_stage: int = 8,
) -> dict:
    """
    Build (nodes, links) for a multi-stage Sankey.

    For each paper, every cartesian product of values across consecutive
    stages contributes one unit of flow.

    Returns:
        {
          "labels":  list[str],            # node labels (prefixed by stage idx)
          "node_stage": list[int],         # stage index per node (for x positioning)
          "source":  list[int],
          "target":  list[int],
          "value":   list[int],
        }
    """
    if len(stages) < 2:
        raise ValueError("Sankey needs at least 2 stages")

    # 1. Collect top-N values per stage so the diagram stays readable
    keep = _topn_per_stage(papers, stages, max_per_stage)

    # 2. Build node list: one node per (stage_index, value).
    #    Node labels are kept human-readable; we use a (stage, value) key for indexing.
    node_index: dict[tuple[int, str], int] = {}
    labels: list[str] = []
    node_stage: list[int] = []

    def _node(stage_i: int, val: str) -> int:
        key = (stage_i, val)
        if key in node_index:
            return node_index[key]
        idx = len(labels)
        labels.append(val)
        node_stage.append(stage_i)
        node_index[key] = idx
        return idx

    # 3. Build link counts between consecutive stages
    link_counts: dict[tuple[int, int], int] = defaultdict(int)
    for p in papers:
        per_stage_vals: list[list[str]] = []
        for i, field in enumerate(stages):
            vals = _values_for_paper(p, field)
            vals = [v for v in vals if v in keep[i]]
            per_stage_vals.append(list(set(vals)))  # dedupe within paper

        # For each consecutive pair, contribute flow per cartesian product
        for i in range(len(stages) - 1):
            for s in per_stage_vals[i]:
                for t in per_stage_vals[i + 1]:
                    s_idx = _node(i, s)
                    t_idx = _node(i + 1, t)
                    link_counts[(s_idx, t_idx)] += 1

    sources, targets, values = [], [], []
    for (s, t), v in sorted(link_counts.items(), key=lambda kv: -kv[1]):
        sources.append(s)
        targets.append(t)
        values.append(v)

    return {
        "labels":     labels,
        "node_stage": node_stage,
        "source":     sources,
        "target":     targets,
        "value":      values,
    }


def make_sankey_figure(data: dict, title: str, stages: list[str]) -> go.Figure:
    """Build a Plotly Sankey figure from the data dict produced above."""
    if not data["labels"] or not data["source"]:
        # Fallback: empty annotation
        fig = go.Figure()
        fig.update_layout(
            title=title + " (no data)",
            annotations=[dict(text="No flow data available", showarrow=False)],
        )
        return fig

    # Position nodes horizontally by stage (Plotly requires x in (0, 1))
    n_stages = max(data["node_stage"]) + 1
    x_pos = [
        (stage_i + 0.5) / n_stages for stage_i in data["node_stage"]
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.4)", width=0.5),
            label=data["labels"],
            x=x_pos,
        ),
        link=dict(
            source=data["source"],
            target=data["target"],
            value=data["value"],
        ),
    ))

    # Stage labels above the diagram
    stage_annotations = []
    for i, name in enumerate(stages):
        stage_annotations.append(dict(
            x=(i + 0.5) / len(stages),
            y=1.06,
            xref="paper", yref="paper",
            text=f"<b>{name.replace('_', ' ').title()}</b>",
            showarrow=False, font=dict(size=12),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=stage_annotations,
        margin=dict(t=70, l=20, r=20, b=20),
        height=500,
    )
    return fig


def save_figure_png(fig: go.Figure, path: str, scale: float = 2.0) -> None:
    """Save the figure as a PNG (requires `kaleido`)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_image(path, format="png", scale=scale, width=1200, height=600)


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def generate_figures_for_question(question: dict, papers: list[dict]) -> list[dict]:
    """
    Generate every Sankey diagram declared in question["sankey_diagrams"].
    Returns a list of {"title": ..., "stages": ..., "rel_path": ..., "data": ...}
    so the caller (report) can embed images and (Streamlit) can re-render
    interactively from the same data.
    """
    qid = question["id"]
    diagrams = question.get("sankey_diagrams") or []
    if not diagrams or not papers:
        return []

    out_dir = os.path.join("data", "runs", qid, "figures")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for spec in diagrams:
        title = spec["title"]
        stages = spec["stages"]
        max_per = spec.get("max_per_stage", 8)
        data = compute_sankey_data(papers, stages, max_per_stage=max_per)
        fig = make_sankey_figure(data, title, stages)

        fname = f"sankey_{_slugify(title)}.png"
        abs_path = os.path.join(out_dir, fname)
        try:
            save_figure_png(fig, abs_path)
            saved = True
        except Exception as e:
            print(f"  [Figures] {qid} '{title}' PNG export failed: {type(e).__name__}: {e}")
            saved = False

        results.append({
            "title":    title,
            "stages":   stages,
            "rel_path": os.path.join("figures", fname) if saved else None,
            "data":     data,
        })
    return results
