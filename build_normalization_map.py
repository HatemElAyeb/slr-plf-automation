"""
Phase 0c — Build a normalization map for sensor_types, ml_methods, and
animal_species across ALL 6 question collections, using a single LLM call.

Output: data/normalization_map.json with structure:
{
  "sensor_types": {
    "camera":          ["camera", "rgb camera", "rgb video", "video", "cctv", ...],
    "accelerometer":   ["accelerometer", "tri-axial accelerometer", ...],
    ...
  },
  "ml_methods":  { ... same shape ... },
  "animal_species": {
    "cattle":          ["cattle", "dairy cattle", "beef cattle", "cows", ...],
    "_NON_LIVESTOCK_": ["songbird species", "common marmosets", ...]
  }
}

Run: python build_normalization_map.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.indexer.indexer import QdrantIndexer
from research_questions import QUESTIONS
from config.settings import settings


OUT_PATH = os.path.join("data", "normalization_map.json")

PROMPT = ChatPromptTemplate.from_template("""
You are an expert in Precision Livestock Farming (PLF). Your task is to
group raw extracted values into canonical (normalized) categories.

CATEGORY: {category}

RAW VALUES (one per line):
{raw_values}

TASK:
1. Group similar/synonymous values under one canonical lowercase name.
2. Use simple, broadly understood canonical names (e.g. "camera" not "imaging_device").
3. Be aggressive in merging variants. Example: ["camera", "rgb camera", "rgb video",
   "video", "cctv", "video surveillance", "thermal camera"] -> "camera".
4. Keep distinct technologies separate. Example: don't merge "accelerometer" and "gyroscope".
5. {special_rule}

Output STRICT JSON of the form:
{{
  "<canonical_name>": ["raw_value_1", "raw_value_2", ...],
  ...
}}

Every raw value must appear EXACTLY ONCE in the output across all canonical groups.
Use lowercase canonical names. Use the EXACT raw values as listed above (do not edit them).

Return ONLY the JSON object. No explanation, no preamble.
""")

SPECIAL_RULES = {
    "sensor_types": (
        "Group cameras (RGB, video, thermal, depth, infrared) under \"camera\". "
        "Group all accelerometer variants. Keep GPS, RFID, microphone, "
        "and biosensors as separate canonical groups."
    ),
    "ml_methods": (
        "Group YOLO variants (yolov5, yolov8, yolov11, etc.) under \"yolo\". "
        "Group CNN variants under \"cnn\". Keep \"deep learning\" as a category "
        "only when no specific method is mentioned. Keep classical ML separate "
        "(SVM, Random Forest, Logistic Regression, Decision Tree, k-NN)."
    ),
    "animal_species": (
        "Group all cattle types under \"cattle\" (dairy cattle, dairy cows, "
        "beef cattle, beef bulls, holstein, holstein friesian, calves, cows, "
        "korean native cows, etc.). Group all pigs/swine under \"pigs\". "
        "Group all poultry under \"poultry\". Keep sheep, goats separate. "
        "**IMPORTANT**: Any value that is NOT a farm livestock species "
        "(e.g. songbird, marmoset, fish, wildlife, insects) MUST go under "
        "the canonical key \"_non_livestock_\" so it can be excluded later."
    ),
}


def collect_unique_values(field: str) -> list[str]:
    """Pull all unique non-empty values for the given field across the 6 collections."""
    seen: set[str] = set()
    for q in QUESTIONS:
        try:
            ix = QdrantIndexer(collection_suffix="_" + q["id"])
        except Exception:
            continue
        points = ix.client.scroll(
            collection_name=ix.collection_name,
            limit=100000,
            with_payload=True,
            with_vectors=False,
        )[0]
        for p in points:
            for v in (p.payload.get(field) or []):
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s:
                        seen.add(s)
    return sorted(seen)


def normalize_with_llm(category: str, raw_values: list[str]) -> dict[str, list[str]]:
    if not raw_values:
        return {}
    llm = ChatOpenAI(
        model=settings.openai_screening_model,
        api_key=settings.openai_api_key,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "category":     category,
        "raw_values":   "\n".join(raw_values),
        "special_rule": SPECIAL_RULES.get(category, ""),
    })
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object from response
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


if __name__ == "__main__":
    map_out: dict[str, dict[str, list[str]]] = {}

    for category in ["sensor_types", "ml_methods", "animal_species"]:
        print(f"\n=== {category} ===")
        raw = collect_unique_values(category)
        print(f"  {len(raw)} unique raw values")
        if not raw:
            continue
        groups = normalize_with_llm(category, raw)
        print(f"  → {len(groups)} canonical groups")
        for canonical, variants in sorted(groups.items()):
            preview = ", ".join(variants[:3])
            more = f" ... (+{len(variants)-3})" if len(variants) > 3 else ""
            print(f"    {canonical:<25} = [{preview}{more}]")
        map_out[category] = groups

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(map_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_PATH}")
