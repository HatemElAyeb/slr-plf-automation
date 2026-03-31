import json
import re
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.models import Paper, ScreeningResult, ScreeningStatus
from src.indexer.indexer import QdrantIndexer
from config.settings import settings


INCLUSION_CRITERIA = """
INCLUDE the paper ONLY if it explicitly addresses ALL of the following:
- Animals: cattle, swine, poultry, sheep, goats, or other farm livestock (NOT pets, wildlife, or humans)
- Technology: sensors, cameras, accelerometers, RFID, microphones, wearables, IoT devices, or computer vision APPLIED TO livestock
- Goal: monitoring health, behavior, productivity, or welfare of farm livestock specifically

EXCLUDE the paper if ANY of the following apply:
- Sensor or AI technology applied to vehicles, robots, humans, or non-livestock subjects
- Focuses only on crops, plants, or aquaculture (fish/seafood)
- No direct application to farm animals
- No sensor or technology component (purely economic or social studies)
- Published before 2010
- Not a peer-reviewed research article (editorials, conference abstracts without methodology)
"""

SCREENING_PROMPT = ChatPromptTemplate.from_template("""
You are an expert screener for a systematic literature review on Precision Livestock Farming (PLF).

Your task is to decide whether the following paper should be INCLUDED or EXCLUDED based on the criteria below.

CRITERIA:
{criteria}

PAPER TITLE: {title}

ABSTRACT: {abstract}

Respond with a JSON object only, no explanation outside the JSON:
{{
  "decision": "included" or "excluded",
  "confidence": a number between 0.0 and 1.0,
  "reason": "one sentence explaining your decision"
}}
""")


class AbstractScreener:
    def __init__(self, indexer: QdrantIndexer | None = None):
        self.llm = ChatOllama(
            model=settings.screening_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            format="json",
        )
        self.chain = SCREENING_PROMPT | self.llm | StrOutputParser()
        self.indexer = indexer or QdrantIndexer()

    def _parse_response(self, raw: str) -> ScreeningResult:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group() if match else raw)

            decision_str = data.get("decision", "excluded").lower().strip()
            decision = (
                ScreeningStatus.INCLUDED
                if decision_str == "included"
                else ScreeningStatus.EXCLUDED
            )
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "")
            return ScreeningResult(decision=decision, confidence=confidence, reason=reason)

        except Exception:
            return ScreeningResult(
                decision=ScreeningStatus.EXCLUDED,
                confidence=0.0,
                reason="Failed to parse LLM response",
            )

    def screen_paper(self, paper: Paper) -> ScreeningResult:
        raw = self.chain.invoke({
            "criteria": INCLUSION_CRITERIA,
            "title": paper.title,
            "abstract": paper.abstract[:2000],
        })
        return self._parse_response(raw)

    def screen_all(self, papers: list[Paper]) -> list[tuple[Paper, ScreeningResult]]:
        results = []
        for paper in tqdm(papers, desc="Screening"):
            result = self.screen_paper(paper)
            self.indexer.update_screening(
                paper_id=paper.id,
                status=result.decision.value,
                confidence=result.confidence,
                reason=result.reason,
            )
            results.append((paper, result))
        return results

    def screen_pending(self) -> list[tuple[Paper, ScreeningResult]]:
        points = self.indexer.get_points_by_status(ScreeningStatus.PENDING)
        papers = [self.indexer._point_to_paper(p) for p in points]
        print(f"[Screener] Found {len(papers)} pending papers to screen")
        return self.screen_all(papers)
