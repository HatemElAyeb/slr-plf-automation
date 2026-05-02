import json
import re
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.models import Paper, ScreeningResult, ScreeningStatus
from src.indexer.indexer import QdrantIndexer
from src.llm import get_llm
from config.settings import settings


# Retryable network/transient errors. Keep it broad — we don't want a 12-hour
# overnight run to die because of a single hiccup.
_RETRY_EXCEPTIONS = (
    ConnectionError,    # generic
    TimeoutError,       # generic
    OSError,            # covers httpx/httpcore connection errors
)


INCLUSION_CRITERIA = """
INCLUDE the paper ONLY if it meets ALL of the following:
- Animals: cattle, swine, poultry, sheep, or goats — specifically FARM livestock (NOT pets, wildlife, insects, aquatic animals, or humans)
- Technology: sensors, cameras, accelerometers, RFID, microphones, wearables, IoT devices, or computer vision DIRECTLY APPLIED to farm livestock
- Study type: original PRIMARY research — presents its own experiment, dataset, or system (NOT a literature review, survey, or meta-analysis)
- Goal: monitoring health, behavior, productivity, or welfare of farm livestock

EXCLUDE the paper if ANY of the following apply:
- It is a literature review, systematic review, survey, or meta-analysis
- Animals studied are wildlife, insects, aquatic animals, pets, or humans
- Sensor or AI technology applied to vehicles, crops, plants, or non-livestock subjects
- No original experiment or dataset — purely theoretical or economic
- Published before 2010
"""

SCREENING_PROMPT = ChatPromptTemplate.from_template("""
You are an expert screener for a systematic literature review on Precision Livestock Farming (PLF).

Your task is to decide whether the following paper should be INCLUDED or EXCLUDED based on the criteria below.

IMPORTANT RULE: If the title contains any of these words — "review", "survey", "meta-analysis", "systematic", "overview", "decades", "trends" — you MUST exclude it immediately, as it is not primary research.

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
    def __init__(
        self,
        indexer: QdrantIndexer | None = None,
        criteria: str | None = None,
    ):
        self.llm = get_llm(temperature=0, json_mode=True)
        self.chain = SCREENING_PROMPT | self.llm | StrOutputParser()
        self.indexer = indexer or QdrantIndexer()
        # Use provided criteria or fall back to default
        self.criteria = criteria if criteria is not None else INCLUSION_CRITERIA

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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _invoke_chain(self, paper: Paper) -> str:
        """LLM call wrapped with retry — bounces back from transient network errors."""
        return self.chain.invoke({
            "criteria": self.criteria,
            "title": paper.title,
            "abstract": paper.abstract[:2000],
        })

    def screen_paper(self, paper: Paper) -> ScreeningResult:
        try:
            raw = self._invoke_chain(paper)
        except Exception as e:
            # Even after retries it failed — return EXCLUDED with low confidence
            # so the long run continues instead of crashing.
            print(f"  [Screener] Failed after retries on {paper.id}: {type(e).__name__}")
            return ScreeningResult(
                decision=ScreeningStatus.EXCLUDED,
                confidence=0.0,
                reason=f"LLM error after retries: {type(e).__name__}",
            )
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
