from typing import List, Dict
from collections import defaultdict

from shared.models import RetrievedEvidence
from shared.logger import setup_logger

logger = setup_logger(__name__)


class ReasoningEngine:
    """
    Converts evidence into a reasoning plan.
    This is the intelligence layer.
    """

    def analyze(
        self,
        question: str,
        evidences: List[RetrievedEvidence],
    ) -> Dict:
        if not evidences:
            return {
                "can_answer": False,
                "reason": "No evidence",
            }

        # Group by document
        by_doc = defaultdict(list)
        for ev in evidences:
            by_doc[ev.chunk.source].append(ev)

        documents = []
        for source, evs in by_doc.items():
            regions = []
            for ev in evs:
                regions.append({
                    "page": ev.chunk.page,
                    "score": ev.score,
                    "text": ev.chunk.text,
                })

            documents.append({
                "source": source,
                "regions": regions,
            })

        # Determine question intent (simple but explicit)
        q = question.lower()
        if any(k in q for k in ["theme", "overall", "main", "summary"]):
            intent = "synthesis"
        else:
            intent = "factual"

        return {
            "can_answer": True,
            "intent": intent,
            "documents": documents,
        }
