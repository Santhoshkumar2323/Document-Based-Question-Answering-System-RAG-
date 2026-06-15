from typing import List

from shared.models import RetrievedEvidence, DecisionTrace
from shared.config import MAX_RERANK_DISTANCE
from shared.logger import setup_logger

logger = setup_logger(__name__)


def build_decision_trace(
    evidences: List[RetrievedEvidence],
    question: str,
) -> DecisionTrace:
    used: List[RetrievedEvidence] = []
    ignored: List[RetrievedEvidence] = []

    for ev in evidences:
        if ev.score <= MAX_RERANK_DISTANCE:
            used.append(ev)
        else:
            ignored.append(ev)

    gaps: List[str] = []
    notes: List[str] = []

    if not used:
        gaps.append("No evidence directly relevant to the question.")

    if "why" in question.lower() and used:
        notes.append(
            "Question asks for explanation ('why'), "
            "but documents appear descriptive rather than causal."
        )

    if len(used) == 1:
        notes.append("Answer is based on a single source only.")

    logger.info(
        f"DecisionTrace built: used={len(used)}, ignored={len(ignored)}"
    )

    return DecisionTrace(
        used_evidence=used,
        ignored_evidence=ignored,
        gaps=gaps,
        notes=notes,
    )
