from typing import List

from shared.models import RetrievedEvidence, DecisionTrace
from shared.config import MIN_SIMILARITY_SCORE
from shared.logger import setup_logger

logger = setup_logger(__name__)


def build_decision_trace(
    evidences: List[RetrievedEvidence],
    question: str,
) -> DecisionTrace:
    """
    Separates used vs ignored evidence and identifies gaps.
    """

    used: List[RetrievedEvidence] = []
    ignored: List[RetrievedEvidence] = []

    for ev in evidences:
        if ev.score <= MIN_SIMILARITY_SCORE:
            used.append(ev)
        else:
            ignored.append(ev)

    gaps: List[str] = []
    notes: List[str] = []

    if not used:
        gaps.append("No evidence directly relevant to the question.")

    if any("why" in question.lower() for _ in [0]) and used:
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
