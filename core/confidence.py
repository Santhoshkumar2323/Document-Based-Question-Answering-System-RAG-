from typing import List

from shared.models import RetrievedEvidence
from shared.config import CONFIDENCE_THRESHOLDS
from shared.logger import setup_logger

logger = setup_logger(__name__)


def assess_confidence(evidences: List[RetrievedEvidence]) -> str:
    if not evidences:
        return "Low"

    # We look for the SMALLEST distance (closest match)
    best_score = min(ev.score for ev in evidences)

    # Logic flipped: Lower distance = Higher confidence
    if best_score <= CONFIDENCE_THRESHOLDS["high"]:
        return "High"
    elif best_score <= CONFIDENCE_THRESHOLDS["medium"]:
        return "Medium"
    else:
        return "Low"