from dataclasses import dataclass
from typing import List, Dict, Optional


# -----------------------------
# Core data units
# -----------------------------

@dataclass
class Chunk:
    """
    A single chunk of text derived from a document.
    """
    id: str
    doc_id: str
    text: str
    source: str
    page: Optional[int]
    metadata: Dict[str, str]


@dataclass
class RetrievedEvidence:
    """
    A chunk returned from similarity search with a score.
    """
    chunk: Chunk
    score: float


# -----------------------------
# Decision trace (NEW)
# -----------------------------

@dataclass
class DecisionTrace:
    """
    Explicit reasoning trace for an answer.
    """
    used_evidence: List[RetrievedEvidence]
    ignored_evidence: List[RetrievedEvidence]
    gaps: List[str]
    notes: List[str]


# -----------------------------
# Final answer contract
# -----------------------------

@dataclass
class AnswerResponse:
    """
    The final response returned to the user.
    """
    answer: str
    evidence: List[RetrievedEvidence]
    confidence: str
    unknowns: List[str]
