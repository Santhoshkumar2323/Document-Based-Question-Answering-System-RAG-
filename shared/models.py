from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    page: Optional[int]
    metadata: Dict[str, str]

@dataclass
class RetrievedEvidence:
    chunk: Chunk
    score: float

@dataclass
class DecisionTrace:
    used_evidence: List[RetrievedEvidence]
    ignored_evidence: List[RetrievedEvidence]
    gaps: List[str]
    notes: List[str]

@dataclass
class AnswerResponse:
    answer: str
    evidence: List[RetrievedEvidence]
    confidence: str
    unknowns: List[str]
