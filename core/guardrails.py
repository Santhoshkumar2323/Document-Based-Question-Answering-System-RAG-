from shared.models import DecisionTrace
from shared.config import MIN_EVIDENCE_COUNT
from shared.logger import setup_logger

logger = setup_logger(__name__)


def apply_guardrails(
    trace: DecisionTrace,
    confidence: str,
) -> str | None:
    """
    RELAXED GUARDRAILS:
    Never refuse. Just warn.
    This allows Gemini to use its internal knowledge when docs fail.
    """

    warnings = []

    if len(trace.used_evidence) < MIN_EVIDENCE_COUNT:
        logger.info("Guardrail note: Low evidence count. Relying on LLM knowledge.")
        warnings.append("Evidence is scarce.")

    if confidence == "Low":
        logger.info("Guardrail note: Low confidence. Relying on LLM knowledge.")
        warnings.append("Document match is weak.")

    if trace.gaps:
        logger.info("Guardrail note: Reasoning gaps detected.")
        warnings.append("Information gaps exist.")

    # We return None (meaning "Go ahead") but we log the warnings.
    # The prompt builder will handle the "Be careful" instruction.
    return None