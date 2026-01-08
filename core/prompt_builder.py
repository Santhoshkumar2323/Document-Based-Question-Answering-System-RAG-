from typing import List, Dict, Optional
from shared.logger import setup_logger

logger = setup_logger(__name__)


def build_prompt(
    reasoning: dict, 
    question: str, 
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    
    
    # ALWAYS try to answer.

    intent = reasoning.get("intent", "general")
    documents = reasoning.get("documents", [])

    # 1. Build Evidence Block
    blocks = []
    for doc in documents:
        block = [f"Document: {doc['source']}"]
        for r in doc["regions"]:
            block.append(
                f"- Page {r['page']} | Score {r['score']:.3f}\n{r['text']}"
            )
        blocks.append("\n".join(block))

    evidence_text = "\n\n".join(blocks)
    if not evidence_text:
        evidence_text = "No direct evidence found in local documents."

    # 2. Build History Block
    history_text = "No previous conversation."
    if history:
        history_lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "AI"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    return f"""
You are a highly intelligent Research Assistant. You have access to a library of local documents, but you also possess vast general knowledge.

---------------------------------------------------
CORE OBJECTIVE:
Answer the user's question as comprehensively as possible.

RULES OF ENGAGEMENT:
1. **Priority #1 (Local Data):** ALWAYS prioritize information found in the [NEW EVIDENCE] section.
2. **Priority #2 (General Knowledge):** If the evidence is missing, incomplete, or unrelated, use your internal training data to answer.
3. **Transparency:** - If you use the documents, cite them (e.g., "According to Document A...").
   - If you use your own knowledge, say: *"Based on general knowledge..."* or *"The documents don't mention this, but typically..."*
4. **Context:** Use the [CONVERSATION HISTORY] to understand pronouns (it, they, that).

---------------------------------------------------
CONVERSATION HISTORY:
{history_text}

---------------------------------------------------
NEW EVIDENCE (Local Documents):
{evidence_text}

---------------------------------------------------
CURRENT QUESTION:
{question}

---------------------------------------------------
YOUR ANSWER:
"""