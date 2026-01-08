import os
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

from shared.config import GEMINI_MODEL_NAME, MAX_OUTPUT_TOKENS
from shared.logger import setup_logger

# Load env vars deterministically
load_dotenv(override=True)

logger = setup_logger(__name__)


class AnswerEngine:
    """
    Thin, explicit LLM boundary.
    Now handles both answering and query rewriting.
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        logger.info("Initializing Gemini client with explicit API key")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Execute a single LLM call and return text.
        NEVER returns None.
        """
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            # Primary path
            if hasattr(response, "text") and response.text:
                return response.text.strip()

            # Fallback path (SDK schema drift)
            if hasattr(response, "candidates"):
                parts = response.candidates[0].content.parts
                text = "".join(
                    p.text for p in parts if hasattr(p, "text")
                )
                if text.strip():
                    return text.strip()

            logger.error("Gemini response had no usable text")
            return "LLM response contained no usable text."

        except Exception as e:
            logger.exception("Gemini API call failed")
            return f"LLM error: {str(e)}"

    def rewrite_query(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        FORENSIC ADDITION:
        Uses the LLM to rewrite a follow-up question into a standalone search query.
        """
        if not history:
            return question  # No history? No need to rewrite.

        # Convert history to simple text format for the prompt
        history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])

        prompt = f"""
        You are a Search Query Refiner.
        Your job is to rewrite the User's last question into a standalone keyword-rich search query.
        
        RULES:
        1. Resolve pronouns (it, they, that, there) using the History.
        2. Keep it concise.
        3. Do NOT answer the question. JUST output the new query.
        
        --------------------
        HISTORY:
        {history_text}
        
        CURRENT QUESTION:
        {question}
        --------------------
        
        REFINED SEARCH QUERY:
        """
        
        # We use the existing generate method
        refined = self.generate(prompt)
        
        # Clean up if the LLM adds prefixes like "Refined Query:"
        refined = refined.replace("Refined Search Query:", "").replace("Refined Query:", "").strip()
        
        logger.info(f"Query Rewritten: '{question}' -> '{refined}'")
        return refined