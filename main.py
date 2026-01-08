import sys
from typing import List, Dict

from core.index_pipeline import run_indexing_pipeline
from core.retriever import Retriever
from core.decision_trace import build_decision_trace
from core.reasoning_engine import ReasoningEngine
from core.prompt_builder import build_prompt
from core.answer_engine import AnswerEngine
from core.confidence import assess_confidence
from core.guardrails import apply_guardrails
from shared.logger import setup_logger

logger = setup_logger("main")


def print_decision_trace(trace):
    print("\n--- Decision Trace ---")

    print("\nUsed Evidence:")
    for ev in trace.used_evidence:
        c = ev.chunk
        print(
            f"- {c.source}, page {c.page}, "
            f"score={ev.score:.3f}"
        )

    if trace.ignored_evidence:
        print("\nIgnored Evidence:")
        for ev in trace.ignored_evidence:
            c = ev.chunk
            print(
                f"- {c.source}, page {c.page}, "
                f"score={ev.score:.3f}"
            )

    if trace.gaps:
        print("\nGaps:")
        for g in trace.gaps:
            print(f"- {g}")

    if trace.notes:
        print("\nNotes:")
        for n in trace.notes:
            print(f"- {n}")

    print("----------------------\n")


def run_query(question: str) -> None:
    # Single query mode does not need history
    retriever = Retriever()
    reasoner = ReasoningEngine()
    engine = AnswerEngine()

    candidates = retriever.retrieve(question)
    trace = build_decision_trace(candidates, question)

    confidence = assess_confidence(trace.used_evidence)
    refusal = apply_guardrails(trace, confidence)

    if refusal:
        print(refusal)
        print(f"\nConfidence: {confidence}")
        print_decision_trace(trace)
        return

    reasoning = reasoner.analyze(question, trace.used_evidence)
    # Pass empty history for single queries
    prompt = build_prompt(reasoning, question, history=[])

    answer = engine.generate(prompt)

    print("\n" + answer)
    print(f"\nConfidence: {confidence}")
    print_decision_trace(trace)


def run_chat() -> None:
    print("Interactive RAG Chat (with Contextual Memory)")
    print("Type 'exit' or 'quit' to stop.\n")

    retriever = Retriever()
    reasoner = ReasoningEngine()
    engine = AnswerEngine()

    # FORENSIC UPDATE: Initialize Memory
    history: List[Dict[str, str]] = []

    while True:
        try:
            question = input("Question: ").strip()
        except EOFError:
            print("\nExiting chat.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        # =====================================================
        # STEP 1: REWRITE QUERY (The Fix)
        # =====================================================
        # If we have history, we ask the Brain to clarify the question first.
        search_query = question
        if history:
            print("Thinking...", end="\r") # UI Feedback
            search_query = engine.rewrite_query(question, history)
            print(f"Refined Query: {search_query}") # Show the user what changed

        # =====================================================
        # STEP 2: SEARCH (Using the BETTER query)
        # =====================================================
        # Use the refined 'search_query' for retrieval
        candidates = retriever.retrieve(search_query)
        
        # Trace also uses the refined query for context (logging purposes)
        trace = build_decision_trace(candidates, search_query)

        confidence = assess_confidence(trace.used_evidence)
        refusal = apply_guardrails(trace, confidence)

        if refusal:
            print(refusal)
            print(f"Confidence: {confidence}")
            print_decision_trace(trace)
            # Even if refused, we don't add to history to keep it clean
            continue

        # =====================================================
        # STEP 3: REASON & ANSWER (Using original question + history)
        # =====================================================
        # We pass the original 'question' here because the LLM needs to know
        # what the user *actually* said (including tone/nuance), not the robot search query.
        reasoning = reasoner.analyze(question, trace.used_evidence)
        prompt = build_prompt(reasoning, question, history)

        answer = engine.generate(prompt)

        print(answer)
        print(f"Confidence: {confidence}")
        print_decision_trace(trace)

        # 3. Update Memory
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        # Keep memory size manageable (last 10 interactions)
        if len(history) > 10:
            history = history[-10:]


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py index")
        print('  python main.py query "your question"')
        print("  python main.py chat")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "index":
        run_indexing_pipeline()

    elif command == "query":
        question = " ".join(sys.argv[2:])
        run_query(question)

    elif command == "chat":
        run_chat()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()