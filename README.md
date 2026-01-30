What this project is
This is a Retrieval-Augmented Generation (RAG) system designed to answer questions using a local documents, while explicitly mentioning evidence quality, gaps, and confidence.
It is a document reasoning pipeline with traceability.

This system is designed to answer a harder question:
“Do we actually have enough evidence to answer this?”

It makes retrieval quality and reasoning gaps.

Architecture:
The system is split into four  layers:

1. Document lifecycle & indexing
Deterministic document IDs using content hashing
Registry tracks document state on disk
Automatic detection of:
new documents
deleted documents
registry ↔ vector-store inconsistencies
Safe re-indexing and cleanup

Documents are ingested → normalized → chunked → embedded → indexed.

2. Hybrid retrieval
Querying uses three stages:
Vector search (semantic recall via sentence embeddings)
BM25 keyword search (lexical recall)
Cross-encoder reranking (precision scoring)

Results from vector + BM25 are:
fused
deduplicated
reranked
trimmed to a high-quality evidence set
This avoids both semantic drift and keyword brittleness.

3. Reasoning & traceability
Before calling the LLM, the system constructs a Decision Trace:
Evidence used vs ignored
Gaps where documents do not support the question
Notes when answers rely on weak or single-source evidence
This trace is preserved and shown alongside the answer.

4. Answer generation (LLM boundary)
The LLM is treated as a thin reasoning layer, not a source of truth.
Rules enforced in the prompt:
Prefer local document evidence
Transparently fall back to general knowledge if documents are insufficient
Use conversation history to resolve ambiguity
Avoid silent hallucination
The system always attempts an answer, but never hides uncertainty.

Retrieval & reasoning flow (step-by-step)

User question received
(Chat mode) Question is rewritten into a standalone search query
Hybrid retrieval (Vector + BM25)
Cross-encoder reranking
Evidence split into:
usable evidence
ignored evidence
Confidence score computed from retrieval distances
Guardrails assess evidence sufficiency
Reasoning plan built from evidence
Prompt assembled with:
evidence
history
transparency rules
LLM generates answer
Decision trace and confidence returned

How to use
1. Index documents
Place PDFs or TXT files in data/docs/, then run:
python main.py index

2. Ask a single question
python main.py query "Your question here"

3. Interactive chat mode
python main.py chat

Chat mode maintains short-term memory and rewrites follow-up questions into standalone search queries for better retrieval.

Chat mode maintains short-term memory and rewrites follow-up questions into standalone search queries for better retrieval.

Limitations:
If documents are sparse or poorly written, answers rely more on LLM knowledge
Cross-encoder reranking increases latency
Confidence is heuristic, not probabilistic
No claim of factual correctness beyond available evidence

LLMs are used for:
query rewriting
reasoning synthesis
answer generation
They are not used for:
retrieval
scoring
evidence selection
document ingestion
All retrieval, filtering, and confidence logic is deterministic.

Intended use
Internal knowledge bases
Research document analysis
Technical or analytical document QA

-This is a working, modular RAG system focused on correctness, traceability, and reasoning quality.

