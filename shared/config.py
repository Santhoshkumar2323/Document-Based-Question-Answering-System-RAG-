from pathlib import Path


# =============================
# Project paths
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"

# NEW: registry file (document lifecycle source of truth)
REGISTRY_FILE = DATA_DIR / "registry.json"

# NEW: BM25 Index file (Keyword search source of truth)
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"


# =============================
# Chunking configuration
# =============================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# =============================
# Embedding & Reranking configuration
# =============================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# NEW: Cross-Encoder for high-precision reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =============================
# Vector store configuration
# =============================

CHROMA_COLLECTION_NAME = "decision_rag_chunks"

# NEW: Retrieve MORE candidates initially (Hybrid pool), then rerank down
TOP_K_RETRIEVAL = 50   # How many to fetch from Vector + BM25
TOP_K_RERANKED = 10    # How many to give to the LLM after reranking

# Reranker Score Threshold (Distance Metric)
# Cross-Encoders output 0 (Bad) to 1 (Good).
# convert this to Distance: 1 - Score.
#  0.0 is Perfect Match. 1.0 is Terrible.
MIN_SIMILARITY_SCORE = 0.9  # Reject anything with < 20% relevance


# =============================
# LLM (Gemini) configuration
# =============================

GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_OUTPUT_TOKENS = 2048


# =============================
# Confidence scoring thresholds
# =============================
# LOWER distance is better.

CONFIDENCE_THRESHOLDS = {
    "high": 0.4,   # Any distance < 0.4 is Great
    "medium": 0.8, # Any distance < 0.8 is Good enough
}


# =============================
# Guardrail configuration
# =============================

MIN_EVIDENCE_COUNT = 1