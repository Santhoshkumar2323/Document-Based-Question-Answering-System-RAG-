from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
REGISTRY_FILE = DATA_DIR / "registry.json"
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_COLLECTION_NAME = "decision_rag_chunks"

TOP_K_RETRIEVAL = 50  
TOP_K_RERANKED = 10    
MAX_RERANK_DISTANCE = 0.8

GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_OUTPUT_TOKENS = 2048

CONFIDENCE_THRESHOLDS = {
    "high": 0.4,   
    "medium": 0.8,
}

MIN_EVIDENCE_COUNT = 2