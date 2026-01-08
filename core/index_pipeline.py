import pickle
from rank_bm25 import BM25Okapi

from shared.logger import setup_logger
from shared.config import DOCS_DIR, BM25_INDEX_FILE
from shared.registry import (
    scan_documents,
    load_registry,
    save_registry,
    diff_registry,
)
from core.ingest import ingest_new_documents
from core.chunker import chunk_documents
from core.embed_store import VectorStore

logger = setup_logger(__name__)


def _rebuild_bm25_index(store: VectorStore) -> None:
    """
    Forensic Step:
    Fetches ALL text from the vector store to rebuild the BM25 index.
    This guarantees that Vector Search and Keyword Search are always in sync.
    """
    logger.info("Rebuilding BM25 keyword index from valid chunks...")

    # Fetch all documents currently in the store
    # We access the underlying collection directly for efficiency
    try:
        data = store.collection.get(include=["documents", "metadatas"])
        texts = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not texts:
            logger.warning("No documents found in store. BM25 index will be empty.")
            return

        # Tokenize for BM25 (simple whitespace split is sufficient for this level)
        tokenized_corpus = [doc.lower().split() for doc in texts]
        
        bm25 = BM25Okapi(tokenized_corpus)

        # We must save the map between BM25 index and Chunk IDs
        # because BM25 just returns an integer index.
        chunk_map = {
            i: {
                "id": f"{m['doc_id']}:{m['chunk_index']}",
                "text": t,
                "metadata": m
            }
            for i, (t, m) in enumerate(zip(texts, metadatas))
        }

        # Persist to disk
        payload = {
            "model": bm25,
            "map": chunk_map
        }

        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(payload, f)
        
        logger.info(f"BM25 index rebuilt with {len(texts)} chunks and saved.")

    except Exception as e:
        logger.exception(f"Failed to rebuild BM25 index: {e}")


def run_indexing_pipeline() -> None:
    """
    Orchestrates full indexing lifecycle with registry–vector consistency:
    - Detects deleted documents → deletes vectors
    - Detects new documents → ingest → chunk → embed
    - Detects registry docs missing vectors → re-index
    - Updates registry
    - FINAL STEP: Rebuilds BM25 Index
    """
    logger.info("Starting indexing pipeline")

    # 1. Scan current filesystem
    scanned = scan_documents(DOCS_DIR)

    # 2. Load registry (previous state)
    registry = load_registry()

    # 3. Diff filesystem vs registry
    diff = diff_registry(scanned, registry)

    new_docs = diff["new"]
    deleted_docs = diff["deleted"]

    store = VectorStore()

    # 4. Handle deleted documents
    for doc_id, info in deleted_docs.items():
        logger.info(f"Removing deleted document: {info.get('filename', doc_id)}")
        store.delete_document(doc_id)
        registry.pop(doc_id, None)

    # 5. Detect registry docs that are missing vectors (CRITICAL FIX)
    repair_docs = {}
    for doc_id, info in registry.items():
        if not store.document_exists(doc_id):
            logger.warning(
                f"Registry document missing vectors, scheduling reindex: {info.get('filename', doc_id)}"
            )
            repair_docs[doc_id] = scanned.get(doc_id)

    # 6. Merge truly new docs + repair docs
    docs_to_index = {**new_docs, **repair_docs}

    if docs_to_index:
        records = ingest_new_documents(docs_to_index)
        chunks = chunk_documents(records)
        store.index_chunks(chunks)

        # Update registry entries
        for doc_id, info in docs_to_index.items():
            registry[doc_id] = {
                "filename": info["filename"],
            }
    else:
        logger.info("No documents to index or repair")

    # 7. Persist registry
    save_registry(registry)

    # 8. NEW: Always rebuild BM25 to ensure hybrid search is fresh
    _rebuild_bm25_index(store)

    logger.info("Indexing pipeline completed successfully")