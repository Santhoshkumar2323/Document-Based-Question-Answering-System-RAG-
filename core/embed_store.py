import os
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from shared.models import Chunk
from shared.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
)
from shared.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """
    Production-grade vector store abstraction.
    Owns embedding lifecycle and document-level integrity.
    """

    def __init__(self):
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info(f"Initializing ChromaDB Persistent Client at {CHROMA_DIR}")

        # IMPORTANT: CHROMA_DIR is a Path → cast to str
        os.makedirs(str(CHROMA_DIR), exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    # --------------------------------------------------
    # WRITE PATH
    # --------------------------------------------------

# --------------------------------------------------
    # WRITE PATH
    # --------------------------------------------------

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Embeds and upserts chunks into ChromaDB.
        Safe for massive datasets (automatically batches writes).
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        logger.info(f"Indexing {len(chunks)} chunks")

        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            # HARD GUARANTEE: lifecycle depends on doc_id
            assert chunk.doc_id, "Chunk missing doc_id — lifecycle invariant violated"

            ids.append(chunk.id)
            texts.append(chunk.text)
            metadatas.append({
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "page": chunk.page,
                **chunk.metadata,
            })

        # 1. Generate Embeddings (This can be done in one go, or batched if RAM is low)
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # 2. Upsert in Batches (The Fix for "Batch size > 5461" error)
        BATCH_SIZE = 5000
        total_chunks = len(chunks)
        
        logger.info(f"Upserting to ChromaDB in batches of {BATCH_SIZE}...")
        
        for i in range(0, total_chunks, BATCH_SIZE):
            end = min(i + BATCH_SIZE, total_chunks)
            
            # Slice the data for this batch
            batch_ids = ids[i:end]
            batch_texts = texts[i:end]
            batch_embeddings = embeddings[i:end].tolist()
            batch_metadatas = metadatas[i:end]
            
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            logger.info(f"  - Batch {i} to {end} committed")

        logger.info("Chunk indexing completed")
    # --------------------------------------------------
    # DELETE PATH
    # --------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """
        Deletes all vectors belonging to a document.
        """
        logger.info(f"Deleting vectors for document: {doc_id}")
        self.collection.delete(
            where={"doc_id": doc_id}
        )

    # --------------------------------------------------
    # VERIFY PATH (CRITICAL FOR CONSISTENCY)
    # --------------------------------------------------

    def document_exists(self, doc_id: str) -> bool:
        """
        Checks whether at least one vector exists for a document.
        Used to repair registry ↔ vector-store desync.
        """
        results = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
        )

        # Defensive: protect against schema drift / API changes
        if "ids" not in results:
            logger.warning(
                f"Chroma returned unexpected schema while checking doc_id={doc_id}"
            )
            return False

        exists = len(results["ids"]) > 0

        logger.debug(
            f"Document existence check for {doc_id}: {'FOUND' if exists else 'MISSING'}"
        )

        return exists
