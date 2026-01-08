import pickle
import numpy as np
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from shared.models import RetrievedEvidence, Chunk
from shared.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    BM25_INDEX_FILE,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED,
)
from shared.logger import setup_logger

logger = setup_logger(__name__)


class Retriever:
    """
    Hybrid Retriever (Vector + BM25) with Cross-Encoder Reranking.
    Pipeline:
    1. Recall: Fetch TOP_K_RETRIEVAL from Vector Store.
    2. Recall: Fetch TOP_K_RETRIEVAL from BM25 Index.
    3. Fusion: Deduplicate candidates.
    4. Rerank: Score (Query, Text) pairs using Cross-Encoder.
    5. Output: Return TOP_K_RERANKED sorted by quality.
    """

    def __init__(self):
        # 1. Embedding Model (for Vector Search)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 2. Vector Store
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

        # 3. BM25 Index (for Keyword Search)
        self.bm25_data = None
        if BM25_INDEX_FILE.exists():
            logger.info("Loading BM25 index...")
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25_data = pickle.load(f)
        else:
            logger.warning("BM25 index not found. Hybrid search will fall back to Vector only.")

        # 4. Reranker Model
        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve(self, query: str) -> List[RetrievedEvidence]:
        logger.info(f"Hybrid Retrieval for: {query}")
        
        candidates: Dict[str, Chunk] = {}

        # -------------------------------------------
        # A. Vector Retrieval (Semantic)
        # -------------------------------------------
        try:
            query_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True
            ).tolist()

            v_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K_RETRIEVAL,
                include=["documents", "metadatas"],
            )

            if v_results["documents"]:
                for text, meta in zip(v_results["documents"][0], v_results["metadatas"][0]):
                    chunk_id = f"{meta['doc_id']}:{meta['chunk_index']}"
                    candidates[chunk_id] = Chunk(
                        id=chunk_id,
                        doc_id=meta["doc_id"],
                        text=text,
                        source=meta["source"],
                        page=meta["page"],
                        metadata=meta,
                    )
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")

        # -------------------------------------------
        # B. BM25 Retrieval (Keyword)
        # -------------------------------------------
        if self.bm25_data:
            try:
                bm25 = self.bm25_data["model"]
                chunk_map = self.bm25_data["map"]
                
                tokenized_query = query.lower().split()
                # Get scores for all docs
                scores = bm25.get_scores(tokenized_query)
                # Get top N indices
                top_n_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]

                for idx in top_n_indices:
                    # Ignore zero-score matches (irrelevant)
                    if scores[idx] <= 0:
                        continue
                        
                    mapped = chunk_map[idx]
                    chunk_id = mapped["id"]
                    
                    # If not already in candidates, add it
                    if chunk_id not in candidates:
                        candidates[chunk_id] = Chunk(
                            id=chunk_id,
                            doc_id=mapped["metadata"]["doc_id"],
                            text=mapped["text"],
                            source=mapped["metadata"]["source"],
                            page=mapped["metadata"]["page"],
                            metadata=mapped["metadata"],
                        )
            except Exception as e:
                logger.error(f"BM25 retrieval failed: {e}")

        unique_chunks = list(candidates.values())
        if not unique_chunks:
            logger.warning("No candidates found via Vector or BM25.")
            return []

        logger.info(f"Fusion: {len(unique_chunks)} unique candidates identified.")

        # -------------------------------------------
        # C. Reranking (Cross-Encoder)
        # -------------------------------------------
        # Prepare pairs: (Query, Document Text)
        pairs = [[query, c.text] for c in unique_chunks]

        try:
            # Predict gives us scores (unbounded or 0-1 depending on model).
            # MS-MARCO models usually output raw logits.
            # We apply sigmoid to get 0-1 probability.
            raw_scores = self.reranker.predict(pairs)
            
            # Sigmoid normalization: 1 / (1 + exp(-x))
            scores = 1 / (1 + np.exp(-raw_scores))

            reranked = []
            for chunk, score in zip(unique_chunks, scores):
                reranked.append(
                    RetrievedEvidence(
                        chunk=chunk,
                        # IMPORTANT: Invert score for "Distance" logic.
                        # High Similarity (0.9) -> Low Distance (0.1)
                        score=float(1.0 - score)
                    )
                )

            # Sort by Distance (Ascending) -> Closest first
            reranked.sort(key=lambda x: x.score)
            
            # Slice top K
            final_results = reranked[:TOP_K_RERANKED]

            logger.info(f"Reranking complete. Top score (distance): {final_results[0].score:.4f}")
            return final_results

        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            # Fallback: Just return vector results (simulated by raw list)
            # This is a fail-safe.
            return [RetrievedEvidence(chunk=c, score=0.5) for c in unique_chunks[:TOP_K_RERANKED]]