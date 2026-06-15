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
    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

        self.bm25_data = None
        if BM25_INDEX_FILE.exists():
            logger.info("Loading BM25 index...")
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25_data = pickle.load(f)
        else:
            logger.warning("BM25 index not found. Hybrid search will fall back to Vector only.")


        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve(self, query: str) -> List[RetrievedEvidence]:
        logger.info(f"Hybrid Retrieval for: {query}")
        
        candidates: Dict[str, Chunk] = {}


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
                        page=meta.get("page", 0),
                        metadata=meta,
                    )
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")

        if self.bm25_data:
            try:
                bm25 = self.bm25_data["model"]
                chunk_map = self.bm25_data["map"]
                
                tokenized_query = query.lower().split()
                scores = bm25.get_scores(tokenized_query)
                top_n_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]

                for idx in top_n_indices:
                    if scores[idx] <= 0:
                        continue
                        
                    mapped = chunk_map[idx]
                    chunk_id = mapped["id"]
                    
                    if chunk_id not in candidates:
                        candidates[chunk_id] = Chunk(
                            id=chunk_id,
                            doc_id=mapped["metadata"]["doc_id"],
                            text=mapped["text"],
                            source=mapped["metadata"]["source"],
                            page=mapped["metadata"].get("page", 0),
                            metadata=mapped["metadata"],
                        )
            except Exception as e:
                logger.error(f"BM25 retrieval failed: {e}")

        unique_chunks = list(candidates.values())
        if not unique_chunks:
            logger.warning("No candidates found via Vector or BM25.")
            return []

        logger.info(f"Fusion: {len(unique_chunks)} unique candidates identified.")
        pairs = [[query, c.text] for c in unique_chunks]

        try:
            raw_scores = self.reranker.predict(pairs)
            scores = 1 / (1 + np.exp(-raw_scores))

            reranked = []
            for chunk, score in zip(unique_chunks, scores):
                reranked.append(
                    RetrievedEvidence(
                        chunk=chunk,
                        score=float(1.0 - score)
                    )
                )
            reranked.sort(key=lambda x: x.score)
            final_results = reranked[:TOP_K_RERANKED]

            logger.info(f"Reranking complete. Top score (distance): {final_results[0].score:.4f}")
            return final_results

        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            return [RetrievedEvidence(chunk=c, score=0.5) for c in unique_chunks[:TOP_K_RERANKED]]