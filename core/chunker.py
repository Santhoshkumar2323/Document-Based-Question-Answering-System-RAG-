from typing import List

from shared.models import Chunk
from shared.config import CHUNK_SIZE, CHUNK_OVERLAP
from shared.logger import setup_logger

logger = setup_logger(__name__)


def chunk_documents(records: List[dict]) -> List[Chunk]:
    """
    Splits ingested document records into fixed-size overlapping chunks.

    Each chunk ID is deterministic:
    doc_id:page:chunk_index
    """
    chunks: List[Chunk] = []

    for record in records:
        text = record["text"]
        source = record["source"]
        page = record["page"]
        doc_id = record["doc_id"]

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            # Avoid splitting words mid-way
            if end < text_length:
                last_space = chunk_text.rfind(" ")
                if last_space != -1:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space

            if chunk_text.strip():
                chunk_id = f"{doc_id}:{page}:{chunk_index}"

                chunk = Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    source=source,
                    page=page,
                    metadata={
                        "chunk_index": str(chunk_index),
                        "char_start": str(start),
                        "char_end": str(end),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            start = max(end - CHUNK_OVERLAP, 0)

    logger.info(f"Created {len(chunks)} chunks from {len(records)} records")
    return chunks
