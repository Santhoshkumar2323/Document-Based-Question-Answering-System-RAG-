from pathlib import Path
from typing import List, Tuple, Dict

import fitz  # PyMuPDF

from shared.logger import setup_logger

logger = setup_logger(__name__)


def _read_pdf(path: Path) -> List[Tuple[str, int]]:
    """
    Reads a PDF and returns a list of (text, page_number).
    """
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append((text, i + 1))
    return pages


def _read_txt(path: Path) -> List[Tuple[str, int]]:
    """
    Reads a TXT file and returns a single-page list.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text and text.strip():
        return [(text, None)]
    return []


def _normalize(text: str) -> str:
    """
    Light text normalization.
    Intentionally conservative.
    """
    return " ".join(text.split())


def ingest_new_documents(new_docs: Dict[str, dict]) -> List[dict]:
    """
    Ingests ONLY new documents.

    Args:
        new_docs: Mapping of doc_id -> { filename, path }

    Returns:
        List of dicts with keys:
        - doc_id
        - text
        - source
        - page
    """
    records: List[dict] = []

    for doc_id, info in new_docs.items():
        path = Path(info["path"])
        logger.info(f"Ingesting new document: {info['filename']}")

        try:
            if path.suffix.lower() == ".pdf":
                pages = _read_pdf(path)
            elif path.suffix.lower() == ".txt":
                pages = _read_txt(path)
            else:
                logger.warning(f"Unsupported file type skipped: {path.name}")
                continue

            for text, page in pages:
                records.append({
                    "doc_id": doc_id,
                    "text": _normalize(text),
                    "source": path.name,
                    "page": page,
                })

        except Exception as e:
            logger.exception(f"Failed to ingest {path.name}: {e}")

    logger.info(f"Ingested {len(records)} text units from new documents")
    return records
