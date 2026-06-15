import json
import hashlib
from pathlib import Path
from typing import Dict
from shared.config import REGISTRY_FILE
from shared.logger import setup_logger

logger = setup_logger(__name__)


def compute_doc_id(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_registry() -> Dict[str, dict]:
    if not REGISTRY_FILE.exists():
        logger.info("Registry file not found, initializing new registry")
        return {}

    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: Dict[str, dict]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated")


def scan_documents(docs_dir: Path) -> Dict[str, dict]:
    scanned: Dict[str, dict] = {}

    for path in docs_dir.iterdir():
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        doc_id = compute_doc_id(path)
        scanned[doc_id] = {
            "filename": path.name,
            "path": str(path),
        }

    return scanned


def diff_registry(
    scanned: Dict[str, dict],
    registry: Dict[str, dict],
) -> Dict[str, Dict[str, dict]]:
    scanned_ids = set(scanned.keys())
    registry_ids = set(registry.keys())

    new_docs = scanned_ids - registry_ids
    existing_docs = scanned_ids & registry_ids
    deleted_docs = registry_ids - scanned_ids

    return {
        "new": {doc_id: scanned[doc_id] for doc_id in new_docs},
        "existing": {doc_id: scanned[doc_id] for doc_id in existing_docs},
        "deleted": {doc_id: registry[doc_id] for doc_id in deleted_docs},
    }
