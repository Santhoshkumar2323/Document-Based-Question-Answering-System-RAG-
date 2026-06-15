import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from shared.config import LOG_FILE, LOGS_DIR


def setup_logger(name: str = "decision_rag") -> logging.Logger:

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  
        backupCount=3,
        encoding="utf-8",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
