import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from shared.config import LOG_FILE, LOGS_DIR


def setup_logger(name: str = "decision_rag") -> logging.Logger:
    """
    Sets up and returns a configured logger.
    Logs to both console and file.
    """

    # Ensure logs directory exists
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers (important in repeated runs)
    if logger.handlers:
        return logger

    # ----------------------------
    # File handler (rotating)
    # ----------------------------
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # ----------------------------
    # Console handler
    # ----------------------------
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # ----------------------------
    # Attach handlers
    # ----------------------------
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
