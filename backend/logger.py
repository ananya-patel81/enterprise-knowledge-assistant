"""
logger.py
=========

Centralized logging configuration.

WHY structured logging instead of print()?
--------------------------------------------
- Logging lets us control verbosity (DEBUG vs INFO vs ERROR) without
  editing code.
- Every log line includes a timestamp, module name, and level, which is
  essential for debugging retrieval latency or errors in production.
- This is a small but real signal of engineering maturity that
  interviewers/recruiters notice when reading code.
"""

import logging
import sys
from backend.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` instance with a consistent format.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)

    return logger
