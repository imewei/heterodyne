"""Synthetic input for logs/errors extractor tests."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def do_work(x: int) -> int:
    logger.info("Starting work for x=%d", x)
    if x < 0:
        logger.warning("Negative x: %d", x)
        raise ValueError("x must be non-negative")
    if x > 1000:
        logger.error("x out of range")
        sys.exit(2)
    return x
