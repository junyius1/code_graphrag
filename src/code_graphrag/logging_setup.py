"""Central logging configuration for code_graphrag."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("litellm", "httpx", "typer", "lancedb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
