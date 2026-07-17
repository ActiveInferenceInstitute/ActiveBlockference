"""Centralised stdlib :mod:`logging` configuration for ActiveBlockference.

Two entry points:

* :func:`get_logger` — module-level ``log = get_logger(__name__)`` idiom.
* :func:`configure_run_logging` — called by the pipeline to attach a
  per-run ``run.log`` file handler under ``output/<run_name>/``.

Format choices — short, structured-ish, parseable by ``grep`` *and* by
``jq -R 'split(" | ")'`` if a downstream consumer wants more than a tail.
"""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_BASE_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; idempotent."""
    _ensure_base_config()
    return logging.getLogger(name)


def _ensure_base_config() -> None:
    global _BASE_CONFIGURED
    if _BASE_CONFIGURED:
        return
    root = logging.getLogger("blockference")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _BASE_CONFIGURED = True


def configure_run_logging(
    log_path: Path,
    *,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> logging.FileHandler:
    """Attach a per-run file handler at ``log_path`` and return it.

    Caller is responsible for removing the handler when the run ends:
    ``logger.removeHandler(handler); handler.close()``.
    """
    _ensure_base_config()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    handler.setLevel(level)
    logger = logging.getLogger("blockference")
    logger.addHandler(handler)
    if level < logger.level:
        logger.setLevel(level)
    if capture_warnings:
        logging.captureWarnings(True)
    return handler


def remove_handler(handler: logging.Handler | None) -> None:
    """Detach and close a handler when one is supplied."""
    if handler is None:
        return
    logging.getLogger("blockference").removeHandler(handler)
    handler.close()


__all__ = [
    "configure_run_logging",
    "get_logger",
    "remove_handler",
]
