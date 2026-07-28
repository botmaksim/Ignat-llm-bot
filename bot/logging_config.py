"""
logging_config.py — Centralised logging with request-ID tracing.

Two formatters are provided:

``text`` (development)
  Coloured human-readable lines with ANSI escapes::

      12:34:56 INFO     colab_api  [req=a1b2c3d4] Session ready

``json`` (production)
  Structured JSON per record, one line, suitable for Loki / CloudWatch /
  Datadog log aggregators::

      {"ts": "2025-01-15T12:34:56", "level": "INFO", "logger": "colab_api",
       "message": "Session ready", "request_id": "a1b2c3d4", "session": "ignat"}

Request-ID tracing
------------------
:data:`request_id` is a :class:`contextvars.ContextVar` that is set once per
incoming Telegram message in ``handlers.py``.  Both formatters include its
value automatically so that every log line produced during a single request
carries the same identifier — enabling end-to-end tracing across modules
without passing the ID through every function signature.

Usage::

    from logging_config import configure_logging, request_id
    configure_logging(level="INFO", fmt="json")

    # In a request handler:
    request_id.set(uuid.uuid4().hex[:12])

After ``configure_logging`` is called once (in ``main.py``), every module
that does ``logging.getLogger(__name__)`` inherits the configured handlers.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
from typing import Any, Literal

# Request-ID context variable

request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

# JSON formatter

class JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.

    Standard fields: ``ts``, ``level``, ``logger``, ``message``, ``request_id``.
    Any ``extra`` dict passed to the logger call is merged into the root object.

    Example::

        {"ts": "2025-01-15T12:34:56", "level": "INFO", "logger": "handlers",
         "message": "New message", "request_id": "a1b2c3d4", "chat_id": 123}
    """

    _RESERVED: frozenset[str] = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "ts":         self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.message,
            "request_id": request_id.get(),
        }

        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)

# Colour (development) formatter

_ANSI_RESET = "\033[0m"
_LEVEL_COLOUR: dict[str, str] = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_DIM = "\033[2m"


class ColourFormatter(logging.Formatter):
    """
    Human-readable formatter with ANSI colours and request-ID annotation.

    Format::

        HH:MM:SS LEVEL    logger  [req=<id>] message
    """

    _DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _LEVEL_COLOUR.get(record.levelname, "")
        rid = request_id.get()
        rid_tag = f" {_DIM}[req={rid}]{_ANSI_RESET}" if rid != "-" else ""

        record.levelname = f"{colour}{record.levelname:<8}{_ANSI_RESET}"
        record.name = f"{_DIM}{record.name}{_ANSI_RESET}"

        base = super().format(record)
        return base.replace(record.getMessage(), f"{rid_tag} {record.getMessage()}", 1)

# Public API

def configure_logging(
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
) -> None:
    """
    Configure the root logger.  Call exactly once at application startup.

    Args:
        level: Logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …).
        fmt:   ``"text"`` for coloured dev output, ``"json"`` for production.
    """
    formatter_cls = (
        "logging_config.JsonFormatter"
        if fmt == "json"
        else "logging_config.ColourFormatter"
    )

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "main": {
                "()":      formatter_cls,
                "fmt":     "%(asctime)s %(levelname)s %(name)s  %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "stdout": {
                "class":     "logging.StreamHandler",
                "stream":    "ext://sys.stdout",
                "formatter": "main",
            }
        },
        "root": {
            "handlers": ["stdout"],
            "level":    level.upper(),
        },
        # Suppress noisy third-party loggers.
        "loggers": {
            "aiogram":  {"level": "WARNING", "propagate": True},
            "aiohttp":  {"level": "WARNING", "propagate": True},
            "asyncio":  {"level": "WARNING", "propagate": True},
            "PIL":      {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)
    logging.getLogger(__name__).debug(
        "Logging configured",
        extra={"level": level, "format": fmt},
    )
