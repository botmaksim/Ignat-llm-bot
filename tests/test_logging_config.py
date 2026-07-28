"""
test_logging_config.py — Unit tests for the logging configuration.

This module validates the JSON and colorized text formatters, ensuring
that context variables (like request_id) are correctly injected into log records.
Block comments denote the specific formatters or configuration functions being tested.
"""

from __future__ import annotations

import json
import logging
import pytest
from unittest.mock import patch

from bot.logging_config import (
    JsonFormatter,
    ColourFormatter,
    configure_logging,
    request_id,
)

# JSON Formatter Tests

def test_json_formatter():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )
    record.extra_field = "extra_value"

    request_id.set("test-req-123")
    output = formatter.format(record)
    
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Hello world"
    assert parsed["request_id"] == "test-req-123"
    assert parsed["extra_field"] == "extra_value"
    assert "ts" in parsed


def test_json_formatter_with_exception():
    formatter = JsonFormatter()
    try:
        1 / 0
    except ZeroDivisionError as exc:
        exc_info = (type(exc), exc, exc.__traceback__)

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="An error occurred",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exc_info" in parsed
    assert "ZeroDivisionError" in parsed["exc_info"]

# Colour Formatter Tests

def test_colour_formatter():
    formatter = ColourFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="Warning message",
        args=(),
        exc_info=None,
    )

    request_id.set("test-req-456")
    output = formatter.format(record)
    
    assert "test-req-456" in output
    assert "Warning message" in output
    assert "\033[" in output  # ANSI escape code

# Configure Logging Tests

def test_configure_logging_text():
    with patch("bot.logging_config.logging.config.dictConfig") as mock_dictConfig:
        configure_logging(level="DEBUG", fmt="text")
        mock_dictConfig.assert_called_once()
        config_dict = mock_dictConfig.call_args[0][0]
        assert config_dict["root"]["level"] == "DEBUG"
        assert config_dict["formatters"]["main"]["()"] == "logging_config.ColourFormatter"


def test_configure_logging_json():
    with patch("bot.logging_config.logging.config.dictConfig") as mock_dictConfig:
        configure_logging(level="ERROR", fmt="json")
        mock_dictConfig.assert_called_once()
        config_dict = mock_dictConfig.call_args[0][0]
        assert config_dict["root"]["level"] == "ERROR"
        assert config_dict["formatters"]["main"]["()"] == "logging_config.JsonFormatter"
