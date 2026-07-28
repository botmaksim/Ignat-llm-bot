"""
test_config.py — Unit tests for the configuration module.

This module validates environment variable parsing, default values,
and JSON file operations (stickers/formatting).
Block comments denote the specific components or functions being tested.
"""

from __future__ import annotations

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from bot.config import (
    load_config,
    load_stickers,
    save_stickers,
    load_formatting,
    build_sticker_prompt,
    BotConfig,
    ColabCLIConfig,
    Config,
)

# Config Loading Tests

def test_load_config_success(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123:test")
    monkeypatch.setenv("BOT_USERNAME", "testbot")
    monkeypatch.setenv("TYPING_SPEED", "0.1")
    monkeypatch.setenv("PAUSE_DURATION", "2.0")
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "30")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("COLAB_SESSION_NAME", "test-session")
    monkeypatch.setenv("COLAB_GPU_TYPE", "T4")
    monkeypatch.setenv("COLAB_BIN", "/usr/bin/colab")
    monkeypatch.setenv("COLAB_EXEC_TIMEOUT", "60")

    with patch("bot.config.load_stickers", return_value={}), \
         patch("bot.config.load_formatting", return_value={}):
        
        cfg = load_config()

        assert cfg.bot.token == "123:test"
        assert cfg.bot.username == "testbot"
        assert cfg.bot.typing_speed == 0.1
        assert cfg.bot.pause_duration == 2.0
        assert cfg.bot.max_history == 30
        assert cfg.bot.log_level == "DEBUG"
        assert cfg.bot.log_format == "json"

        assert cfg.colab.session_name == "test-session"
        assert cfg.colab.gpu_type == "T4"
        assert cfg.colab.colab_bin == "/usr/bin/colab"
        assert cfg.colab.exec_timeout == 60.0


def test_load_config_missing_token(monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.setenv("TYPING_SPEED", "0.1")
    monkeypatch.setenv("PAUSE_DURATION", "2.0")
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "30")
    monkeypatch.setenv("COLAB_EXEC_TIMEOUT", "60")
    with patch("bot.config.load_stickers", return_value={}), \
         patch("bot.config.load_formatting", return_value={}):
        with pytest.raises(ValueError, match="BOT_TOKEN is not set"):
            load_config()

# Sticker Operations Tests

def test_load_stickers_success(tmp_path):
    data = [
        {"tag": "happy", "file_id": "FID1", "description": "smile"},
        {"tag": "sad", "file_id": "FID2", "description": "cry"}
    ]
    p = tmp_path / "stickers.json"
    p.write_text(json.dumps(data))

    stickers = load_stickers(file_path=p)
    assert len(stickers) == 2
    assert "happy" in stickers
    assert stickers["happy"]["file_id"] == "FID1"
    assert stickers["happy"]["description"] == "smile"


def test_load_stickers_file_missing(tmp_path):
    p = tmp_path / "missing.json"
    assert load_stickers(file_path=p) == {}


def test_load_stickers_invalid_json(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text("not json")
    assert load_stickers(file_path=p) == {}


def test_save_stickers(tmp_path):
    p = tmp_path / "stickers.json"
    stickers = {
        "laugh": {"file_id": "FID3", "description": "laughing out loud"}
    }
    save_stickers(stickers, file_path=p)
    
    assert p.exists()
    data = json.loads(p.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["tag"] == "laugh"
    assert data[0]["file_id"] == "FID3"

# Formatting Operations Tests

def test_load_formatting_success(tmp_path):
    data = {"b": {"open": "<b>", "close": "</b>"}}
    p = tmp_path / "format.json"
    p.write_text(json.dumps(data))

    fmt = load_formatting(file_path=p)
    assert "b" in fmt
    assert fmt["b"]["open"] == "<b>"


def test_load_formatting_missing_or_invalid(tmp_path):
    p = tmp_path / "missing.json"
    assert load_formatting(file_path=p) == {}

    p = tmp_path / "invalid.json"
    p.write_text("invalid json")
    assert load_formatting(file_path=p) == {}

# Prompt Builder Tests

def test_build_sticker_prompt():
    stickers = {
        "happy": {"file_id": "1", "description": "smile"},
        "sad": {"file_id": "2", "description": "cry"},
        "empty": {"file_id": "3", "description": ""},
    }
    prompt = build_sticker_prompt(stickers)
    assert "happy (smile)" in prompt
    assert "sad (cry)" in prompt
    assert "empty" not in prompt
