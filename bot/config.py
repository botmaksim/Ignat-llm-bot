"""
config.py — Application configuration loaded from environment variables.

All settings are resolved at startup via :func:`load_config`.  The resulting
:class:`Config` object is injected into aiogram handlers through the
Dispatcher DI container.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

load_dotenv(BASE_DIR.parent / ".env")

# Sub-configs

@dataclass
class BotConfig:
    token: str = field(
        default_factory=lambda: os.getenv("BOT_TOKEN", "")
    )
    username: str = field(
        default_factory=lambda: os.getenv("BOT_USERNAME", "")
    )
    typing_speed: float = field(
        default_factory=lambda: float(os.getenv("TYPING_SPEED", "0.05"))
    )
    pause_duration: float = field(
        default_factory=lambda: float(os.getenv("PAUSE_DURATION", "1.5"))
    )
    max_history: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    log_format: Literal["text", "json"] = field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "text").lower()
    )


@dataclass
class ColabCLIConfig:
    session_name: str = field(
        default_factory=lambda: os.getenv("COLAB_SESSION_NAME", "ignat-bot")
    )
    gpu_type: str = field(
        default_factory=lambda: os.getenv("COLAB_GPU_TYPE", "T4")
    )
    colab_bin: str = field(
        default_factory=lambda: os.getenv("COLAB_BIN", "colab")
    )
    exec_timeout: float = field(
        default_factory=lambda: float(os.getenv("COLAB_EXEC_TIMEOUT", "120"))
    )

# Root config

@dataclass
class Config:
    bot:        BotConfig     = field(default_factory=BotConfig)
    colab:      ColabCLIConfig = field(default_factory=ColabCLIConfig)
    stickers:   dict[str, dict] = field(default_factory=dict)   # tag → meta
    formatting: dict[str, dict] = field(default_factory=dict)   # tag → {open, close}

# Sticker persistence

def load_stickers(
    file_path: Path = ASSETS_DIR / "stickers.json",
) -> dict[str, dict]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    return {
        item["tag"]: {
            "file_id":     item.get("file_id", ""),
            "description": item.get("description", ""),
        }
        for item in data
        if "tag" in item
    }


def save_stickers(
    stickers: dict[str, dict],
    file_path: Path = ASSETS_DIR / "stickers.json",
) -> None:
    data = [
        {"tag": tag, "file_id": v["file_id"], "description": v["description"]}
        for tag, v in stickers.items()
    ]
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

# Formatting map persistence

def load_formatting(
    file_path: Path = ASSETS_DIR / "formatting.json",
) -> dict[str, dict]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

# Prompt helpers

def build_sticker_prompt(stickers: dict[str, dict]) -> str:
    parts = [
        f"{tag} ({v['description']})"
        for tag, v in stickers.items()
        if v.get("description")
    ]
    return ", ".join(parts)

# Factory

def load_config() -> Config:
    config = Config()
    config.stickers   = load_stickers()
    config.formatting = load_formatting()

    if not config.bot.token:
        raise ValueError(
            "BOT_TOKEN is not set. "
            "Create a .env file from .env.example and set BOT_TOKEN."
        )

    return config
