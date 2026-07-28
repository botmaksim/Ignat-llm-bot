"""
tests/conftest.py — Shared pytest fixtures.

Fixtures use mock objects exclusively — no network calls, no subprocess calls,
no Telegram API. Tests that need Colab responses mock _exec_script / _upload_payload
directly, which simulates what Colab would return without invoking the CLI.
"""

from __future__ import annotations

import asyncio
import io
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from bot.colab_api import ColabAPIClient, ColabSession
from bot.media import MediaKind, MediaPayload

# Event loop

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Image helpers

def make_jpeg_bytes(width: int = 10, height: int = 10) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def make_png_bytes(width: int = 10, height: int = 10, mode: str = "RGB") -> bytes:
    color = (100, 150, 200) if mode == "RGB" else (100, 150, 200, 128)
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_gif_bytes(width: int = 10, height: int = 10) -> bytes:
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()

# MediaPayload fixtures

def make_payload(kind: MediaKind, data: bytes) -> MediaPayload:
    return MediaPayload(data=data, kind=kind, file_id="fake_file_id_0123456789ab")


@pytest.fixture
def jpeg_payload() -> MediaPayload:
    return make_payload(MediaKind.JPEG, make_jpeg_bytes())


@pytest.fixture
def ogg_payload() -> MediaPayload:
    return make_payload(MediaKind.OGG, b"OggS" + b"\x00" * 100)


@pytest.fixture
def mp3_payload() -> MediaPayload:
    return make_payload(MediaKind.MP3, b"\xff\xfb" + b"\x00" * 100)


@pytest.fixture
def gif_payload() -> MediaPayload:
    return make_payload(MediaKind.GIF, make_gif_bytes())

# ColabSession / ColabAPIClient fixtures

@pytest.fixture
def session() -> ColabSession:
    return ColabSession(
        session_name="test-session",
        gpu_type="T4",
        colab_bin="colab",
    )


@pytest.fixture
def live_client(session: ColabSession) -> ColabAPIClient:
    client = ColabAPIClient(session=session, exec_timeout=10.0)
    session._ready = True
    return client

# Aiogram message mock helpers

def _make_user(first_name: str = "Alice", username: str = "alice") -> MagicMock:
    user = MagicMock()
    user.first_name = first_name
    user.username = username
    return user


def make_message(
    text: Optional[str] = "hello",
    chat_type: str = "private",
    has_photo: bool = False,
    has_voice: bool = False,
    has_audio: bool = False,
    has_animation: bool = False,
    has_sticker: bool = False,
    sticker_is_animated: bool = False,
    sticker_is_video: bool = False,
    reply_to: Optional[MagicMock] = None,
) -> MagicMock:
    msg = MagicMock()
    msg.text = text
    msg.caption = None
    msg.from_user = _make_user()
    msg.chat = MagicMock()
    msg.chat.type = chat_type
    msg.chat.id = 12345

    for attr in (
        "photo", "voice", "audio", "animation", "sticker",
        "video", "document", "video_note", "contact", "location", "poll",
    ):
        setattr(msg, attr, None)

    msg.reply_to_message = reply_to

    if has_photo:
        photo = MagicMock()
        photo.file_id = "photo_file_id"
        msg.photo = [photo]

    if has_voice:
        voice = MagicMock()
        voice.file_id = "voice_file_id"
        msg.voice = voice

    if has_audio:
        audio = MagicMock()
        audio.file_id = "audio_file_id"
        msg.audio = audio

    if has_animation:
        anim = MagicMock()
        anim.file_id = "anim_file_id"
        msg.animation = anim

    if has_sticker:
        sticker = MagicMock()
        sticker.file_id = "sticker_file_id"
        sticker.file_unique_id = "unique_sticker_0"
        sticker.emoji = "😊"
        sticker.is_animated = sticker_is_animated
        sticker.is_video = sticker_is_video
        msg.sticker = sticker

    msg.answer = AsyncMock()
    msg.answer_sticker = AsyncMock()
    return msg
