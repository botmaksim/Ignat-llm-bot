"""
media.py — Media payload abstraction and download/conversion pipeline.

After this module runs, consumers (``colab_api.py``) receive payloads in
exactly **four canonical formats** that map directly to what the Colab
inference models accept:

  ============  =========================================================
  MediaKind     Telegram source
  ============  =========================================================
  ``JPEG``      ``message.photo`` — Telegram already delivers as JPEG;
                Pillow re-encode is skipped via magic-byte fast-path.
  ``GIF``       ``message.animation`` (MP4) or video sticker (WebM) —
                converted to GIF by :func:`~converter.to_gif` via ffmpeg.
  ``OGG``       ``message.voice`` — always OGG Opus from Telegram; no
                conversion needed.
  ``MP3``       ``message.audio`` — arbitrary codec; passed as-is to the
                STT model which handles decoding on the Colab side.
  ============  =========================================================

Unsupported types (animated/static stickers, video, document, contact, …)
return ``None`` from :func:`build_media_payload` so that callers can handle
them uniformly without ``isinstance`` sprawl.

Conversion errors (``ConversionError``, ``MediaSizeLimitError``) are raised
rather than silenced so that ``handlers.py`` can send the user an
informative message.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Optional

from aiogram import Bot
from aiogram.types import Message

from bot.converter import ConversionError, to_gif, to_jpeg

logger = logging.getLogger(__name__)

# Size limits

_BYTES_PER_MB: Final = 1024 * 1024

_SIZE_LIMIT_MB: Final[dict[str, int]] = {
    "image": 8,
    "audio": 20,
    "video": 16,
}

# MediaKind

class MediaKind(Enum):
    JPEG = auto()   # Photo — always JPEG after to_jpeg()
    GIF  = auto()   # Animation / video sticker — always GIF after to_gif()
    OGG  = auto()   # Voice message — Telegram always delivers OGG Opus
    MP3  = auto()   # Audio file — arbitrary codec, Colab handles decoding

    # Derived properties

    @property
    def mime_type(self) -> str:
        return _KIND_META[self]["mime"]

    @property
    def extension(self) -> str:
        return _KIND_META[self]["ext"]

    @property
    def size_limit_bytes(self) -> int:
        category = _KIND_META[self]["category"]
        return _SIZE_LIMIT_MB.get(category, 8) * _BYTES_PER_MB


_KIND_META: dict[MediaKind, dict] = {
    MediaKind.JPEG: {"mime": "image/jpeg", "ext": "jpg",  "category": "image"},
    MediaKind.GIF:  {"mime": "image/gif",  "ext": "gif",  "category": "video"},
    MediaKind.OGG:  {"mime": "audio/ogg",  "ext": "ogg",  "category": "audio"},
    MediaKind.MP3:  {"mime": "audio/mpeg", "ext": "mp3",  "category": "audio"},
}

# MediaPayload

class MediaSizeLimitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MediaPayload:
    data:    bytes
    kind:    MediaKind
    file_id: str

    @property
    def mime_type(self) -> str:
        return self.kind.mime_type

    @property
    def filename(self) -> str:
        return f"media.{self.kind.extension}"

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def is_within_size_limit(self) -> bool:
        return self.size_bytes <= self.kind.size_limit_bytes

    def __repr__(self) -> str:
        return (
            f"MediaPayload(kind={self.kind.name}, "
            f"size={self.size_bytes / 1024:.1f} KB, "
            f"file_id={self.file_id[:12]}…)"
        )

# Download helper

async def _download(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    return buf.getvalue()

# Public builder

async def build_media_payload(
    message: Message,
    bot: Bot,
) -> Optional[MediaPayload]:
    """
    Download and convert the media attachment of *message*, if any.

    Conversion is performed immediately after download:
    - Photos → :func:`~converter.to_jpeg` (Pillow; JPEG fast-path if already JPEG).
    - Animations & video stickers → :func:`~converter.to_gif` (ffmpeg).
    - Voice & audio → passed through without conversion.
    - Animated stickers (TGS/Lottie) and static stickers → ``None`` (not supported).

    Returns:
        A :class:`MediaPayload` with normalised data, or ``None`` if the
        message carries no supported media.

    Raises:
        ConversionError:      If format conversion fails (e.g. ffmpeg missing).
        MediaSizeLimitError:  If the converted payload exceeds the size limit.
    """

    # Detect Telegram media type
    if message.photo:
        file_id = message.photo[-1].file_id
        raw = await _download(bot, file_id)
        logger.debug("Converting photo → JPEG", extra={"file_id": file_id[:12]})
        converted = await to_jpeg(raw)
        return _build_checked(converted, MediaKind.JPEG, file_id)

    if message.voice:
        file_id = message.voice.file_id
        raw = await _download(bot, file_id)
        return _build_checked(raw, MediaKind.OGG, file_id)

    if message.audio:
        file_id = message.audio.file_id
        raw = await _download(bot, file_id)
        return _build_checked(raw, MediaKind.MP3, file_id)

    if message.animation:
        file_id = message.animation.file_id
        raw = await _download(bot, file_id)
        logger.debug(
            "Converting animation (MP4) → GIF", extra={"file_id": file_id[:12]}
        )
        converted = await to_gif(raw, "mp4")
        return _build_checked(converted, MediaKind.GIF, file_id)

    if message.sticker:
        sticker = message.sticker
        if sticker.is_video:
            file_id = sticker.file_id
            raw = await _download(bot, file_id)
            logger.debug(
                "Converting video sticker (WebM) → GIF",
                extra={"file_id": file_id[:12]},
            )
            converted = await to_gif(raw, "webm")
            return _build_checked(converted, MediaKind.GIF, file_id)

        logger.debug(
            "Sticker type not supported for model input",
            extra={
                "is_animated": sticker.is_animated,
                "is_video":    sticker.is_video,
            },
        )
        return None

    return None

# Internal helper

def _build_checked(data: bytes, kind: MediaKind, file_id: str) -> MediaPayload:
    payload = MediaPayload(data=data, kind=kind, file_id=file_id)

    if not payload.is_within_size_limit():
        limit_mb = kind.size_limit_bytes // _BYTES_PER_MB
        raise MediaSizeLimitError(
            f"{kind.name} payload is {payload.size_bytes / _BYTES_PER_MB:.1f} MiB, "
            f"exceeding the {limit_mb} MiB limit for this media type."
        )

    logger.debug(
        "Media payload ready",
        extra={
            "kind":       kind.name,
            "size_bytes": payload.size_bytes,
            "file_id":    file_id[:12],
        },
    )
    return payload
