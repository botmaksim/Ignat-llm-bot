"""
tests/test_media.py — Unit tests for media.py

- All Telegram downloads are mocked via _download (no network calls).
- Conversion functions (to_jpeg, to_gif) are mocked so converter.py is
  tested separately in test_converter.py.
- This file tests only the routing / detection / size-limit logic in media.py.

Covers:
- MediaKind: 4 canonical kinds, mime_type, extension, size_limit_bytes
- MediaPayload: filename, size_bytes, mime_type, is_within_size_limit
- build_media_payload: photo→JPEG, voice→OGG, audio→MP3,
  animation→GIF (via to_gif), video sticker→GIF (via to_gif),
  animated sticker→None, static sticker→None, text-only→None,
  size limit enforcement, file_id preserved in payload
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.media import (
    MediaKind,
    MediaPayload,
    MediaSizeLimitError,
    _build_checked,
    build_media_payload,
)
from tests.conftest import make_gif_bytes, make_jpeg_bytes, make_message

# MediaKind properties

class TestMediaKind:

    @pytest.mark.parametrize("kind, expected_mime", [
        (MediaKind.JPEG, "image/jpeg"),
        (MediaKind.GIF,  "image/gif"),
        (MediaKind.OGG,  "audio/ogg"),
        (MediaKind.MP3,  "audio/mpeg"),
    ])
    def test_mime_type(self, kind, expected_mime):
        assert kind.mime_type == expected_mime

    @pytest.mark.parametrize("kind, expected_ext", [
        (MediaKind.JPEG, "jpg"),
        (MediaKind.GIF,  "gif"),
        (MediaKind.OGG,  "ogg"),
        (MediaKind.MP3,  "mp3"),
    ])
    def test_extension(self, kind, expected_ext):
        assert kind.extension == expected_ext

    def test_image_size_limit_is_at_least_1mb(self):
        assert MediaKind.JPEG.size_limit_bytes >= 1024 * 1024

    def test_audio_limit_larger_than_image_limit(self):
        assert MediaKind.OGG.size_limit_bytes > MediaKind.JPEG.size_limit_bytes

    def test_only_four_kinds(self):
        """Canonical output kinds must be exactly JPEG, GIF, OGG, MP3."""
        assert set(MediaKind) == {MediaKind.JPEG, MediaKind.GIF, MediaKind.OGG, MediaKind.MP3}

# MediaPayload properties

class TestMediaPayload:

    def test_filename_uses_kind_extension(self, jpeg_payload):
        assert jpeg_payload.filename == "media.jpg"

    def test_gif_filename(self, gif_payload):
        assert gif_payload.filename == "media.gif"

    def test_size_bytes_matches_data_length(self, jpeg_payload):
        assert jpeg_payload.size_bytes == len(jpeg_payload.data)

    def test_mime_type_delegated_to_kind(self, jpeg_payload):
        assert jpeg_payload.mime_type == "image/jpeg"

    def test_within_size_limit_for_small_payload(self, jpeg_payload):
        assert jpeg_payload.is_within_size_limit()

    def test_exceeds_size_limit_for_oversized_payload(self):
        limit = MediaKind.JPEG.size_limit_bytes
        huge = MediaPayload(data=b"\x00" * (limit + 1), kind=MediaKind.JPEG, file_id="x")
        assert not huge.is_within_size_limit()

    def test_file_id_stored_correctly(self, jpeg_payload):
        assert jpeg_payload.file_id == "fake_file_id_0123456789ab"


# _build_checked helper

class TestBuildChecked:

    def test_returns_payload_within_limit(self):
        payload = _build_checked(b"\x00" * 100, MediaKind.OGG, "fid")
        assert payload.kind == MediaKind.OGG
        assert payload.size_bytes == 100

    def test_raises_size_limit_error_when_oversized(self):
        limit = MediaKind.JPEG.size_limit_bytes
        with pytest.raises(MediaSizeLimitError, match="MiB"):
            _build_checked(b"\x00" * (limit + 1), MediaKind.JPEG, "fid")


# build_media_payload — Telegram media type routing

class TestBuildMediaPayload:
    """
    Each test:
      1. Builds a mock Message for a specific media type.
      2. Mocks _download to return fake raw bytes.
      3. Mocks the appropriate converter (to_jpeg / to_gif) to return
         a known output, so we verify routing without real conversion.
      4. Asserts the returned MediaPayload has the correct kind and data.
    """

    def _mock_bot(self) -> MagicMock:
        return MagicMock()

    # Photo

    @pytest.mark.asyncio
    async def test_photo_yields_jpeg(self):
        msg = make_message(has_photo=True)
        msg.photo[-1].file_id = "photo_fid"
        raw = b"\x89PNG" + b"\x00" * 50
        converted_jpeg = make_jpeg_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=raw)):
            with patch("bot.media.to_jpeg", new=AsyncMock(return_value=converted_jpeg)):
                payload = await build_media_payload(msg, self._mock_bot())

        assert payload is not None
        assert payload.kind == MediaKind.JPEG
        assert payload.data == converted_jpeg
        assert payload.file_id == "photo_fid"

    @pytest.mark.asyncio
    async def test_photo_calls_to_jpeg_with_raw_bytes(self):
        msg = make_message(has_photo=True)
        raw = b"\x89PNG" + b"\x00" * 50
        jpeg_calls: list[bytes] = []

        async def capture_jpeg(data, **kw):
            jpeg_calls.append(data)
            return make_jpeg_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=raw)):
            with patch("bot.media.to_jpeg", side_effect=capture_jpeg):
                await build_media_payload(msg, self._mock_bot())

        assert jpeg_calls == [raw]

    # Voice

    @pytest.mark.asyncio
    async def test_voice_yields_ogg(self):
        msg = make_message(has_voice=True)
        msg.voice.file_id = "voice_fid"
        raw = b"OggS" + b"\x00" * 50

        with patch("bot.media._download", new=AsyncMock(return_value=raw)):
            payload = await build_media_payload(msg, self._mock_bot())

        assert payload is not None
        assert payload.kind == MediaKind.OGG
        assert payload.data == raw
        assert payload.file_id == "voice_fid"

    # Audio

    @pytest.mark.asyncio
    async def test_audio_yields_mp3(self):
        msg = make_message(has_audio=True)
        msg.audio.file_id = "audio_fid"
        raw = b"\xff\xfb" + b"\x00" * 50

        with patch("bot.media._download", new=AsyncMock(return_value=raw)):
            payload = await build_media_payload(msg, self._mock_bot())

        assert payload is not None
        assert payload.kind == MediaKind.MP3
        assert payload.data == raw
        assert payload.file_id == "audio_fid"

    # Animation

    @pytest.mark.asyncio
    async def test_animation_yields_gif(self):
        msg = make_message(has_animation=True)
        msg.animation.file_id = "anim_fid"
        raw_mp4 = b"\x00\x00\x00\x18ftyp" + b"\x00" * 50
        gif_out = make_gif_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=raw_mp4)):
            with patch("bot.media.to_gif", new=AsyncMock(return_value=gif_out)):
                payload = await build_media_payload(msg, self._mock_bot())

        assert payload is not None
        assert payload.kind == MediaKind.GIF
        assert payload.data == gif_out
        assert payload.file_id == "anim_fid"

    @pytest.mark.asyncio
    async def test_animation_calls_to_gif_with_mp4_ext(self):
        msg = make_message(has_animation=True)
        gif_calls: list[tuple] = []

        async def capture_gif(data, ext, **kw):
            gif_calls.append((data, ext))
            return make_gif_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=b"\x00" * 50)):
            with patch("bot.media.to_gif", side_effect=capture_gif):
                await build_media_payload(msg, self._mock_bot())

        assert len(gif_calls) == 1
        assert gif_calls[0][1] == "mp4"

    # Video sticker

    @pytest.mark.asyncio
    async def test_video_sticker_yields_gif(self):
        msg = make_message(has_sticker=True, sticker_is_video=True)
        msg.sticker.file_id = "vsticker_fid"
        raw_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 50
        gif_out = make_gif_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=raw_webm)):
            with patch("bot.media.to_gif", new=AsyncMock(return_value=gif_out)):
                payload = await build_media_payload(msg, self._mock_bot())

        assert payload is not None
        assert payload.kind == MediaKind.GIF
        assert payload.file_id == "vsticker_fid"

    @pytest.mark.asyncio
    async def test_video_sticker_calls_to_gif_with_webm_ext(self):
        msg = make_message(has_sticker=True, sticker_is_video=True)
        gif_calls: list[tuple] = []

        async def capture_gif(data, ext, **kw):
            gif_calls.append((data, ext))
            return make_gif_bytes()

        with patch("bot.media._download", new=AsyncMock(return_value=b"\x00" * 50)):
            with patch("bot.media.to_gif", side_effect=capture_gif):
                await build_media_payload(msg, self._mock_bot())

        assert gif_calls[0][1] == "webm"

    # Unsupported types

    @pytest.mark.asyncio
    async def test_animated_sticker_returns_none(self):
        """TGS/Lottie stickers are not supported and must return None."""
        msg = make_message(has_sticker=True, sticker_is_animated=True)
        payload = await build_media_payload(msg, self._mock_bot())
        assert payload is None

    @pytest.mark.asyncio
    async def test_static_sticker_returns_none(self):
        msg = make_message(
            has_sticker=True, sticker_is_animated=False, sticker_is_video=False
        )
        payload = await build_media_payload(msg, self._mock_bot())
        assert payload is None

    @pytest.mark.asyncio
    async def test_text_only_returns_none(self):
        msg = make_message(text="hello, world!")
        payload = await build_media_payload(msg, self._mock_bot())
        assert payload is None

    # Size limit

    @pytest.mark.asyncio
    async def test_oversized_photo_raises_size_limit_error(self):
        msg = make_message(has_photo=True)
        limit = MediaKind.JPEG.size_limit_bytes
        huge_jpeg = make_jpeg_bytes()
        oversized = b"\xff\xd8\xff" + b"\x00" * limit

        with patch("bot.media._download", new=AsyncMock(return_value=huge_jpeg)):
            with patch("bot.media.to_jpeg", new=AsyncMock(return_value=oversized)):
                with pytest.raises(MediaSizeLimitError):
                    await build_media_payload(msg, self._mock_bot())

    @pytest.mark.asyncio
    async def test_oversized_animation_raises_size_limit_error(self):
        msg = make_message(has_animation=True)
        limit = MediaKind.GIF.size_limit_bytes
        oversized_gif = b"GIF89a" + b"\x00" * limit

        with patch("bot.media._download", new=AsyncMock(return_value=b"\x00" * 50)):
            with patch("bot.media.to_gif", new=AsyncMock(return_value=oversized_gif)):
                with pytest.raises(MediaSizeLimitError):
                    await build_media_payload(msg, self._mock_bot())
