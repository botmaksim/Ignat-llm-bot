"""
tests/test_converter.py — Unit tests for converter.py

- JPEG conversion uses real Pillow (no mocking) on in-memory pixel data.
- GIF conversion mocks ``asyncio.create_subprocess_exec`` so no ffmpeg binary
  is required. The mock writes test GIF bytes to the output path to simulate
  what ffmpeg would produce.

Covers:
- to_jpeg: JPEG passthrough, RGB PNG → JPEG, RGBA PNG → JPEG (transparency),
  Palette PNG → JPEG, already-JPEG fast-path, empty input error
- to_gif: MP4 → GIF (ffmpeg mock), WebM → GIF, GIF passthrough (magic bytes),
  ffmpeg timeout → ConversionError, ffmpeg non-zero rc → ConversionError,
  ffmpeg not found → ConversionError, empty output → ConversionError
"""

from __future__ import annotations

import asyncio
import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from bot.converter import ConversionError, _GIF_MAGIC, _JPEG_MAGIC, to_gif, to_jpeg
from tests.conftest import make_gif_bytes, make_jpeg_bytes, make_png_bytes

# Helpers

def _make_palette_png() -> bytes:
    img = Image.new("P", (10, 10))
    img.putpalette([i for i in range(256)] * 3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_valid_jpeg(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _is_valid_gif(data: bytes) -> bool:
    return any(data[:6] == magic for magic in _GIF_MAGIC)

# ffmpeg subprocess mock factory

def _make_ffmpeg_mock(gif_output: bytes, returncode: int = 0, stderr: bytes = b""):
    async def _subprocess(*args, **kwargs):
        # Find the .gif output path among the ffmpeg arguments
        for arg in reversed(args):
            if isinstance(arg, str) and arg.endswith(".gif"):
                os.makedirs(os.path.dirname(arg) or ".", exist_ok=True)
                with open(arg, "wb") as f:
                    f.write(gif_output)
                break

        proc = MagicMock()
        proc.returncode = returncode
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", stderr))
        return proc

    return _subprocess

# to_jpeg — JPEG passthrough

class TestToJpegPassthrough:

    @pytest.mark.asyncio
    async def test_jpeg_bytes_returned_unchanged(self):
        original = make_jpeg_bytes()
        result = await to_jpeg(original)
        assert result is original, "Expected identical object (no re-encode)"

    @pytest.mark.asyncio
    async def test_jpeg_magic_bytes_preserved(self):
        result = await to_jpeg(make_jpeg_bytes())
        assert result[:3] == _JPEG_MAGIC


# to_jpeg — real format conversions (no mocking)

class TestToJpegConversions:

    @pytest.mark.asyncio
    async def test_rgb_png_to_jpeg(self):
        png = make_png_bytes(mode="RGB")
        result = await to_jpeg(png)
        assert result[:3] == _JPEG_MAGIC
        assert _is_valid_jpeg(result)

    @pytest.mark.asyncio
    async def test_rgba_png_to_jpeg_composites_on_white(self):
        png = make_png_bytes(mode="RGBA")
        result = await to_jpeg(png)
        assert _is_valid_jpeg(result)
        with Image.open(io.BytesIO(result)) as img:
            assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_palette_png_to_jpeg(self):
        png = _make_palette_png()
        result = await to_jpeg(png)
        assert _is_valid_jpeg(result)

    @pytest.mark.asyncio
    async def test_output_is_smaller_than_lossless_png_for_large_image(self):
        import random
        import struct

        img = Image.new("RGB", (200, 200))
        pixels = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                  for _ in range(200 * 200)]
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=0)
        large_png = buf.getvalue()

        result = await to_jpeg(large_png, quality=85)
        assert len(result) < len(large_png), (
            f"Expected JPEG ({len(result)} B) < uncompressed PNG ({len(large_png)} B)"
        )

    @pytest.mark.asyncio
    async def test_custom_quality_affects_output_size(self):
        import random
        img = Image.new("RGB", (200, 200))
        img.putdata([(random.randint(0,255), random.randint(0,255), random.randint(0,255))
                     for _ in range(200*200)])
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=0)
        large_png = buf.getvalue()

        hi = await to_jpeg(large_png, quality=95)
        lo = await to_jpeg(large_png, quality=5)
        assert len(lo) < len(hi), (
            f"Expected quality=5 ({len(lo)} B) < quality=95 ({len(hi)} B)"
        )

    @pytest.mark.asyncio
    async def test_invalid_bytes_raise_conversion_error(self):
        with pytest.raises(ConversionError, match="Pillow"):
            await to_jpeg(b"not an image at all")

    @pytest.mark.asyncio
    async def test_empty_bytes_raise_conversion_error(self):
        with pytest.raises(ConversionError):
            await to_jpeg(b"")

# to_gif — GIF passthrough

class TestToGifPassthrough:

    @pytest.mark.asyncio
    async def test_gif89a_returned_unchanged(self):
        original = make_gif_bytes()
        assert original[:3] == b"GIF"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await to_gif(original, "gif")
            mock_exec.assert_not_called()

        assert result is original

    @pytest.mark.asyncio
    async def test_gif87a_returned_unchanged(self):
        gif87 = b"GIF87a" + b"\x00" * 50
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await to_gif(gif87, "gif")
            mock_exec.assert_not_called()
        assert result is gif87

# to_gif — ffmpeg conversion (mocked subprocess)

class TestToGifConversion:

    _FAKE_GIF = b"GIF89a" + b"\xAB" * 200

    @pytest.mark.asyncio
    async def test_mp4_to_gif_calls_ffmpeg(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(self._FAKE_GIF),
        ):
            result = await to_gif(b"\x00" * 100, "mp4")

        assert result == self._FAKE_GIF

    @pytest.mark.asyncio
    async def test_webm_to_gif_calls_ffmpeg(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(self._FAKE_GIF),
        ):
            result = await to_gif(b"\x00" * 100, "webm")

        assert result == self._FAKE_GIF

    @pytest.mark.asyncio
    async def test_ffmpeg_receives_correct_flags(self):
        captured_args: list[tuple] = []

        async def recording_mock(*args, **kwargs):
            captured_args.append(args)
            for arg in reversed(args):
                if isinstance(arg, str) and arg.endswith(".gif"):
                    with open(arg, "wb") as f:
                        f.write(self._FAKE_GIF)
                    break
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=recording_mock):
            await to_gif(b"\x00" * 100, "mp4")

        args = captured_args[0]
        assert "ffmpeg" in args
        assert "-loop" in args
        assert "0" in args
        assert any("palettegen" in str(a) for a in args), "Palette filter missing"

    @pytest.mark.asyncio
    async def test_ffmpeg_timeout_raises_conversion_error(self):
        async def slow_ffmpeg(*args, **kwargs):
            proc = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=slow_ffmpeg):
            with pytest.raises(ConversionError, match="timed out"):
                await to_gif(b"\x00" * 100, "mp4", timeout=0.01)

    @pytest.mark.asyncio
    async def test_ffmpeg_nonzero_exit_raises_conversion_error(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(b"", returncode=1, stderr=b"codec error"),
        ):
            with pytest.raises(ConversionError, match="exited 1"):
                await to_gif(b"\x00" * 100, "mp4")

    @pytest.mark.asyncio
    async def test_ffmpeg_not_found_raises_conversion_error(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("No such file: ffmpeg"),
        ):
            with pytest.raises(ConversionError, match="ffmpeg not found"):
                await to_gif(b"\x00" * 100, "mp4")

    @pytest.mark.asyncio
    async def test_ffmpeg_empty_output_raises_conversion_error(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(b"", returncode=0),
        ):
            with pytest.raises(ConversionError, match="empty"):
                await to_gif(b"\x00" * 100, "mp4")

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up_on_success(self, tmp_path):
        import tempfile
        initial_tmp = set(os.listdir(tempfile.gettempdir()))

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(self._FAKE_GIF),
        ):
            await to_gif(b"\x00" * 100, "mp4")

        final_tmp = set(os.listdir(tempfile.gettempdir()))
        new_files = final_tmp - initial_tmp
        assert not any(f.endswith((".mp4", ".gif")) for f in new_files), (
            f"Temp files not cleaned up: {new_files}"
        )

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up_on_error(self):
        import tempfile
        initial_tmp = set(os.listdir(tempfile.gettempdir()))

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_ffmpeg_mock(b"", returncode=1, stderr=b"err"),
        ):
            with pytest.raises(ConversionError):
                await to_gif(b"\x00" * 100, "mp4")

        final_tmp = set(os.listdir(tempfile.gettempdir()))
        new_files = final_tmp - initial_tmp
        assert not any(f.endswith((".mp4", ".gif")) for f in new_files)
