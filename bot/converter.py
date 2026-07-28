"""
converter.py — Media format conversion to model-accepted formats.

All Colab inference models accept exactly two binary formats:
  - **JPEG** for photos and static images.
  - **GIF**  for animations, video stickers, and any moving content.

Audio (OGG, MP3) is passed through unchanged — it goes to the STT model
which accepts both formats natively.

Conversion pipeline
-------------------
+----------------------+--------------------+-----------+
| Telegram media type  | Downloaded format  | Output    |
+======================+====================+===========+
| message.photo        | JPEG (server-side) | JPEG      |
| message.voice        | OGG Opus           | OGG  ──── no conversion
| message.audio        | arbitrary          | MP3  ──── no conversion
| message.animation    | MP4 (server-side)  | GIF  ──── ffmpeg
| sticker (video)      | WebM/VP9           | GIF  ──── ffmpeg
| sticker (animated)   | not supported      | —
| sticker (static)     | not supported      | —
+----------------------+--------------------+-----------+

Design principles
-----------------
- :func:`to_jpeg` runs Pillow in a thread-pool executor (CPU-bound, non-blocking).
- :func:`to_gif` spawns ``ffmpeg`` as a non-blocking async subprocess.
- Both functions log input/output byte sizes and elapsed time for observability.
- :class:`ConversionError` is the single public exception type; callers need
  only catch this one error to handle all conversion failures.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Final

logger = logging.getLogger(__name__)

# Exceptions

class ConversionError(Exception):
    pass

# Constants

_JPEG_MAGIC: Final = b"\xff\xd8\xff"
_GIF_MAGIC: Final = (b"GIF87a", b"GIF89a")
_JPEG_QUALITY: Final = 85

_GIF_FILTER: Final = (
    "fps=10,scale=480:-1:flags=lanczos,"
    "split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer"
)


# JPEG conversion

def _to_jpeg_sync(data: bytes, quality: int) -> bytes:
    """
    Synchronous JPEG conversion — runs inside a thread-pool executor.

    Handles all colour modes Telegram photos may arrive in:
    - ``RGB``  : passed to JPEG encoder directly.
    - ``RGBA`` : alpha channel composited onto white before encoding.
    - ``LA``   : greyscale+alpha composited onto white.
    - ``P``    : palette mode converted to ``RGB``.
    - other    : converted to ``RGB`` via Pillow's generic convert().

    Raises:
        ConversionError: On Pillow decode or encode failure.
    """
    if data[:3] == _JPEG_MAGIC:
        return data

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ConversionError("Pillow is not installed: pip install Pillow") from exc

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.mode in ("RGBA", "LA"):
                # Composite transparent pixels onto a white background.
                background = Image.new("RGB", img.size, (255, 255, 255))
                alpha_channel = img.split()[-1]  # last channel is always alpha
                background.paste(img.convert("RGB"), mask=alpha_channel)
                img_rgb = background
            elif img.mode == "P":
                img_rgb = img.convert("RGB")
            elif img.mode != "RGB":
                img_rgb = img.convert("RGB")
            else:
                img_rgb = img

            buf = io.BytesIO()
            img_rgb.save(
                buf,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            return buf.getvalue()

    except Exception as exc:
        raise ConversionError(f"Pillow failed to convert image to JPEG: {exc}") from exc


async def to_jpeg(data: bytes, quality: int = _JPEG_QUALITY) -> bytes:
    """
    Convert any Pillow-supported image format to JPEG bytes.

    Already-JPEG input is returned unchanged (magic-byte check, no re-encode).
    Conversion runs in a thread-pool executor so it never blocks the event loop.

    Args:
        data:    Raw image bytes (JPEG, PNG, WebP, BMP, …).
        quality: JPEG quality factor 1–95. Default is :data:`_JPEG_QUALITY`.

    Returns:
        JPEG-encoded bytes.

    Raises:
        ConversionError: On any decode or encode failure.
    """
    t0 = time.monotonic()

    loop = asyncio.get_running_loop()
    result: bytes = await loop.run_in_executor(None, _to_jpeg_sync, data, quality)

    elapsed = time.monotonic() - t0
    logger.debug(
        "Image converted to JPEG",
        extra={
            "input_bytes":  len(data),
            "output_bytes": len(result),
            "compress_ratio": round(len(data) / max(len(result), 1), 2),
            "elapsed_ms":   round(elapsed * 1000),
            "skipped":      data[:3] == _JPEG_MAGIC,
        },
    )
    return result

# GIF conversion

async def to_gif(
    data: bytes,
    source_ext: str,
    *,
    timeout: float = 60.0,
) -> bytes:
    """
    Convert a video or animation file to GIF using ``ffmpeg``.

    Supports ``mp4``, ``webm``, and ``gif`` inputs.  For already-GIF data
    (detected via magic bytes) the bytes are returned as-is to avoid quality
    degradation from a needless re-encode.

    The source data is written to a temporary file (ffmpeg needs seekable
    input for container formats), ffmpeg is run, and the result is read back.
    Both temp files are deleted in a ``finally`` block regardless of outcome.

    Args:
        data:       Raw bytes of the source media.
        source_ext: File extension without leading dot, e.g. ``"mp4"`` or
                    ``"webm"``.  Used to choose the correct temp-file suffix
                    so ffmpeg can auto-detect the input container.
        timeout:    Seconds to wait for ffmpeg before raising
                    :class:`ConversionError`.

    Returns:
        GIF-encoded bytes.

    Raises:
        ConversionError: If ``ffmpeg`` is not found, exits non-zero, times out,
                         or produces empty output.
    """
    if any(data[:6] == magic for magic in _GIF_MAGIC):
        logger.debug(
            "GIF passthrough (already GIF)",
            extra={"input_bytes": len(data)},
        )
        return data

    t0 = time.monotonic()
    src_path = dst_path = ""

    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{source_ext}", delete=False
        ) as src_f:
            src_f.write(data)
            src_path = src_f.name

        dst_path = src_path[: -len(source_ext) - 1] + ".gif"

        logger.debug(
            "Starting GIF conversion",
            extra={
                "source_ext":  source_ext,
                "input_bytes": len(data),
                "src_path":    src_path,
            },
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",             
                "-i", src_path,
                "-vf", _GIF_FILTER,
                "-loop", "0",      
                dst_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ConversionError(
                "ffmpeg not found. Install it: apt-get install ffmpeg"
            ) from exc

        try:
            _, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ConversionError(
                f"ffmpeg timed out after {timeout:.0f}s "
                f"converting {source_ext.upper()} → GIF"
            )

        if proc.returncode != 0:
            stderr_text = stderr_b.decode(errors="replace")
            meaningful = [ln for ln in stderr_text.splitlines() if ln.strip()]
            last_line = meaningful[-1] if meaningful else "no stderr output"
            raise ConversionError(
                f"ffmpeg exited {proc.returncode} converting "
                f"{source_ext.upper()} → GIF: {last_line}"
            )

        if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
            raise ConversionError(
                "ffmpeg succeeded but produced an empty output file"
            )

        with open(dst_path, "rb") as out_f:
            result = out_f.read()

    finally:
        for path in (src_path, dst_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError as exc:
                    logger.warning(
                        "Failed to delete temp file",
                        extra={"path": path, "error": str(exc)},
                    )

    elapsed = time.monotonic() - t0
    logger.debug(
        "Animation converted to GIF",
        extra={
            "source_ext":   source_ext,
            "input_bytes":  len(data),
            "output_bytes": len(result),
            "elapsed_ms":   round(elapsed * 1000),
        },
    )
    return result
