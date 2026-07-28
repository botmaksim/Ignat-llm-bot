"""
parser.py — Formatting helpers for Telegram ↔ LLM message conversion.

Two responsibilities:
  1. :func:`parse_incoming_message` — converts an aiogram ``Message`` (with
     optional resolved media descriptions) into a structured string for the LLM.
  2. :func:`parse_llm_formatting` — converts ``[BEGIN:tag] … [END:tag]`` markup
     from LLM responses into Telegram HTML tags.
  3. :func:`extract_llm_actions` — extracts ``[PAUSE]``, ``[STICKER:]``,
     ``[EMOTION:]`` control tokens for logging and debugging.
"""

from __future__ import annotations

import re
from typing import Optional

from aiogram.types import Message


# Reply formatting

_MAX_REPLY_LENGTH = 50


def format_reply_context(reply_msg: Message) -> str:
    """
    Produce a short human-readable summary of a replied-to message.

    - Sticker → ``[STICKER]``
    - Animation → ``[GIF]``
    - Media without caption → generic label
    - Long text → truncated to 47 chars + ``"..."``
    """
    if reply_msg.sticker:
        return "[STICKER]"
    if reply_msg.animation:
        return "[GIF]"

    text = reply_msg.text or reply_msg.caption or ""
    if not text:
        if reply_msg.photo:
            text = "User sent a photo"
        elif reply_msg.voice:
            text = "User sent a voice message"
        elif reply_msg.audio:
            text = "User sent an audio file"
        elif reply_msg.video or reply_msg.video_note:
            text = "User sent a video"
        else:
            text = "Media message"

    if len(text) > _MAX_REPLY_LENGTH:
        return text[: _MAX_REPLY_LENGTH - 3] + "..."
    return text

# Incoming message → LLM string

def parse_incoming_message(
    message: Message,
    stickers_map: Optional[dict] = None,
    image_description: Optional[str] = None,
    voice_transcript: Optional[str] = None,
    gif_description: Optional[str] = None,
) -> str:
    """
    Serialise a Telegram message into a structured string for the LLM context.

    Token order:
      ``[Name: <first_name>]``
      ``[REPLY: '<excerpt>']``       (if the message is a reply)
      ``[IMAGE/VOICE/AUDIO/GIF/STICKER: <description>]``  (if media is present)
      ``<raw message text or caption>``

    Args:
        message:           The incoming aiogram Message object.
        stickers_map:      ``{tag: {"file_id": ..., "description": ...}}`` dict
                           loaded from ``stickers.json``.
        image_description: Text description returned by the Vision model.
        voice_transcript:  Transcript returned by the STT model.
        gif_description:   Text description returned by the Vision model for GIF/WebM/TGS.

    Returns:
        A single string suitable for appending to the LLM history as a user turn.
    """
    user_name = (message.from_user.first_name if message.from_user else "User") or "User"
    tokens: list[str] = [f"[Name: {user_name}]"]

    # ── Reply context ────────────────────────────────────────────────────────
    if message.reply_to_message:
        excerpt = format_reply_context(message.reply_to_message)
        tokens.append(f"[REPLY: '{excerpt}']")

    # ── Media tokens ─────────────────────────────────────────────────────────
    if message.photo:
        desc = image_description or "User sent a photo"
        tokens.append(f"[IMAGE: {desc}]")

    elif message.voice:
        desc = voice_transcript or "User sent a voice message"
        tokens.append(f"[VOICE: {desc}]")

    elif message.audio:
        desc = voice_transcript or "User sent an audio file"
        tokens.append(f"[AUDIO: {desc}]")

    elif message.animation:
        desc = gif_description or "User sent a GIF"
        tokens.append(f"[GIF: {desc}]")

    elif message.sticker:
        sticker = message.sticker
        if sticker.is_video or sticker.is_animated:
            desc = gif_description or "User sent an animated sticker"
            tokens.append(f"[GIF: {desc}]")
        else:
            desc = "User sent a sticker"
            if stickers_map:
                for _tag, info in stickers_map.items():
                    if info.get("file_id") == sticker.file_id:
                        tag_label = _tag
                        tag_desc = info.get("description", "")
                        desc = f"[{tag_label}] {tag_desc}".strip()
                        break
            tokens.append(f"[STICKER: {desc}]")

    raw_text = (message.text or message.caption or "").strip()
    if raw_text:
        tokens.append(raw_text)

    return " ".join(tokens)

# LLM response → Telegram HTML

_DEFAULT_FORMATTING: dict[str, dict[str, str]] = {
    "bold":      {"open": "<b>",          "close": "</b>"},
    "italic":    {"open": "<i>",          "close": "</i>"},
    "code":      {"open": "<code>",       "close": "</code>"},
    "strike":    {"open": "<s>",          "close": "</s>"},
    "underline": {"open": "<u>",          "close": "</u>"},
    "spoiler":   {"open": "<tg-spoiler>", "close": "</tg-spoiler>"},
}


def parse_llm_formatting(
    text: str,
    formatting_map: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    """
    Replace ``[BEGIN:tag] … [END:tag]`` markup with Telegram HTML tags.

    Example::

        parse_llm_formatting("[BEGIN:bold]hello[END:bold]")
        # → "<b>hello</b>"

    Args:
        text:           Raw LLM response text.
        formatting_map: Tag → ``{"open": str, "close": str}`` mapping.
                        Defaults to ``_DEFAULT_FORMATTING`` if ``None``.

    Returns:
        The text with all recognised formatting tags replaced by HTML.
    """
    mapping = formatting_map if formatting_map is not None else _DEFAULT_FORMATTING

    result = text
    for tag_name, html in mapping.items():
        pattern = re.compile(
            rf"\[BEGIN:{re.escape(tag_name)}\](.*?)\[END:{re.escape(tag_name)}\]",
            flags=re.IGNORECASE | re.DOTALL,
        )
        result = pattern.sub(
            lambda m, o=html["open"], c=html["close"]: f"{o}{m.group(1)}{c}",
            result,
        )

    return result


# LLM action extraction

def extract_llm_actions(text: str) -> dict[str, list[str]]:
    """
    Extract control tokens from a raw LLM response.

    Tokens recognised:
    - ``[PAUSE]`` or ``[PAUSE:N]``
    - ``[STICKER:tag]``
    - ``[EMOTION:description]``

    This is used for diagnostics; the actual dispatch logic lives in
    ``handlers.py``.

    Returns:
        Dict with keys ``"pauses"``, ``"stickers"``, ``"emotions"``.
    """
    return {
        "pauses":   re.findall(r"\[PAUSE(?::\d+)?\]", text, flags=re.IGNORECASE),
        "stickers": re.findall(r"\[STICKER:([a-zA-Z0-9_]+)\]", text, flags=re.IGNORECASE),
        "emotions": re.findall(r"\[EMOTION:([^\]]+)\]", text, flags=re.IGNORECASE),
    }
