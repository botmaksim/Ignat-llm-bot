"""
tests/test_parser.py — Unit tests for parser.py

Covers:
- format_reply_context: all media types and text truncation
- parse_incoming_message: user name, reply, image, voice, audio, GIF, stickers
- parse_llm_formatting: known tags, unknown tags, nested content
- extract_llm_actions: pause, sticker, emotion tokens
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from bot.parser import (
    extract_llm_actions,
    format_reply_context,
    parse_incoming_message,
    parse_llm_formatting,
)
from tests.conftest import make_message

# format_reply_context

class TestFormatReplyContext:

    def _msg(self, **kwargs) -> MagicMock:
        m = MagicMock()
        m.text = kwargs.get("text", None)
        m.caption = kwargs.get("caption", None)
        m.sticker = kwargs.get("sticker", None)
        m.animation = kwargs.get("animation", None)
        m.photo = kwargs.get("photo", None)
        m.voice = kwargs.get("voice", None)
        m.audio = kwargs.get("audio", None)
        m.video = kwargs.get("video", None)
        m.video_note = kwargs.get("video_note", None)
        return m

    def test_sticker_reply(self):
        reply = self._msg(sticker=MagicMock())
        assert format_reply_context(reply) == "[STICKER]"

    def test_animation_reply(self):
        reply = self._msg(animation=MagicMock())
        assert format_reply_context(reply) == "[GIF]"

    def test_short_text_unchanged(self):
        reply = self._msg(text="Hi there")
        assert format_reply_context(reply) == "Hi there"

    def test_long_text_truncated(self):
        long_text = "A" * 60
        result = format_reply_context(self._msg(text=long_text))
        assert result.endswith("...")
        assert len(result) == 50

    def test_photo_fallback(self):
        reply = self._msg(photo=MagicMock())
        assert format_reply_context(reply) == "User sent a photo"

    def test_voice_fallback(self):
        reply = self._msg(voice=MagicMock())
        assert format_reply_context(reply) == "User sent a voice message"

    def test_generic_fallback(self):
        reply = self._msg()
        assert format_reply_context(reply) == "Media message"


# parse_incoming_message

class TestParseIncomingMessage:

    def test_plain_text_contains_name_and_text(self):
        msg = make_message(text="hello world")
        result = parse_incoming_message(msg)
        assert "[Name: Alice]" in result
        assert "hello world" in result

    def test_no_text_no_text_token(self):
        msg = make_message(text=None)
        result = parse_incoming_message(msg)
        # Should not have a trailing empty token
        assert result.strip() != ""
        assert "None" not in result

    def test_image_description_injected(self):
        msg = make_message(text=None, has_photo=True)
        result = parse_incoming_message(msg, image_description="a cat")
        assert "[IMAGE: a cat]" in result

    def test_image_fallback_when_no_description(self):
        msg = make_message(text=None, has_photo=True)
        result = parse_incoming_message(msg)
        assert "[IMAGE:" in result
        assert "photo" in result.lower()

    def test_voice_transcript_injected(self):
        msg = make_message(text=None, has_voice=True)
        result = parse_incoming_message(msg, voice_transcript="transcribed text")
        assert "[VOICE: transcribed text]" in result

    def test_audio_transcript_injected(self):
        msg = make_message(text=None, has_audio=True)
        result = parse_incoming_message(msg, voice_transcript="audio transcript")
        assert "[AUDIO: audio transcript]" in result

    def test_gif_description_injected(self):
        msg = make_message(text=None, has_animation=True)
        result = parse_incoming_message(msg, gif_description="cat dancing")
        assert "[GIF: cat dancing]" in result

    def test_static_sticker_lookup(self):
        msg = make_message(has_sticker=True)
        stickers_map = {
            "happy": {"file_id": "sticker_file_id", "description": "laughing character"}
        }
        result = parse_incoming_message(msg, stickers_map=stickers_map)
        assert "[STICKER:" in result
        assert "laughing character" in result

    def test_static_sticker_unknown_fallback(self):
        msg = make_message(has_sticker=True)
        result = parse_incoming_message(msg, stickers_map={})
        assert "[STICKER:" in result

    def test_animated_sticker_uses_gif_description(self):
        msg = make_message(has_sticker=True, sticker_is_animated=True)
        result = parse_incoming_message(msg, gif_description="spinning cube")
        assert "[GIF: spinning cube]" in result

    def test_video_sticker_uses_gif_description(self):
        msg = make_message(has_sticker=True, sticker_is_video=True)
        result = parse_incoming_message(msg, gif_description="bouncing ball")
        assert "[GIF: bouncing ball]" in result

    def test_reply_included(self):
        reply = MagicMock()
        reply.text = "original"
        reply.caption = None
        reply.sticker = None
        reply.animation = None
        reply.photo = None
        reply.voice = None
        reply.audio = None
        reply.video = None
        reply.video_note = None
        msg = make_message(text="response", reply_to=reply)
        result = parse_incoming_message(msg)
        assert "[REPLY:" in result
        assert "original" in result

    def test_missing_from_user_defaults_to_user(self):
        msg = make_message(text="hi")
        msg.from_user = None
        result = parse_incoming_message(msg)
        assert "[Name: User]" in result

# parse_llm_formatting

class TestParseLlmFormatting:

    def test_bold_tag(self):
        result = parse_llm_formatting("[BEGIN:bold]hello[END:bold]")
        assert result == "<b>hello</b>"

    def test_italic_tag(self):
        result = parse_llm_formatting("[BEGIN:italic]world[END:italic]")
        assert result == "<i>world</i>"

    def test_code_tag(self):
        result = parse_llm_formatting("[BEGIN:code]x = 1[END:code]")
        assert result == "<code>x = 1</code>"

    def test_spoiler_tag(self):
        result = parse_llm_formatting("[BEGIN:spoiler]secret[END:spoiler]")
        assert result == "<tg-spoiler>secret</tg-spoiler>"

    def test_unknown_tag_left_unchanged(self):
        text = "[BEGIN:rainbow]colourful[END:rainbow]"
        assert parse_llm_formatting(text) == text

    def test_case_insensitive(self):
        result = parse_llm_formatting("[BEGIN:BOLD]hi[END:BOLD]")
        assert result == "<b>hi</b>"

    def test_multiple_tags(self):
        text = "[BEGIN:bold]A[END:bold] and [BEGIN:italic]B[END:italic]"
        assert parse_llm_formatting(text) == "<b>A</b> and <i>B</i>"

    def test_multiline_content_preserved(self):
        text = "[BEGIN:code]line1\nline2[END:code]"
        result = parse_llm_formatting(text)
        assert "<code>line1\nline2</code>" == result

    def test_plain_text_unchanged(self):
        text = "no tags here"
        assert parse_llm_formatting(text) == text

    def test_custom_formatting_map(self):
        custom = {"shout": {"open": "<<", "close": ">>"}}
        result = parse_llm_formatting("[BEGIN:shout]hey[END:shout]", custom)
        assert result == "<<hey>>"


# extract_llm_actions

class TestExtractLlmActions:

    def test_pause_tokens(self):
        actions = extract_llm_actions("Hello [PAUSE] world [PAUSE:3]")
        assert len(actions["pauses"]) == 2

    def test_sticker_tokens(self):
        actions = extract_llm_actions("text [STICKER:happy] more [STICKER:sad]")
        assert actions["stickers"] == ["happy", "sad"]

    def test_emotion_tokens(self):
        actions = extract_llm_actions("[EMOTION:excited] text")
        assert actions["emotions"] == ["excited"]

    def test_empty_text(self):
        actions = extract_llm_actions("")
        assert actions == {"pauses": [], "stickers": [], "emotions": []}

    def test_case_insensitive(self):
        actions = extract_llm_actions("[pause] [STICKER:Happy] [emotion:calm]")
        assert len(actions["pauses"]) == 1
        assert actions["stickers"] == ["Happy"]
        assert actions["emotions"] == ["calm"]
