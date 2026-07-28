"""
tests/test_handlers.py — Unit tests for handlers.py helper functions.

Tests the pure / deterministic logic that does not require a running
aiogram Dispatcher:

- _should_respond: always responds to all messages (code may be rewritten, so bot replies on messages where it was mentioned for example)
- _get_state / evict_stale_states: init schema, TTL eviction
- _stream_response: single chunk, PAUSE splitting, interrupt signal,
  sticker dispatch, placeholder file_id, control tag stripping, empty response
"""

from __future__ import annotations

import time
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.handlers import (
    _chat_states,
    _CHAT_STATE_TTL_S,
    _get_state,
    _should_respond,
    _stream_response,
    evict_stale_states,
)
from tests.conftest import make_message

# _should_respond

class TestShouldRespond:

    def _config(self, username: str = "testbot") -> MagicMock:
        cfg = MagicMock()
        cfg.bot.username = username
        return cfg

    def test_private_chat_always_responds(self):
        msg = make_message(chat_type="private")
        assert _should_respond(msg, self._config()) is True

    def test_group_without_mention_responds(self):
        msg = make_message(text="no mention here", chat_type="group")
        assert _should_respond(msg, self._config()) is True

    def test_group_with_mention_responds(self):
        msg = make_message(text="@testbot help", chat_type="group")
        assert _should_respond(msg, self._config()) is True

    def test_supergroup_with_mention_responds(self):
        msg = make_message(text="@testbot hi", chat_type="supergroup")
        assert _should_respond(msg, self._config()) is True



# _get_state / evict_stale_states

class TestChatState:

    def setup_method(self):
        _chat_states.clear()

    def test_new_state_has_correct_schema(self):
        state = _get_state(99)
        assert state["history"] == []
        assert state["is_typing"] is False
        assert state["interrupt_signal"] is False
        assert isinstance(state["last_active"], float)

    def test_same_chat_id_returns_same_object(self):
        assert _get_state(42) is _get_state(42)

    def test_different_chat_ids_get_different_objects(self):
        s1 = _get_state(1)
        s2 = _get_state(2)
        s1["history"].append("x")
        assert s2["history"] == []

    def test_eviction_removes_stale_entries(self):
        _chat_states[111] = {
            "history": [], "is_typing": False,
            "interrupt_signal": False,
            "last_active": time.monotonic() - _CHAT_STATE_TTL_S - 1,
        }
        _chat_states[222] = {
            "history": [], "is_typing": False,
            "interrupt_signal": False,
            "last_active": time.monotonic(),
        }
        evicted = evict_stale_states()
        assert evicted == 1
        assert 111 not in _chat_states
        assert 222 in _chat_states

    def test_eviction_on_empty_store_returns_zero(self):
        assert evict_stale_states() == 0

    def test_eviction_removes_all_stale_entries(self):
        for cid in range(5):
            _chat_states[cid] = {
                "history": [], "is_typing": False,
                "interrupt_signal": False,
                "last_active": time.monotonic() - _CHAT_STATE_TTL_S - 1,
            }
        assert evict_stale_states() == 5
        assert len(_chat_states) == 0


# _stream_response

class TestStreamResponse:

    def _config(self, typing_speed: float = 0.0) -> MagicMock:
        cfg = MagicMock()
        cfg.bot.typing_speed = typing_speed
        cfg.formatting = {}
        cfg.stickers = {}
        return cfg

    def _state(self) -> dict:
        return {"interrupt_signal": False, "is_typing": True}

    @pytest.mark.asyncio
    async def test_single_chunk_sent(self):
        msg = AsyncMock()
        await _stream_response(msg, "hello world", self._state(), self._config(), 1)
        msg.answer.assert_called_once()
        assert "hello world" in msg.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_pause_splits_into_multiple_answers(self):
        msg = AsyncMock()
        await _stream_response(
            msg, "part one [PAUSE] part two [PAUSE] part three",
            self._state(), self._config(), 1,
        )
        assert msg.answer.call_count == 3

    @pytest.mark.asyncio
    async def test_pause_with_duration_also_splits(self):
        msg = AsyncMock()
        await _stream_response(
            msg, "first [PAUSE:3] second", self._state(), self._config(), 1
        )
        assert msg.answer.call_count == 2

    @pytest.mark.asyncio
    async def test_interrupt_signal_stops_after_first_chunk(self):
        msg = AsyncMock()
        state = self._state()

        async def set_interrupt(text):
            state["interrupt_signal"] = True

        msg.answer.side_effect = set_interrupt
        await _stream_response(
            msg, "first [PAUSE] second [PAUSE] third",
            state, self._config(), 1,
        )
        assert msg.answer.call_count == 1

    @pytest.mark.asyncio
    async def test_sticker_sent_after_chunk(self):
        msg = AsyncMock()
        cfg = self._config()
        cfg.stickers = {"happy": {"file_id": "REAL_FILE_ID_XYZ", "description": "smiling"}}
        await _stream_response(
            msg, "great! [STICKER:happy]", self._state(), cfg, 1
        )
        msg.answer_sticker.assert_called_once_with("REAL_FILE_ID_XYZ")

    @pytest.mark.asyncio
    async def test_unknown_sticker_tag_does_not_raise(self):
        msg = AsyncMock()
        cfg = self._config()
        cfg.stickers = {}
        await _stream_response(msg, "ok [STICKER:unknown]", self._state(), cfg, 1)
        msg.answer_sticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_sticker_with_placeholder_file_id_not_sent(self):
        msg = AsyncMock()
        cfg = self._config()
        cfg.stickers = {"sad": {"file_id": "CAACAgIA...", "description": "sad"}}
        await _stream_response(msg, "oh no [STICKER:sad]", self._state(), cfg, 1)
        msg.answer_sticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_emotion_tag_stripped_from_text(self):
        msg = AsyncMock()
        await _stream_response(
            msg, "hello [EMOTION:excited]", self._state(), self._config(), 1
        )
        sent = msg.answer.call_args[0][0]
        assert "[EMOTION:" not in sent
        assert "hello" in sent

    @pytest.mark.asyncio
    async def test_sticker_tag_stripped_from_text(self):
        msg = AsyncMock()
        cfg = self._config()
        cfg.stickers = {"happy": {"file_id": "FID", "description": "ok"}}
        await _stream_response(
            msg, "nice [STICKER:happy]", self._state(), cfg, 1
        )
        sent = msg.answer.call_args[0][0]
        assert "[STICKER:" not in sent
        assert "nice" in sent

    @pytest.mark.asyncio
    async def test_empty_response_sends_nothing(self):
        msg = AsyncMock()
        await _stream_response(msg, "", self._state(), self._config(), 1)
        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_response_sends_nothing(self):
        msg = AsyncMock()
        await _stream_response(msg, "   \n  ", self._state(), self._config(), 1)
        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_only_response_sends_nothing(self):
        msg = AsyncMock()
        await _stream_response(
            msg, "[PAUSE] [PAUSE] [PAUSE]", self._state(), self._config(), 1
        )
        msg.answer.assert_not_called()
