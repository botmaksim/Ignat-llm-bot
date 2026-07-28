"""
test_handlers_coverage.py — Edge-case coverage tests for handlers.

This module covers specific missed branches in media resolution, cold-start
orchestration, and complex exception scenarios to achieve maximum test coverage.
Block comments denote the specific sub-components or scenarios being tested.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aiogram.types import Message
from bot.handlers import (
    _resolve_media,
    _do_cold_start,
    process_user_message,
    _get_state,
)
from bot.media import MediaKind, MediaPayload
from bot.colab_api import ColabQuotaError, ColabExecError, ColabColdStartError
from tests.conftest import make_message

# Fixtures

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.bot.username = "testbot"
    cfg.bot.max_history = 5
    cfg.stickers = {}
    cfg.formatting = {}
    return cfg


@pytest.fixture
def mock_colab():
    client = AsyncMock()
    client.session.cold_start = AsyncMock()
    return client

# Media Resolution Tests

@pytest.mark.asyncio
@patch("bot.handlers.build_media_payload")
async def test_resolve_media_jpeg(mock_build, mock_colab):
    msg = make_message(has_photo=True)
    mock_build.return_value = MediaPayload(b"data", MediaKind.JPEG, "fid")
    mock_colab.describe_image.return_value = "A nice photo"
    
    res = await _resolve_media(msg, AsyncMock(), mock_colab, 123)
    assert res == {"image_description": "A nice photo"}


@pytest.mark.asyncio
@patch("bot.handlers.build_media_payload")
async def test_resolve_media_ogg(mock_build, mock_colab):
    msg = make_message(has_voice=True)
    mock_build.return_value = MediaPayload(b"data", MediaKind.OGG, "fid")
    mock_colab.transcribe_audio.return_value = "Hello spoken"
    
    res = await _resolve_media(msg, AsyncMock(), mock_colab, 123)
    assert res == {"voice_transcript": "Hello spoken"}


@pytest.mark.asyncio
@patch("bot.handlers.build_media_payload")
async def test_resolve_media_gif(mock_build, mock_colab):
    msg = make_message(has_animation=True)
    mock_build.return_value = MediaPayload(b"data", MediaKind.GIF, "fid")
    mock_colab.describe_gif.return_value = "A funny gif"
    
    res = await _resolve_media(msg, AsyncMock(), mock_colab, 123)
    assert res == {"gif_description": "A funny gif"}

# Cold Start Orchestration Tests

@pytest.mark.asyncio
async def test_do_cold_start_quota_error(mock_colab):
    msg = make_message()
    mock_colab.session.cold_start.side_effect = ColabQuotaError("Quota full")
    
    res = await _do_cold_start(msg, mock_colab, 123)
    assert res is False
    assert msg.answer.call_count == 2


@pytest.mark.asyncio
async def test_do_cold_start_exec_error(mock_colab):
    msg = make_message()
    mock_colab.session.cold_start.side_effect = ColabExecError("Crash")
    
    res = await _do_cold_start(msg, mock_colab, 123)
    assert res is False
    assert msg.answer.call_count == 2

# Message Processing Edge Cases

@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
@patch("bot.handlers._stream_response")
async def test_process_user_message_interrupt_existing(mock_stream, mock_resolve, mock_config, mock_colab):
    msg = make_message(text="hello")
    chat_id = msg.chat.id
    state = _get_state(chat_id)
    state["is_typing"] = True
    
    async def stop_typing():
        await asyncio.sleep(0.01)
        state["is_typing"] = False
    
    asyncio.create_task(stop_typing())
    await process_user_message(msg, mock_config, AsyncMock(), mock_colab)
    assert state["interrupt_signal"] is False


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
@patch("bot.handlers._stream_response")
async def test_process_user_message_sticker_registration(mock_stream, mock_resolve, mock_config, mock_colab):
    msg = make_message(has_sticker=True, sticker_is_animated=False, sticker_is_video=False)
    await process_user_message(msg, mock_config, AsyncMock(), mock_colab)
    
    assert any("sticker_" in tag for tag in mock_config.stickers)


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media")
async def test_process_user_message_unexpected_error(mock_resolve, mock_config, mock_colab):
    msg = make_message(text="Hello")
    mock_resolve.side_effect = ValueError("Something broke badly")
    
    await process_user_message(msg, mock_config, AsyncMock(), mock_colab)
    msg.answer.assert_called_with("An error occurred while processing your message. Please try again.")


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
@patch("bot.handlers._stream_response")
async def test_process_user_message_cold_start_on_llm(mock_stream, mock_resolve, mock_config, mock_colab):
    msg = make_message(text="hello")
    mock_colab.call_llm.side_effect = [ColabColdStartError(), "OK"]
    
    await process_user_message(msg, mock_config, AsyncMock(), mock_colab)
    assert mock_colab.session.cold_start.call_count == 1
    assert mock_colab.call_llm.call_count == 2
