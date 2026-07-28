"""
test_handlers_extended.py — Extended unit tests for message handlers.

This module covers core commands, unsupported media handling, and the
happy-path / error branches of the main message processing pipeline.
Block comments denote the specific handlers or pipelines being tested.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.fsm.context import FSMContext
from bot.handlers import (
    cmd_start,
    cmd_reset,
    handle_unsupported,
    process_user_message,
    _get_state,
    _MSG_UNSUPPORTED,
    _MSG_SIZE_LIMIT,
    _MSG_CONVERSION_ERROR,
    _MSG_QUOTA_ERROR,
    _MSG_EXEC_ERROR,
    _MSG_COLD_START,
)
from bot.colab_api import ColabColdStartError, ColabQuotaError, ColabExecError
from bot.converter import ConversionError
from bot.media import MediaSizeLimitError
from tests.conftest import make_message

# Fixtures

@pytest.fixture
def fsm_context():
    ctx = AsyncMock(spec=FSMContext)
    return ctx


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.bot.username = "testbot"
    cfg.bot.max_history = 5
    cfg.stickers = {}
    cfg.formatting = {}
    return cfg


@pytest.fixture
def mock_bot():
    return AsyncMock()


@pytest.fixture
def mock_colab():
    client = AsyncMock()
    client.session.cold_start = AsyncMock()
    return client

# Command Handlers Tests

@pytest.mark.asyncio
async def test_cmd_start(fsm_context):
    msg = make_message(chat_type="private")
    chat_id = msg.chat.id
    
    state = _get_state(chat_id)
    state["history"].append("old")
    state["is_typing"] = True
    
    await cmd_start(msg, fsm_context)
    
    fsm_context.clear.assert_called_once()
    msg.answer.assert_called_once_with("Привет!")
    assert state["history"] == []
    assert state["is_typing"] is False
    assert state["interrupt_signal"] is False


@pytest.mark.asyncio
async def test_cmd_reset():
    msg = make_message(chat_type="private")
    chat_id = msg.chat.id
    
    state = _get_state(chat_id)
    state["history"].append("old")
    
    await cmd_reset(msg)
    
    msg.answer.assert_called_once_with("История очищена.")
    assert state["history"] == []
    assert state["is_typing"] is False

# Unsupported Media Tests

@pytest.mark.asyncio
async def test_handle_unsupported(mock_config):
    msg = make_message(chat_type="private")
    await handle_unsupported(msg, mock_config)
    msg.answer.assert_called_once_with(_MSG_UNSUPPORTED)

# Message Processing Tests

@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
@patch("bot.handlers._stream_response")
async def test_process_user_message_happy_path(mock_stream, mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    chat_id = msg.chat.id
    
    mock_colab.call_llm.return_value = "Response text"
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    state = _get_state(chat_id)
    assert len(state["history"]) == 2
    assert state["history"][0]["role"] == "user"
    assert state["history"][1]["role"] == "assistant"
    
    mock_colab.call_llm.assert_called_once()
    mock_stream.assert_called_once()


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media")
async def test_process_user_message_size_limit(mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    mock_resolve.side_effect = MediaSizeLimitError("Too large")
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    msg.answer.assert_called_once_with(_MSG_SIZE_LIMIT)


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media")
async def test_process_user_message_conversion_error(mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    mock_resolve.side_effect = ConversionError("Bad format")
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    msg.answer.assert_called_once_with(_MSG_CONVERSION_ERROR)


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
async def test_process_user_message_quota_error(mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    mock_colab.call_llm.side_effect = ColabQuotaError("Quota exceeded")
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    msg.answer.assert_called_with(_MSG_QUOTA_ERROR)


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media", return_value={})
async def test_process_user_message_exec_error(mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    mock_colab.call_llm.side_effect = ColabExecError("Crash")
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    msg.answer.assert_called_with(_MSG_EXEC_ERROR)


@pytest.mark.asyncio
@patch("bot.handlers._resolve_media")
@patch("bot.handlers._stream_response")
async def test_process_user_message_cold_start(mock_stream, mock_resolve, mock_config, mock_bot, mock_colab):
    msg = make_message(text="Hello", chat_type="private")
    
    mock_colab.call_llm.side_effect = [ColabColdStartError("Dead"), "New Response"]
    mock_resolve.return_value = {}
    
    await process_user_message(msg, mock_config, mock_bot, mock_colab)
    
    assert msg.answer.call_args_list[0][0][0] == _MSG_COLD_START
    mock_colab.session.cold_start.assert_called_once()
    assert mock_colab.call_llm.call_count == 2
    mock_stream.assert_called_once()
