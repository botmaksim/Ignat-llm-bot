"""
handlers.py — aiogram routers, FSM, media dispatch, and Colab cold-start UX.

Request tracing
---------------
Each incoming message sets a unique ``request_id`` in the
:data:`~logging_config.request_id` ContextVar before any other work.
This ID appears in every log line produced during that request — in all
modules — without threading the ID through function arguments.

Handler flow
------------
1. ``process_user_message`` sets request_id and checks _should_respond.
2. If the bot is already generating, an interrupt signal is raised.
3. Media is downloaded and converted by :func:`~media.build_media_payload`.
4. The payload is passed to the appropriate Colab method.
5. On :class:`~colab_api.ColabColdStartError`, the user is notified and the
   session is re-provisioned transparently before retrying.
6. The LLM response is split on ``[PAUSE]`` tokens and streamed to the user
   with typing delays and optional stickers.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, Optional

from aiogram import Bot, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.colab_api import ColabAPIClient, ColabColdStartError, ColabExecError, ColabQuotaError
from bot.config import Config, save_stickers
from bot.logging_config import request_id as _request_id
from bot.media import MediaKind, MediaPayload, MediaSizeLimitError, build_media_payload
from bot.parser import parse_incoming_message, parse_llm_formatting

logger = logging.getLogger(__name__)

router = Router()

# In-memory chat state store

ChatState = dict[str, Any]
_chat_states: dict[int, ChatState] = {}

_CHAT_STATE_TTL_S: float = 24 * 3600.0


# User-facing messages

_MSG_COLD_START = (
    "Restarting neural networks — this will take a couple of minutes. "
    "Your message is saved and will be processed automatically!"
)
_MSG_COLD_START_DONE   = "Neural networks online. Processing your message…"
_MSG_QUOTA_ERROR       = (
    "Google Colab GPU quota has been exhausted. "
    "Please try again later — quotas reset daily."
)
_MSG_EXEC_ERROR        = "An error occurred while processing your message. Please try again."
_MSG_SIZE_LIMIT        = (
    "The file you sent is too large for the neural network to process. "
    "Please send a smaller file."
)
_MSG_CONVERSION_ERROR  = (
    "Could not convert your media file to the required format. "
    "Please try a different file."
)
_MSG_UNSUPPORTED       = (
    "у меня за мкадом такая фигня не грузит, скинь что-нибудь другое"
)


# Chat state helpers

def _get_state(chat_id: int) -> ChatState:
    if chat_id not in _chat_states:
        _chat_states[chat_id] = {
            "history":          [],
            "is_typing":        False,
            "interrupt_signal": False,
            "last_active":      time.monotonic(),
        }
    return _chat_states[chat_id]


def _touch(state: ChatState) -> None:
    state["last_active"] = time.monotonic()


def evict_stale_states() -> int:
    now = time.monotonic()
    stale = [
        cid for cid, st in _chat_states.items()
        if now - st["last_active"] > _CHAT_STATE_TTL_S
    ]
    for cid in stale:
        del _chat_states[cid]
    if stale:
        logger.info("Evicted stale chat states", extra={"count": len(stale)})
    return len(stale)

# Message filtering

def _should_respond(message: Message, config: Config) -> bool:
    return True

# Media resolution

async def _resolve_media(
    message: Message,
    bot: Bot,
    colab: ColabAPIClient,
    chat_id: int,
) -> dict[str, str]:
    from bot.converter import ConversionError

    payload: Optional[MediaPayload]
    try:
        payload = await build_media_payload(message, bot)
    except ConversionError as exc:
        logger.error(
            "Media conversion failed",
            extra={"chat_id": chat_id, "error": str(exc)},
        )
        raise

    if payload is None:
        return {}

    logger.debug(
        "Media payload built",
        extra={
            "chat_id":    chat_id,
            "kind":       payload.kind.name,
            "size_bytes": payload.size_bytes,
        },
    )

    if payload.kind == MediaKind.JPEG:
        return {"image_description": await colab.describe_image(payload)}

    if payload.kind in (MediaKind.OGG, MediaKind.MP3):
        return {"voice_transcript": await colab.transcribe_audio(payload)}

    if payload.kind == MediaKind.GIF:
        return {"gif_description": await colab.describe_gif(payload)}

    return {}

# Cold-start orchestration

async def _do_cold_start(
    message: Message,
    colab: ColabAPIClient,
    chat_id: int,
) -> bool:
    logger.info("Initiating cold start", extra={"chat_id": chat_id})
    try:
        await message.answer(_MSG_COLD_START)
    except Exception:
        pass

    try:
        await colab.session.cold_start()
        logger.info("Cold start succeeded", extra={"chat_id": chat_id})
        try:
            await message.answer(_MSG_COLD_START_DONE)
        except Exception:
            pass
        return True

    except ColabQuotaError as exc:
        logger.error(
            "GPU quota exhausted during cold start",
            extra={"chat_id": chat_id, "error": str(exc)},
        )
        await message.answer(_MSG_QUOTA_ERROR)
        return False

    except ColabExecError as exc:
        logger.error(
            "Cold start provisioning failed",
            extra={"chat_id": chat_id, "error": str(exc)},
        )
        await message.answer(_MSG_EXEC_ERROR)
        return False


# Response streaming

async def _stream_response(
    message: Message,
    llm_raw: str,
    state: ChatState,
    config: Config,
    chat_id: int,
) -> None:
    """
    Split the LLM response on ``[PAUSE]`` tokens and send each chunk with a
    typing-speed delay.  ``[STICKER:tag]`` triggers a sticker send.  An
    ``interrupt_signal`` in *state* aborts output immediately.
    """
    chunks = re.split(r"\[PAUSE(?::\d+)?\]", llm_raw, flags=re.IGNORECASE)

    for chunk in chunks:
        if not chunk.strip():
            continue

        if state["interrupt_signal"]:
            logger.debug("Output interrupted", extra={"chat_id": chat_id})
            return

        sticker_match = re.search(
            r"\[STICKER:([a-zA-Z0-9_]+)\]", chunk, flags=re.IGNORECASE
        )
        sticker_tag: Optional[str] = sticker_match.group(1) if sticker_match else None

        clean = re.sub(r"\[STICKER:[^\]]+\]", "", chunk, flags=re.IGNORECASE)
        clean = re.sub(r"\[EMOTION:[^\]]+\]", "", clean, flags=re.IGNORECASE).strip()

        formatted = parse_llm_formatting(clean, config.formatting)

        if formatted:
            await message.answer(formatted)
            delay = len(formatted) * config.bot.typing_speed
            elapsed = 0.0
            while elapsed < delay:
                if state["interrupt_signal"]:
                    logger.debug(
                        "Output interrupted during delay", extra={"chat_id": chat_id}
                    )
                    return
                await asyncio.sleep(0.1)
                elapsed += 0.1

        if sticker_tag and not state["interrupt_signal"]:
            sticker_info = config.stickers.get(sticker_tag)
            if sticker_info:
                file_id = sticker_info.get("file_id", "")
                if file_id and not file_id.endswith("..."):
                    try:
                        await message.answer_sticker(file_id)
                    except Exception as exc:
                        logger.warning(
                            "Failed to send sticker",
                            extra={"chat_id": chat_id, "tag": sticker_tag, "error": str(exc)},
                        )
                else:
                    logger.debug("Sticker has no valid file_id", extra={"tag": sticker_tag})
            else:
                logger.debug("Unknown sticker tag", extra={"tag": sticker_tag})


# Command handlers

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    chat_state = _get_state(message.chat.id)
    chat_state["history"].clear()
    chat_state["is_typing"] = False
    chat_state["interrupt_signal"] = False
    _touch(chat_state)
    await message.answer("Привет!")


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    chat_state = _get_state(message.chat.id)
    chat_state["history"].clear()
    chat_state["is_typing"] = False
    chat_state["interrupt_signal"] = False
    _touch(chat_state)
    await message.answer("История очищена.")

# Unsupported media

@router.message(F.video | F.document | F.video_note | F.contact | F.location | F.poll)
async def handle_unsupported(message: Message, config: Config) -> None:
    if not _should_respond(message, config):
        return
    await message.answer(_MSG_UNSUPPORTED)

# Main message handler

@router.message(F.text | F.photo | F.voice | F.audio | F.sticker | F.animation)
async def process_user_message(
    message: Message,
    config: Config,
    bot: Bot,
    colab: ColabAPIClient,
) -> None:
    from bot.converter import ConversionError

    if not _should_respond(message, config):
        return

    chat_id = message.chat.id

    req_id = uuid.uuid4().hex[:12]
    _request_id.set(req_id)

    state = _get_state(chat_id)
    _touch(state)

    logger.info(
        "Message received",
        extra={
            "chat_id":  chat_id,
            "chat_type": message.chat.type,
            "has_text": bool(message.text),
            "has_media": not bool(message.text),
        },
    )

    if state["is_typing"]:
        logger.info(
            "Interrupting current generation",
            extra={"chat_id": chat_id},
        )
        state["interrupt_signal"] = True
        for _ in range(50):
            if not state["is_typing"]:
                break
            await asyncio.sleep(0.1)

    if (
        message.sticker
        and not message.sticker.is_animated
        and not message.sticker.is_video
    ):
        sticker = message.sticker
        existing_ids = {v.get("file_id") for v in config.stickers.values()}
        if sticker.file_id not in existing_ids:
            tag = f"sticker_{sticker.file_unique_id[:8]}"
            if tag not in config.stickers:
                config.stickers[tag] = {
                    "file_id":     sticker.file_id,
                    "description": sticker.emoji or "unknown sticker",
                }
                save_stickers(config.stickers)
                logger.info(
                    "New sticker registered",
                    extra={"chat_id": chat_id, "tag": tag},
                )

    state["is_typing"] = True
    state["interrupt_signal"] = False

    try:
        try:
            media_kwargs = await _resolve_media(message, bot, colab, chat_id)
        except MediaSizeLimitError as exc:
            logger.warning(
                "Media exceeds size limit",
                extra={"chat_id": chat_id, "error": str(exc)},
            )
            await message.answer(_MSG_SIZE_LIMIT)
            return
        except ConversionError as exc:
            logger.error(
                "Media conversion error",
                extra={"chat_id": chat_id, "error": str(exc)},
            )
            await message.answer(_MSG_CONVERSION_ERROR)
            return
        except ColabColdStartError:
            ok = await _do_cold_start(message, colab, chat_id)
            if not ok:
                return
            media_kwargs = await _resolve_media(message, bot, colab, chat_id)
        except Exception as exc:
            logger.error(
                "Unexpected error resolving media",
                extra={"chat_id": chat_id, "error": str(exc)},
                exc_info=True,
            )
            media_kwargs = {}

        user_turn = parse_incoming_message(
            message,
            stickers_map=config.stickers,
            **media_kwargs,
        )
        state["history"].append({"role": "user", "content": user_turn})

        max_h = config.bot.max_history
        if len(state["history"]) > max_h:
            state["history"] = state["history"][-max_h:]

        try:
            llm_raw = await colab.call_llm(state["history"])
        except ColabColdStartError:
            ok = await _do_cold_start(message, colab, chat_id)
            if not ok:
                return
            llm_raw = await colab.call_llm(state["history"])

        logger.debug(
            "LLM response received",
            extra={"chat_id": chat_id, "response_len": len(llm_raw)},
        )

        await _stream_response(message, llm_raw, state, config, chat_id)

        if not state["interrupt_signal"]:
            state["history"].append({"role": "assistant", "content": llm_raw})

    except ColabQuotaError:
        logger.error("GPU quota exhausted", extra={"chat_id": chat_id}, exc_info=True)
        if message.chat.type == "private":
            await message.answer(_MSG_QUOTA_ERROR)

    except ColabExecError as exc:
        logger.error(
            "Colab exec error",
            extra={"chat_id": chat_id, "error": str(exc)},
            exc_info=True,
        )
        if message.chat.type == "private":
            await message.answer(_MSG_EXEC_ERROR)

    except Exception as exc:
        logger.error(
            "Unhandled exception in message handler",
            extra={"chat_id": chat_id, "error": str(exc)},
            exc_info=True,
        )
        if message.chat.type == "private":
            await message.answer(_MSG_EXEC_ERROR)

    finally:
        state["is_typing"] = False
