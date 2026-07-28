"""
main.py — Application entry point.

Bootstraps logging, loads config, initialises the Colab CLI client,
configures the aiogram Dispatcher with dependency injection, and starts
long-polling.

A background coroutine periodically evicts stale in-memory chat states to
prevent unbounded memory growth in high-traffic deployments.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from bot.colab_api import ColabAPIClient, ColabSession
from bot.config import build_sticker_prompt, load_config
from bot.handlers import evict_stale_states, router as main_router
from bot.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Interval between in-memory chat-state eviction passes.
_EVICT_INTERVAL_S = 3600.0


async def _state_eviction_loop() -> None:
    """Background task: periodically evict stale in-memory chat states."""
    while True:
        await asyncio.sleep(_EVICT_INTERVAL_S)
        evict_stale_states()


async def main() -> None:
    # Load configuration
    try:
        config = load_config()
    except ValueError as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    configure_logging(level=config.bot.log_level, fmt=config.bot.log_format)

    logger.info(
        "Bot starting",
        extra={
            "log_level":    config.bot.log_level,
            "log_format":   config.bot.log_format,
            "session_name": config.colab.session_name,
            "gpu_type":     config.colab.gpu_type,
            "exec_timeout": config.colab.exec_timeout,
        },
    )

    sticker_hint = build_sticker_prompt(config.stickers)
    if sticker_hint:
        logger.info("Loaded stickers", extra={"stickers": sticker_hint})

    # Initialise Colab CLI client
    session = ColabSession(
        session_name=config.colab.session_name,
        gpu_type=config.colab.gpu_type,
        colab_bin=config.colab.colab_bin,
    )
    colab = ColabAPIClient(
        session=session,
        exec_timeout=config.colab.exec_timeout,
    )
    await colab.start()

    # Set up aiogram
    bot = Bot(token=config.bot.token, parse_mode=ParseMode.HTML)
    dp  = Dispatcher(storage=MemoryStorage())

    # Inject shared dependencies via the Dispatcher DI container.
    dp["config"] = config
    dp["colab"]  = colab
    dp["bot"]    = bot

    dp.include_router(main_router)

    # Start background tasks
    eviction_task = asyncio.create_task(
        _state_eviction_loop(), name="chat_state_eviction"
    )

    logger.info("Long-polling started")

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        eviction_task.cancel()
        await colab.stop()
        await bot.session.close()
        logger.info("Bot stopped gracefully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
