"""
tests/test_colab_api.py — Unit tests for colab_api.py

All Colab responses are simulated by mocking ``_exec_script`` to return the
JSON string that the Colab kernel would print.  ``_upload_payload`` is also
mocked (no-op) so subprocess calls are never made.

  - tests the actual JSON parsing / field extraction logic,
  - tests the kind assertion guards (JPEG, GIF, OGG/MP3),
  - tests the dead-session detection and invalidation path,
  - tests the cold-start concurrency lock,
  - avoids any real subprocess or network calls.

Covers:
- ColabSession.status: running/stopped/unknown detection
- ColabSession.ensure_ready: fast path, dead session, already-warming
- ColabSession.cold_start: success, quota, exec error, concurrency
- ColabSession.invalidate
- ColabAPIClient._parse_stdout: JSON extraction, fallback, empty output
- ColabAPIClient.call_llm: happy path, cold-start recovery
- ColabAPIClient.describe_image: happy path, wrong-kind assertion
- ColabAPIClient.transcribe_audio: OGG, MP3, wrong-kind assertion
- ColabAPIClient.describe_gif: happy path, wrong-kind assertion
- Dead-session invalidation via _exec_script side-effect
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.colab_api import (
    ColabAPIClient,
    ColabColdStartError,
    ColabExecError,
    ColabQuotaError,
    ColabSession,
)
from bot.media import MediaKind
from tests.conftest import make_payload, make_jpeg_bytes, make_gif_bytes

# Helpers

def _rc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return (returncode, stdout, stderr)


def _colab_json(key: str, value: str) -> str:
    return json.dumps({key: value})

# ColabSession.status

class TestColabSessionStatus:

    @pytest.mark.asyncio
    async def test_running_when_stdout_contains_running(self, session):
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(0, "Status: running"))):
            assert await session.status() == "running"

    @pytest.mark.asyncio
    async def test_running_when_stdout_contains_active(self, session):
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(0, "Session active"))):
            assert await session.status() == "running"

    @pytest.mark.asyncio
    async def test_stopped_on_nonzero_rc(self, session):
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(1, "", "not found"))):
            assert await session.status() == "stopped"

    @pytest.mark.asyncio
    async def test_stopped_on_ambiguous_output(self, session):
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(0, "Session idle"))):
            assert await session.status() == "stopped"

    @pytest.mark.asyncio
    async def test_unknown_on_exec_error(self, session):
        with patch.object(session, "_run", new=AsyncMock(side_effect=ColabExecError("timeout"))):
            assert await session.status() == "unknown"

# ColabSession.ensure_ready

class TestColabSessionEnsureReady:

    @pytest.mark.asyncio
    async def test_fast_path_when_already_ready(self, session):
        session._ready = True
        with patch.object(session, "_run", new=AsyncMock()) as mock_run:
            await session.ensure_ready()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_ready_when_running(self, session):
        session._ready = False
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(0, "running"))):
            await session.ensure_ready()
        assert session._ready is True

    @pytest.mark.asyncio
    async def test_raises_cold_start_error_when_stopped(self, session):
        session._ready = False
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(1))):
            with pytest.raises(ColabColdStartError):
                await session.ensure_ready()

    @pytest.mark.asyncio
    async def test_waits_for_ongoing_warm_up(self, session):
        session._ready = False
        session._warming = True

        async def finish_warmup():
            await asyncio.sleep(0.01)
            session._ready = True
            session._lock.release()

        await session._lock.acquire()
        asyncio.create_task(finish_warmup())
        await session.ensure_ready()
        assert session._ready is True

# ColabSession.cold_start

class TestColabSessionColdStart:

    @pytest.mark.asyncio
    async def test_success_marks_ready(self, session):
        session._ready = False
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(0, "Session created"))):
            await session.cold_start()
        assert session._ready is True

    @pytest.mark.asyncio
    async def test_quota_error_on_quota_keyword(self, session):
        session._ready = False
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(1, "", "GPU quota exceeded"))):
            with pytest.raises(ColabQuotaError):
                await session.cold_start()
        assert session._ready is False

    @pytest.mark.asyncio
    async def test_exec_error_on_generic_failure(self, session):
        session._ready = False
        with patch.object(session, "_run", new=AsyncMock(return_value=_rc(1, "", "internal error"))):
            with pytest.raises(ColabExecError):
                await session.cold_start()

    @pytest.mark.asyncio
    async def test_concurrent_cold_starts_serialised(self, session):
        session._ready = False
        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return (0, "ok", "")

        with patch.object(session, "_run", new=mock_run):
            await asyncio.gather(session.cold_start(), session.cold_start())

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_already_ready_under_lock(self, session):
        session._ready = False

        async def mock_run(*args, **kwargs):
            session._ready = True
            return (0, "ok", "")

        with patch.object(session, "_run", new=mock_run):
            await session.cold_start()

        assert session._ready is True

# ColabSession.invalidate

class TestColabSessionInvalidate:

    def test_invalidate_clears_ready(self, session):
        session._ready = True
        session.invalidate()
        assert session._ready is False

    def test_invalidate_idempotent_when_already_false(self, session):
        session._ready = False
        session.invalidate()
        assert session._ready is False

# ColabAPIClient._parse_stdout

class TestParseStdout:

    def test_extracts_key_from_last_json_line(self):
        raw = "debug print\n" + _colab_json("response", "hello") + "\n"
        assert ColabAPIClient._parse_stdout(raw, "response", "") == "hello"

    def test_uses_last_json_object_when_multiple_present(self):
        raw = _colab_json("response", "first") + "\n" + _colab_json("response", "second")
        assert ColabAPIClient._parse_stdout(raw, "response", "") == "second"

    def test_returns_raw_strip_when_no_json_found(self):
        raw = "plain text output"
        result = ColabAPIClient._parse_stdout(raw, "response", "FALLBACK")
        assert result == raw.strip()

    def test_returns_fallback_when_output_is_empty(self):
        assert ColabAPIClient._parse_stdout("", "response", "FALLBACK") == "FALLBACK"

    def test_returns_fallback_when_key_missing_in_json(self):
        raw = json.dumps({"other_key": "value"})
        result = ColabAPIClient._parse_stdout(raw, "response", "DEFAULT")
        assert result == "DEFAULT"

    def test_ignores_non_json_lines(self):
        raw = "Loading model…\nDone.\n" + _colab_json("text", "transcript") + "\n"
        assert ColabAPIClient._parse_stdout(raw, "text", "") == "transcript"

# ColabAPIClient.call_llm

class TestCallLlm:

    @pytest.mark.asyncio
    async def test_happy_path_returns_response(self, live_client):
        expected = "Hello from LLM! [PAUSE] How are you? [STICKER:happy]"
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("response", expected)),
            ):
                result = await live_client.call_llm([{"role": "user", "content": "hi"}])
        assert result == expected

    @pytest.mark.asyncio
    async def test_passes_full_history_in_payload(self, live_client):
        history = [
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user",      "content": "bye"},
        ]
        captured_payload: list[dict] = []

        async def capture_upload(payload, request_id):
            captured_payload.append(payload)

        with patch.object(live_client, "_upload_payload", side_effect=capture_upload):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("response", "ok")),
            ):
                await live_client.call_llm(history)

        assert captured_payload[0]["history"] == history

    @pytest.mark.asyncio
    async def test_cold_start_error_propagates(self, live_client):
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(side_effect=ColabColdStartError("session not found")),
            ):
                with pytest.raises(ColabColdStartError):
                    await live_client.call_llm([{"role": "user", "content": "ping"}])

    @pytest.mark.asyncio
    async def test_exec_error_propagates(self, live_client):
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(side_effect=ColabExecError("kernel died")),
            ):
                with pytest.raises(ColabExecError):
                    await live_client.call_llm([{"role": "user", "content": "ping"}])

# ColabAPIClient.describe_image

class TestDescribeImage:

    @pytest.mark.asyncio
    async def test_happy_path_returns_description(self, live_client, jpeg_payload):
        desc = "a sunset over the ocean with orange hues"
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("description", desc)),
            ):
                result = await live_client.describe_image(jpeg_payload)
        assert result == desc

    @pytest.mark.asyncio
    async def test_payload_contains_base64_encoded_data(self, live_client, jpeg_payload):
        import base64
        captured: list[dict] = []

        async def capture(payload, _req_id):
            captured.append(payload)

        with patch.object(live_client, "_upload_payload", side_effect=capture):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("description", "ok")),
            ):
                await live_client.describe_image(jpeg_payload)

        assert "image_b64" in captured[0]
        decoded = base64.b64decode(captured[0]["image_b64"])
        assert decoded == jpeg_payload.data

    @pytest.mark.asyncio
    async def test_wrong_kind_raises_assertion_error(self, live_client, ogg_payload):
        with pytest.raises(AssertionError, match="JPEG"):
            await live_client.describe_image(ogg_payload)

    @pytest.mark.asyncio
    async def test_gif_kind_raises_assertion_error(self, live_client, gif_payload):
        with pytest.raises(AssertionError, match="JPEG"):
            await live_client.describe_image(gif_payload)


# ColabAPIClient.transcribe_audio

class TestTranscribeAudio:

    @pytest.mark.asyncio
    async def test_ogg_happy_path(self, live_client, ogg_payload):
        transcript = "привет, как дела?"
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("text", transcript)),
            ):
                result = await live_client.transcribe_audio(ogg_payload)
        assert result == transcript

    @pytest.mark.asyncio
    async def test_mp3_happy_path(self, live_client, mp3_payload):
        transcript = "hello from mp3"
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("text", transcript)),
            ):
                result = await live_client.transcribe_audio(mp3_payload)
        assert result == transcript

    @pytest.mark.asyncio
    async def test_ogg_sends_correct_audio_fmt(self, live_client, ogg_payload):
        captured: list[dict] = []

        async def capture(payload, _req_id):
            captured.append(payload)

        with patch.object(live_client, "_upload_payload", side_effect=capture):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("text", "ok")),
            ):
                await live_client.transcribe_audio(ogg_payload)

        assert captured[0]["audio_fmt"] == "ogg"

    @pytest.mark.asyncio
    async def test_mp3_sends_correct_audio_fmt(self, live_client, mp3_payload):
        captured: list[dict] = []

        async def capture(payload, _req_id):
            captured.append(payload)

        with patch.object(live_client, "_upload_payload", side_effect=capture):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("text", "ok")),
            ):
                await live_client.transcribe_audio(mp3_payload)

        assert captured[0]["audio_fmt"] == "mp3"

    @pytest.mark.asyncio
    async def test_jpeg_kind_raises_assertion_error(self, live_client, jpeg_payload):
        with pytest.raises(AssertionError, match="OGG or MP3"):
            await live_client.transcribe_audio(jpeg_payload)

    @pytest.mark.asyncio
    async def test_gif_kind_raises_assertion_error(self, live_client, gif_payload):
        with pytest.raises(AssertionError, match="OGG or MP3"):
            await live_client.transcribe_audio(gif_payload)


# ColabAPIClient.describe_gif

class TestDescribeGif:

    @pytest.mark.asyncio
    async def test_happy_path_returns_description(self, live_client, gif_payload):
        desc = "a cartoon character laughing and pointing"
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("description", desc)),
            ):
                result = await live_client.describe_gif(gif_payload)
        assert result == desc

    @pytest.mark.asyncio
    async def test_payload_contains_base64_encoded_data(self, live_client, gif_payload):
        import base64
        captured: list[dict] = []

        async def capture(payload, _req_id):
            captured.append(payload)

        with patch.object(live_client, "_upload_payload", side_effect=capture):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(return_value=_colab_json("description", "ok")),
            ):
                await live_client.describe_gif(gif_payload)

        assert "gif_b64" in captured[0]
        decoded = base64.b64decode(captured[0]["gif_b64"])
        assert decoded == gif_payload.data

    @pytest.mark.asyncio
    async def test_jpeg_kind_raises_assertion_error(self, live_client, jpeg_payload):
        with pytest.raises(AssertionError, match="GIF"):
            await live_client.describe_gif(jpeg_payload)

    @pytest.mark.asyncio
    async def test_ogg_kind_raises_assertion_error(self, live_client, ogg_payload):
        with pytest.raises(AssertionError, match="GIF"):
            await live_client.describe_gif(ogg_payload)


# Dead-session detection

class TestDeadSessionDetection:

    @pytest.mark.asyncio
    async def test_exec_script_raises_cold_start_on_session_not_found(self, live_client):
        with patch.object(live_client, "_upload_payload", new=AsyncMock()):
            with patch.object(
                live_client, "_exec_script",
                new=AsyncMock(side_effect=ColabColdStartError("session not found")),
            ):
                with pytest.raises(ColabColdStartError):
                    await live_client.call_llm([{"role": "user", "content": "ping"}])

    @pytest.mark.asyncio
    async def test_upload_failure_invalidates_session(self, live_client):
        with patch.object(
            live_client.session, "_run",
            new=AsyncMock(return_value=(1, "", "session not found")),
        ):
            with pytest.raises(ColabExecError):
                await live_client.call_llm([{"role": "user", "content": "ping"}])
        assert live_client.session._ready is False
