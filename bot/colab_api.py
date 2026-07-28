"""
colab_api.py — Google Colab CLI client for the Ignat orchestrator bot.

Architecture
------------
The bot communicates with a running Colab session via the official
``google-colab-cli`` tool (subprocess).  Every inference call follows
a three-step protocol:

  1. **Upload** a JSON payload to ``/tmp/ignat_payload.json`` in the
     remote kernel (via ``colab exec`` piped stdin).
  2. **Execute** the inference script (also via ``colab exec`` piped stdin).
  3. **Parse** the JSON result printed to stdout by the script.

Accepted media formats
----------------------
After :mod:`media` + :mod:`converter` normalise all downloads, the Colab
inference scripts receive **exactly two binary formats**:

  - **JPEG** for all images (``describe_image``).
  - **GIF**  for all animations and video stickers (``describe_gif``).
  - **OGG / MP3** for audio (``transcribe_audio``).

Cold-start flow
---------------
When a session is dead (12-hour Colab limit), :meth:`ColabSession.ensure_ready`
raises :class:`ColabColdStartError`.  ``handlers.py`` catches it, notifies
the user, calls :meth:`ColabSession.cold_start`, and retries the original
request.  An :class:`asyncio.Lock` serialises concurrent cold-start attempts.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from bot.media import MediaKind, MediaPayload

logger = logging.getLogger(__name__)

# Exceptions

class ColabError(Exception):
    pass

class ColabColdStartError(ColabError):
    pass

class ColabQuotaError(ColabError):
    pass

class ColabExecError(ColabError):
    pass



"""
Inference scripts executed on the remote Colab kernel

Each script:
  1. Opens /tmp/ignat_payload.json (written by _upload_payload).
  2. Calls a model function already defined in the kernel.
  3. Prints exactly one JSON line to stdout.

The model functions (generate, describe_image, transcribe_audio, describe_gif)
must be defined by initialisation cells in the notebook that runs at session
startup.
"""

_SCRIPT_LLM = """\
import json
with open('/tmp/ignat_payload.json') as f:
    data = json.load(f)
result = generate(data['history'])
print(json.dumps({'response': result}), flush=True)
"""

_SCRIPT_IMAGE = """\
import json, base64
with open('/tmp/ignat_payload.json') as f:
    data = json.load(f)
img_bytes = base64.b64decode(data['image_b64'])  # always JPEG
result = describe_image(img_bytes)
print(json.dumps({'description': result}), flush=True)
"""

_SCRIPT_AUDIO = """\
import json, base64
with open('/tmp/ignat_payload.json') as f:
    data = json.load(f)
audio_bytes = base64.b64decode(data['audio_b64'])
audio_fmt   = data['audio_fmt']  # 'ogg' or 'mp3'
result = transcribe_audio(audio_bytes, fmt=audio_fmt)
print(json.dumps({'text': result}), flush=True)
"""

_SCRIPT_GIF = """\
import json, base64
with open('/tmp/ignat_payload.json') as f:
    data = json.load(f)
gif_bytes = base64.b64decode(data['gif_b64'])  # always GIF
result = describe_gif(gif_bytes)
print(json.dumps({'description': result}), flush=True)
"""

# ColabSession

@dataclass
class ColabSession:
    session_name: str
    gpu_type:     str
    colab_bin:    str

    _ready:   bool         = field(default=False, init=False, repr=False)
    _warming: bool         = field(default=False, init=False, repr=False)
    _lock:    asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    # Low-level subprocess helper

    async def _run(
        self,
        *args: str,
        stdin_data: Optional[bytes] = None,
        timeout: float = 60.0,
    ) -> tuple[int, str, str]:
        cmd = [self.colab_bin, *args]
        logger.debug("CLI command", extra={"cmd": " ".join(cmd)})

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_data),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ColabExecError(
                f"CLI command timed out after {timeout:.0f}s: {' '.join(cmd)}"
            )

        return (
            proc.returncode or 0,
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
        )

    # Status

    async def status(self) -> str:
        try:
            rc, out, _ = await self._run("status", "-s", self.session_name, timeout=20.0)
        except ColabExecError:
            return "unknown"

        if rc != 0:
            return "stopped"

        out_lower = out.lower()
        if "running" in out_lower or "active" in out_lower:
            return "running"
        return "stopped"

    # Readiness gate

    async def ensure_ready(self) -> None:
        if self._ready:
            return

        if self._warming:
            logger.debug("Cold start in progress — waiting")
            async with self._lock:
                pass
            if not self._ready:
                raise ColabExecError("Cold start completed with an error.")
            return

        current = await self.status()
        if current == "running":
            self._ready = True
            logger.info("Session ready", extra={"session": self.session_name})
            return

        raise ColabColdStartError(
            f"Session '{self.session_name}' is not running (status={current!r}). "
            "Cold start required."
        )

    # Cold start

    async def cold_start(self) -> None:
        async with self._lock:
            if self._ready:
                logger.debug("Session already ready after lock — skipping")
                return

            self._warming = True
            logger.info(
                "Cold start initiated",
                extra={"session": self.session_name, "gpu": self.gpu_type},
            )
            t0 = time.monotonic()

            try:
                rc, out, err = await self._run(
                    "new", "-s", self.session_name, "--gpu", self.gpu_type,
                    timeout=300.0,
                )
            except ColabExecError as exc:
                self._ready = False
                raise ColabExecError(f"Cold start subprocess timed out: {exc}") from exc

            elapsed = time.monotonic() - t0

            if rc != 0:
                combined = (out + err).lower()
                if any(kw in combined for kw in ("quota", "limit exceeded", "no gpus")):
                    raise ColabQuotaError(
                        f"GPU quota exhausted. stderr: {err.strip()}"
                    )
                raise ColabExecError(
                    f"'colab new' exited {rc}. stderr: {err.strip()}"
                )

            self._ready = True
            logger.info(
                "Cold start complete",
                extra={"session": self.session_name, "elapsed_s": round(elapsed, 1)},
            )

        self._warming = False

    def invalidate(self) -> None:
        if self._ready:
            logger.warning("Session invalidated", extra={"session": self.session_name})
        self._ready = False

# ColabAPIClient

class ColabAPIClient:
    def __init__(
        self,
        session: ColabSession,
        exec_timeout: float = 120.0,
    ) -> None:
        self.session = session
        self.exec_timeout = exec_timeout

    async def start(self) -> None:
        status = await self.session.status()
        logger.info(
            "Colab session status at startup",
            extra={"session": self.session.session_name, "status": status},
        )
        if status == "running":
            self.session._ready = True

    async def stop(self) -> None:
        logger.info(
            "Bot stopping — Colab session left running",
            extra={"session": self.session.session_name},
        )

    # Internal helpers

    async def _upload_payload(self, payload: dict[str, Any], request_id: str) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False)
        write_script = (
            "import json\n"
            f"payload_str = {payload_json!r}\n"
            "with open('/tmp/ignat_payload.json', 'w', encoding='utf-8') as _f:\n"
            "    json.dump(json.loads(payload_str), _f)\n"
        )
        rc, _, err = await self.session._run(
            "exec", "-s", self.session.session_name,
            stdin_data=write_script.encode(),
            timeout=30.0,
        )
        if rc != 0:
            self.session.invalidate()
            raise ColabExecError(
                f"[{request_id}] Payload upload failed (rc={rc}): {err.strip()}"
            )

    async def _exec_script(self, script: str, request_id: str) -> str:
        rc, out, err = await self.session._run(
            "exec", "-s", self.session.session_name,
            stdin_data=script.encode(),
            timeout=self.exec_timeout,
        )

        if rc != 0:
            combined = (out + err).lower()
            dead_keywords = (
                "session not found", "not running", "expired",
                "disconnected", "runtime disconnected",
            )
            if any(kw in combined for kw in dead_keywords):
                self.session.invalidate()
                raise ColabColdStartError(
                    f"[{request_id}] Session died during exec — cold start required. "
                    f"stdout={out[:200]!r} stderr={err[:200]!r}"
                )
            raise ColabExecError(
                f"[{request_id}] exec exited rc={rc}. "
                f"stdout={out[:200]!r} stderr={err[:200]!r}"
            )

        return out

    @staticmethod
    def _parse_stdout(raw: str, key: str, fallback: str) -> str:
        for line in reversed(raw.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                with contextlib.suppress(json.JSONDecodeError):
                    data = json.loads(line)
                    return str(data.get(key, fallback))
        logger.warning(
            "No valid JSON found in exec stdout",
            extra={"stdout_preview": raw[:300]},
        )
        return raw.strip() or fallback

    @contextlib.contextmanager
    def _trace(self, operation: str, request_id: str):
        logger.info(
            "Inference request started",
            extra={"op": operation, "request_id": request_id},
        )
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            logger.info(
                "Inference request finished",
                extra={
                    "op":          operation,
                    "request_id":  request_id,
                    "elapsed_s":   round(elapsed, 3),
                },
            )

    async def _call(
        self,
        payload: dict[str, Any],
        script: str,
        operation: str,
    ) -> str:
        request_id = uuid.uuid4().hex[:8]

        with self._trace(operation, request_id):
            await self.session.ensure_ready()
            await self._upload_payload(payload, request_id)
            return await self._exec_script(script, request_id)

    # Public inference API

    async def call_llm(self, context_history: list[dict[str, Any]]) -> str:
        raw = await self._call(
            {"history": context_history},
            _SCRIPT_LLM,
            "call_llm",
        )
        return self._parse_stdout(raw, "response", "")

    async def describe_image(self, payload: MediaPayload) -> str:
        assert payload.kind == MediaKind.JPEG, (
            f"describe_image requires JPEG payload, got {payload.kind.name}. "
            "Ensure build_media_payload() has run before calling this method."
        )
        raw = await self._call(
            {"image_b64": base64.b64encode(payload.data).decode()},
            _SCRIPT_IMAGE,
            "describe_image",
        )
        return self._parse_stdout(raw, "description", "failed to describe image")

    async def transcribe_audio(self, payload: MediaPayload) -> str:
        assert payload.kind in (MediaKind.OGG, MediaKind.MP3), (
            f"transcribe_audio requires OGG or MP3, got {payload.kind.name}."
        )
        raw = await self._call(
            {
                "audio_b64": base64.b64encode(payload.data).decode(),
                "audio_fmt": payload.kind.extension,   # 'ogg' or 'mp3'
            },
            _SCRIPT_AUDIO,
            "transcribe_audio",
        )
        return self._parse_stdout(raw, "text", "failed to transcribe audio")

    async def describe_gif(self, payload: MediaPayload) -> str:
        assert payload.kind == MediaKind.GIF, (
            f"describe_gif requires GIF payload, got {payload.kind.name}. "
            "Ensure build_media_payload() has run before calling this method."
        )
        raw = await self._call(
            {"gif_b64": base64.b64encode(payload.data).decode()},
            _SCRIPT_GIF,
            "describe_gif",
        )
        return self._parse_stdout(raw, "description", "failed to describe animation")
