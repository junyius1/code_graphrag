"""Capture + stream local-LLM reasoning (chain-of-thought) from GraphRAG queries.

GraphRAG's LiteLLM completion only forwards ``delta.content`` to callers; the
reasoning tokens (``delta.reasoning_content`` on vLLM/Qwen3-style servers) are
dropped. This module installs a thin interceptor on the completion factory so
that, while a search is running, every reasoning token is written to a
process-wide sink that the CLI / caller drains and prints in real time.

The interceptor passes all chunks through unmodified (only the local ``content``
copy is removed from reasoning-bearing chunks), so normal answer streaming is
unaffected.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator
from typing import Any

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

_installed = False
_lock = threading.Lock()
_sink: list[str] = []  # streaming buffer, flushed to stderr
_collected: list[str] = []  # full capture for SearchResult.reasoning
_sink_enabled = False
_last_emit = 0.0


def _emit(token: str) -> None:
    global _last_emit
    with _lock:
        _collected.append(token)
        if not _sink_enabled:
            return
        _sink.append(token)
        now = time.monotonic()
        if now - _last_emit < 0.05:
            return  # keep buffering; a later token or exit will flush
        buf = "".join(_sink)
        _sink.clear()
        _last_emit = now
    if buf:
        import sys

        sys.stderr.write(f"\x1b[2m{buf}\x1b[0m")
        sys.stderr.flush()


def _flush_tail() -> None:
    with _lock:
        if not _sink:
            return
        buf = "".join(_sink)
        _sink.clear()
    import sys

    sys.stderr.write(f"\x1b[2m{buf}\x1b[0m")
    sys.stderr.flush()


@contextlib.contextmanager
def capture_reasoning(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager: live-print + collect reasoning tokens while active."""
    _install()
    global _sink_enabled
    with _lock:
        _sink_enabled = enabled
        _sink.clear()
        _collected.clear()
        _last_emit = 0.0
    try:
        yield
    finally:
        _flush_tail()
        with _lock:
            _sink_enabled = False
            _sink.clear()


def drain() -> str:
    """Return (and clear) the full reasoning captured during the last capture."""
    with _lock:
        buf = "".join(_collected)
        _collected.clear()
    return buf


def _extract(obj: Any, field: str) -> str | None:
    """Pull ``reasoning_content``/``reasoning`` from a delta or message obj."""
    choices = getattr(obj, "choices", None)
    if not choices:
        return None
    target = getattr(choices[0], field, None)
    if target is None:
        return None
    value = getattr(target, "reasoning_content", None)
    if value is None and hasattr(target, "model_extra"):
        extra = target.model_extra or {}
        value = extra.get("reasoning_content") or extra.get("reasoning")
    return value or None


def _intercept_stream(gen: Any) -> Any:
    """Wrap an async chunk iterator: tee reasoning tokens into the sink."""

    async def wrapped() -> Any:
        async for chunk in gen:
            reasoning = _extract(chunk, "delta")
            if reasoning:
                _emit(reasoning)
            yield chunk

    return wrapped()


def _install() -> None:
    """Patch LiteLLMCompletion (sync+async) to tee reasoning tokens. Idempotent."""
    global _installed
    if _installed:
        return
    try:
        from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
    except Exception:  # noqa: BLE001
        return

    if getattr(LiteLLMCompletion, "_cg_reasoning_patch", False):
        _installed = True
        return

    orig_sync = LiteLLMCompletion.completion
    orig_async = LiteLLMCompletion.completion_async

    def completion(self, /, **kwargs: Any) -> Any:
        out = orig_sync(self, **kwargs)
        if kwargs.get("stream"):
            return _intercept_stream(out)
        reasoning = _extract(out, "message")
        if reasoning:
            _emit(reasoning)
        return out

    async def completion_async(self, /, **kwargs: Any) -> Any:
        out = await orig_async(self, **kwargs)
        if kwargs.get("stream"):
            return _intercept_stream(out)
        reasoning = _extract(out, "message")
        if reasoning:
            _emit(reasoning)
        return out

    LiteLLMCompletion.completion = completion
    LiteLLMCompletion.completion_async = completion_async
    LiteLLMCompletion._cg_reasoning_patch = True
    _installed = True
    logger.debug("reasoning interceptor installed")
