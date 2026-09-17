"""Monkeypatch ``langchain_openai`` to surface local-LLM reasoning tokens.

vLLM serving Qwen3-style models puts the chain-of-thought in a
``reasoning`` field (message for non-stream, ``delta.reasoning`` for stream);
older/vLLM configs use ``reasoning_content``. ``langchain_openai`` only
parses the official OpenAI spec, so both are dropped. This patch wraps the
two internal converters in ``langchain_openai.chat_models.base`` and copies
either field into ``additional_kwargs["reasoning_content"]`` — the key that
``langchain_core`` natively turns into a reasoning content block and that
message merging concatenates chunk by chunk.

Install once at startup (idempotent):

    from code_graphrag.llm.vllm_reasoning_patch import install_vllm_reasoning_patch

    install_vllm_reasoning_patch()
"""

from __future__ import annotations

from typing import Any

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

_TARGET_KEYS = ("reasoning", "reasoning_content")


def _take_reasoning(message: Any, _dict: Any) -> str | None:
    if not isinstance(_dict, dict):
        return None
    for key in _TARGET_KEYS:
        value = _dict.get(key)
        if isinstance(value, str) and value:
            message.additional_kwargs["reasoning_content"] = value
            return value
    return None


def install_vllm_reasoning_patch() -> bool:
    """Patch langchain_openai converters to keep reasoning in additional_kwargs.

    Returns True if the patch was (re)installed, False if already active.
    """
    from langchain_openai.chat_models import base as _base

    if getattr(_base, "_cg_reasoning_patch", False):
        return False

    orig_convert_dict = _base._convert_dict_to_message
    orig_convert_delta = _base._convert_delta_to_message_chunk

    def patched_convert_dict(_dict: Any) -> Any:
        message = orig_convert_dict(_dict)
        _take_reasoning(message, _dict)
        return message

    def patched_convert_delta(_dict: Any, default_class: Any) -> Any:
        chunk = orig_convert_delta(_dict, default_class)
        _take_reasoning(chunk, _dict)
        return chunk

    _base._convert_dict_to_message = patched_convert_dict
    _base._convert_delta_to_message_chunk = patched_convert_delta
    _base._cg_reasoning_patch = True
    logger.info("langchain_openai reasoning patch installed")
    return True
