"""LangChain model factories for the LLM-driven parts of Code GraphRAG.

LangChain v1 (verified against 3partylibs/langchain 1.3.x):
- ``from langchain.chat_models import init_chat_model`` (NOT the package root).
- ``from langchain.embeddings import init_embeddings``.
- Provider-agnostic; kwargs (api_key, base_url) pass through to partner classes.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from code_graphrag.config.models import EmbeddingConfig, LLMConfig, LLMProvider
from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


class _MockChatModel(BaseChatModel):
    """Deterministic offline stand-in so LLM paths are testable without a key.

    Supports plain completion *and* a minimal tool-calling loop: if the last
    message is a tool result it returns a final text answer, otherwise it emits
    one tool call (so Deep Agents agents can run fully offline). Used only when
    ``llm.provider=mock``.
    """

    model_name: str = "mock"

    @property
    def _llm_type(self) -> str:
        return "code-graphrag-mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage):
            # We already ran a tool; return a final grounded answer.
            text = (
                "Mock answer (offline): I queried the code graph and inspected "
                "the relevant source. "
                f"Tool result: {str(last.content)[:200]}"
            )
            msg = AIMessage(content=text)
        else:
            # Emit a single deterministic tool call.
            import re

            human = next(
                (m for m in reversed(messages) if getattr(m, "type", None) == "human"),
                None,
            )
            q = human.content if human and isinstance(getattr(human, "content", ""), str) else ""
            # Pick a code-like token from the question: underscore/dot first,
            # then a non-stopword camelCase/snake token, then any non-stopword.
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]+", q)
            stop = {
                "what",
                "does",
                "the",
                "do",
                "a",
                "an",
                "is",
                "are",
                "to",
                "of",
                "in",
                "on",
                "how",
                "why",
                "when",
                "where",
                "which",
                "function",
                "class",
                "method",
                "module",
                "defined",
                "implemented",
                "work",
                "used",
                "explain",
            }
            symbol = next((t for t in tokens if "_" in t or "." in t), "")
            if not symbol:
                symbol = next(
                    (
                        t
                        for t in tokens
                        if t.lower() not in stop and (re.search(r"[A-Z]", t) or "_" in t)
                    ),
                    "",
                )
            if not symbol:
                symbol = next(
                    (t for t in tokens if t.lower() not in stop),
                    "handle",
                )
            tool_name = "where_is"
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"symbol": symbol},
                        "id": "call_mock_1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        """No-op: the mock ignores tool schemas and emits its own tool call."""
        return self

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model_name}


def create_chat_model(cfg: LLMConfig) -> BaseChatModel:
    """Create a chat model from LLMConfig (env-var keys are respected)."""
    if cfg.provider is LLMProvider.MOCK:
        return _MockChatModel()
    from langchain.chat_models import init_chat_model

    provider = (
        "openai"
        if cfg.provider in (LLMProvider.OPENAI, LLMProvider.OPENAI_COMPATIBLE)
        else cfg.provider.value.replace("_", "-")
    )
    kwargs: dict = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.max_tokens:
        kwargs["max_tokens"] = cfg.max_tokens
    logger.debug("Creating chat model %s:%s", provider, cfg.model)
    return init_chat_model(f"{provider}:{cfg.model}", temperature=cfg.temperature, **kwargs)


def create_embeddings(cfg: EmbeddingConfig) -> Embeddings:
    """Create an embeddings model from EmbeddingConfig."""
    from langchain.embeddings import init_embeddings

    kwargs: dict = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.dimensions:
        kwargs["dimensions"] = cfg.dimensions
    provider = cfg.provider or "openai"
    return init_embeddings(f"{provider}:{cfg.model}", **kwargs)
