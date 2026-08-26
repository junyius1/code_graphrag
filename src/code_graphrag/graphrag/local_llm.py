"""Local-LLM support: ``local:<path>`` model ids + offline embedding fallback.

A local server (e.g. vLLM) is addressed as ``local:<path>`` in user config
(``LOCAL_LLM_BASE_URL=http://localhost:8000/v1`` + model ``local:/mnt/D/models/Qwen3.8-27B``).
This module:

- normalizes ``local:<path>`` to the bare id the server advertises (``<path>``);
- probes the server's ``/v1/models`` + ``/v1/embeddings`` (best effort);
- registers a deterministic **local hash embedding** with GraphRAG's
  ``embedding_factory`` (type ``"local"``). GraphRAG's vector store has no
  semantic fallback and requires an embedding for index AND query, so when the
  local server only serves chat (vLLM has no ``/v1/embeddings``) this keeps the
  whole pipeline + local search working; the hash is deterministic, so vectors
  are consistent between indexing and query time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import Any

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

LOCAL_PREFIX = "local:"
DEFAULT_LOCAL_DIM = 384

_HASH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "with",
    "as",
    "at",
    "by",
    "it",
    "this",
    "that",
    "be",
    "def",
    "return",
    "import",
    "from",
    "self",
    "none",
    "true",
    "false",
    "class",
}


def normalize_model_id(model: str | None) -> str | None:
    """Strip a ``local:`` prefix; pass other ids through unchanged."""
    if model and model.startswith(LOCAL_PREFIX):
        return model[len(LOCAL_PREFIX) :]
    return model


def is_local_model(model: str | None) -> bool:
    return bool(model and model.startswith(LOCAL_PREFIX))


def probe_local_server(base_url: str | None, model: str | None) -> dict[str, Any]:
    """Best-effort probe of a local OpenAI-compatible server.

    Returns ``{"reachable", "models", "supports_embeddings", "warning"}``. Never
    raises: on any failure it degrades to warnings so indexing can proceed.
    """
    out: dict[str, Any] = {
        "reachable": False,
        "models": [],
        "supports_embeddings": False,
        "warning": None,
    }
    if not base_url:
        return out
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        out["reachable"] = True
        out["models"] = [m.get("id") for m in data.get("data", [])]
    except Exception as exc:  # noqa: BLE001
        out["warning"] = f"could not reach {base}/models: {exc}"
        return out
    want = normalize_model_id(model)
    if want and want not in out["models"]:
        out["warning"] = (
            f"model id '{want}' not listed by {base}/models "
            f"(available: {out['models']}); continuing anyway"
        )
    try:
        req = urllib.request.Request(
            f"{base}/embeddings",
            data=json.dumps({"model": want or "x", "input": "probe"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
            out["supports_embeddings"] = isinstance(body, dict) and "data" in body
    except Exception:  # noqa: BLE001
        out["supports_embeddings"] = False
        if not out["warning"]:
            out["warning"] = (
                f"{base}/embeddings unavailable -> using built-in local hash "
                f"embeddings (deterministic, non-semantic) for the vector store"
            )
    return out


class LocalHashEmbedding:
    """Deterministic local embedding (token hashing) for servers without /v1/embeddings.

    Implements the ``graphrag_llm`` embedding protocol surface used by the
    vector-store layer: ``embedding``, ``embedding_async``, ``metrics_store``,
    ``tokenizer``. Registered as type ``"local"`` in GraphRAG's
    ``embedding_factory``.
    """

    _metrics_store: Any
    _tokenizer: Any
    _dim: int

    def __init__(
        self, *, model_id: str, model_config: Any, tokenizer: Any, metrics_store: Any, **kwargs: Any
    ) -> None:
        extra = getattr(model_config, "model_extra", None) or {}
        self._dim = int(extra.get("dimensions") or DEFAULT_LOCAL_DIM)
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in re.findall(r"[a-z0-9_]+", (text or "").lower()):
            if len(tok) < 2 or tok in _HASH_STOPWORDS:
                continue
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _response(self, texts: list[str]) -> Any:
        # Build the exact type GraphRAG's vector store expects, from a plain
        # dict so we never depend on a specific litellm export.
        from graphrag_llm.types import LLMEmbeddingResponse

        return LLMEmbeddingResponse(
            **{
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": self._embed_one(t),
                    }
                    for i, t in enumerate(texts)
                ],
                "model": "local-hash",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )

    def embedding(self, /, **kwargs: Any) -> Any:
        return self._response(list(kwargs.get("input") or []))

    async def embedding_async(self, /, **kwargs: Any) -> Any:
        return self._response(list(kwargs.get("input") or []))

    @property
    def metrics_store(self) -> Any:
        return self._metrics_store

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


_registered = False


def register_local_embedding() -> None:
    """Register ``LocalHashEmbedding`` as GraphRAG embedding type ``"local"`` (idempotent)."""
    global _registered
    if _registered:
        return
    try:
        from graphrag_llm.embedding.embedding_factory import embedding_factory

        if "local" in embedding_factory:
            _registered = True
            return
        embedding_factory.register("local", LocalHashEmbedding, scope="singleton")
        _registered = True
    except Exception:  # noqa: BLE001
        logger.exception("could not register local hash embedding")
