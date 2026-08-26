"""Microsoft GraphRAG semantic search over a built Code GraphRAG index.

Rebuilds the (redacted) GraphRagConfig from ``graphrag_settings.json`` saved at
index time, loads the parquet tables + LanceDB via GraphRAG's own readers, and
runs local / global / drift / basic search. Requires a working LLM (mock mode
works offline).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    answer: str
    search_type: str
    context: dict[str, Any] = None
    error: str | None = None
    reasoning: str = ""


def _rekey_model_configs(data: dict[str, Any]) -> None:
    """Re-resolve redacted api_keys from the environment at query time.

    Indexing redacts ``api_key`` to ``None`` so no secret is persisted. For real
    (litellm) models we recover the key from the env vars the indexer used;
    mock models need no key.
    """
    for section in ("completion_models", "embedding_models"):
        for mc in (data.get(section) or {}).values():
            if not isinstance(mc, dict):
                continue
            mtype = mc.get("type")
            if mtype in ("mock", "local"):
                continue
            if mc.get("api_key"):
                continue
            provider = str(mc.get("model_provider", "")).lower()
            if "anthropic" in provider:
                mc["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "") or "sk-ant-no-key-offline"
            else:
                mc["api_key"] = os.environ.get("OPENAI_API_KEY", "") or "sk-no-key-offline"


def _load_graphrag_config(index_dir: Path) -> Any:
    from graphrag.config.models.graph_rag_config import GraphRagConfig

    settings_path = index_dir / "graphrag_settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"graphrag_settings.json not found in {index_dir}")
    data = json.loads(settings_path.read_text())
    data.pop("_code_graphrag", None)
    _rekey_model_configs(data)
    config = GraphRagConfig(**data)
    if any(
        isinstance(m, dict) and m.get("type") == "local"
        for m in (data.get("embedding_models") or {}).values()
    ):
        from code_graphrag.graphrag.local_llm import register_local_embedding

        register_local_embedding()
    _enable_mock_streaming(config)
    return config


def _is_mock_config(config: Any) -> bool:
    def _mock(mc) -> bool:
        return isinstance(mc, dict) and mc.get("type") == "mock"

    try:
        comp = config.completion_models
        emb = config.embedding_models
    except Exception:  # noqa: BLE001
        return False
    comps = [m for m in comp.values()] if hasattr(comp, "values") else []
    embs = [m for m in emb.values()] if hasattr(emb, "values") else []
    return all(_mock(m.model_dump()) for m in comps) and all(_mock(m.model_dump()) for m in embs)


def _synthesize_response(model: Any, canned_json: str) -> Any:
    """Return a valid ``model`` instance.

    Try the canned JSON first (matches the indexer's schema); if it doesn't fit
    the query-side schema (e.g. drift's ``PrimerResponse``), fall back to a
    minimal instance built from the model's required fields.
    """
    import json
    import typing

    from pydantic import BaseModel

    if not isinstance(model, type) or not issubclass(model, BaseModel):
        return None
    try:
        return model(**json.loads(canned_json))
    except Exception:  # noqa: BLE001
        pass

    def _default_for(annotation: Any) -> Any:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is not None and type(None) in args:
            return None
        if annotation is str or origin is str:
            return "N/A (offline mock)"
        if annotation is float:
            return 0.0
        if annotation is int:
            return 0
        if annotation is bool:
            return False
        if origin is list or annotation is list:
            # A list of strings needs at least one item for downstream logic
            # (e.g. drift requires non-empty follow_up_queries).
            if args and args[0] in (str, type(None)):
                return ["N/A (offline mock)"]
            return []
        if origin is dict or annotation is dict:
            return {}
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_validate({})
        return "N/A (offline mock)"

    data: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if field.is_required():
            data[name] = _default_for(field.annotation)
    return model(**data)


def _enable_mock_streaming(config: Any) -> None:
    """Make GraphRAG's ``MockLLMCompletion`` work offline at query time.

    Two gaps in the stock mock break the search API offline:
    1. ``local_search`` / ``basic_search`` call ``completion_async(stream=True)``,
       which the mock rejects -> we drain to one canned response and emit a
       single delta chunk.
    2. Structured queries (drift) pass a ``response_format`` whose schema differs
       from the indexer's canned JSON -> we synthesize a valid instance.

    No-op for real (litellm) models.
    """
    if not _is_mock_config(config):
        return
    try:
        import graphrag_llm.completion.mock_llm_completion as mm
    except Exception:  # noqa: BLE001
        return

    if getattr(mm.MockLLMCompletion, "_cg_stream_patched", False):
        return

    def _tolerant_sync(self, /, **kwargs):
        """Non-streaming completion that survives arbitrary response_format."""
        response_format = kwargs.pop("response_format", None)
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)
        from graphrag_llm.utils import (
            create_completion_response,
            structure_completion_response,
        )

        raw = self._mock_responses[self._mock_index % len(self._mock_responses)]
        self._mock_index += 1
        response = create_completion_response(raw)
        if response_format is not None:
            try:
                response.formatted_response = structure_completion_response(raw, response_format)
            except Exception:  # noqa: BLE001
                response.formatted_response = _synthesize_response(response_format, raw)
        return response

    def tolerant_completion(self, /, **kwargs):
        if kwargs.get("stream", False):
            # Streaming handled by tolerant_async; degrade to one-shot response.
            return _tolerant_sync(self, **kwargs)
        return _tolerant_sync(self, **kwargs)

    async def tolerant_async(self, /, **kwargs):
        stream = bool(kwargs.get("stream", False))
        if not stream:
            return _tolerant_sync(self, **kwargs)
        from types import SimpleNamespace

        response = _tolerant_sync(self, **kwargs)
        text = getattr(response, "content", "") or ""

        async def _gen():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

        return _gen()

    mm.MockLLMCompletion.completion = tolerant_completion
    mm.MockLLMCompletion.completion_async = tolerant_async
    mm.MockLLMCompletion._cg_stream_patched = True


async def _load_tables(index_dir: Path) -> dict[str, Any]:
    from graphrag.data_model.data_reader import DataReader
    from graphrag_storage import StorageConfig, create_storage
    from graphrag_storage.tables.table_provider_config import TableProviderConfig
    from graphrag_storage.tables.table_provider_factory import create_table_provider

    storage = create_storage(StorageConfig(type="file", base_dir=str(index_dir / "output")))
    provider = create_table_provider(TableProviderConfig(type="parquet"), storage)
    reader = DataReader(provider)

    return {
        "entities": await reader.entities(),
        "relationships": await reader.relationships(),
        "text_units": await reader.text_units(),
        "communities": await reader.communities(),
        "community_reports": await reader.community_reports(),
    }


async def _run_search(
    config: Any,
    tables: dict[str, Any],
    search_type: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> SearchResult:
    from graphrag.api import (
        basic_search,
        drift_search,
        global_search,
        local_search,
    )

    try:
        if search_type == "basic":
            answer, context = await basic_search(
                config=config,
                text_units=tables["text_units"],
                response_type=response_type,
                query=query,
            )
        elif search_type == "global":
            answer, context = await global_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                community_level=community_level,
                dynamic_community_selection=True,
                response_type=response_type,
                query=query,
            )
        elif search_type == "drift":
            answer, context = await drift_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
        else:  # local
            answer, context = await local_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                covariates=None,
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
        return SearchResult(
            answer=str(answer), search_type=search_type, context=_ctx_to_dict(context)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("GraphRAG %s search failed", search_type)
        return SearchResult(answer="", search_type=search_type, error=str(exc))


def _ctx_to_dict(context: Any) -> dict[str, Any]:
    """Best-effort conversion of the context payload to a JSON-safe dict."""
    import pandas as pd

    if isinstance(context, dict):
        out: dict[str, Any] = {}
        for k, v in context.items():
            if isinstance(v, pd.DataFrame):
                out[k] = v.head(20).to_dict("records")
            else:
                out[k] = v
        return out
    if isinstance(context, pd.DataFrame):
        return {"context": context.head(20).to_dict("records")}
    return {"context": str(context)}


async def agraphrag_search(
    index_dir: str | Path,
    query: str,
    search_type: str = "local",
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> SearchResult:
    """Run a GraphRAG semantic search against the index (async).

    Always returns a ``SearchResult``; setup problems (missing settings, bad
    tables) are reported in ``.error`` rather than raised, so callers (CLI,
    workflow, agent tool) never have to catch exceptions.
    """
    index_dir = Path(index_dir)
    try:
        config = _load_graphrag_config(index_dir)
    except Exception as exc:  # noqa: BLE001
        return SearchResult(answer="", search_type=search_type, error=f"load config: {exc}")
    try:
        tables = await _load_tables(index_dir)
    except Exception as exc:  # noqa: BLE001
        return SearchResult(answer="", search_type=search_type, error=f"load tables: {exc}")
    from code_graphrag.query.reasoning import capture_reasoning, drain

    with capture_reasoning():
        result = await _run_search(
            config,
            tables,
            search_type,
            query,
            community_level=community_level,
            response_type=response_type,
        )
    result.reasoning = drain()
    return result


def graphrag_search(
    index_dir: str | Path,
    query: str,
    search_type: str = "local",
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> SearchResult:
    """Run a GraphRAG semantic search against the index (blocking wrapper).

    Safe to call from a sync context (uses a fresh event loop) or from inside a
    running loop (delegates to a worker thread with its own loop).
    """
    import asyncio
    import concurrent.futures

    kwargs = dict(
        index_dir=index_dir,
        query=query,
        search_type=search_type,
        community_level=community_level,
        response_type=response_type,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(agraphrag_search(**kwargs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(agraphrag_search(**kwargs))).result()
