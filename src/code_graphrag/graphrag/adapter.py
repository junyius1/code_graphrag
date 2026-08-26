"""Adapter: feed the Code Knowledge Graph into the Microsoft GraphRAG v3 pipeline.

Strategy (verified against 3partylibs/graphrag v3.1.2 source):
- Register ONE custom workflow, ``load_code_graph``, that writes the
  deterministic ``documents`` / ``text_units`` / ``entities`` / ``relationships``
  tables (pulled from ``context.state["additional_context"]``).
- Set ``config.workflows`` to a custom pipeline that skips GraphRAG's
  ``load_input_documents`` / ``create_base_text_units`` / ``extract_graph`` /
  ``extract_covariates`` (the LLM extraction we replaced with static analysis)
  and keeps the deterministic stock workflows: ``create_final_documents``,
  ``finalize_graph``, ``create_communities``, ``create_final_text_units``,
  ``create_community_reports``, ``generate_text_embeddings``.
- ``build_index(config, ...)`` then runs the pipeline; storage is created from
  config inside ``run_pipeline``.

The index sidecar (``code_graphrag_index.json``) maps GraphRAG titles back to
code entity ids / locations so queries can cite ``file:line``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from code_graphrag.config.models import IndexConfig
from code_graphrag.graphrag.tables import GraphRagTables, build_graphrag_tables
from code_graphrag.logging_setup import get_logger
from code_graphrag.semantic.builder import FileContext
from code_graphrag.semantic.model import CodeKnowledgeGraph

logger = get_logger(__name__)

# The code-graph workflows we register (idempotent registration).
_CODE_WORKFLOWS: set[str] = set()


def _resolve_api_key(explicit: str | None, provider: str) -> str:
    """Resolve an API key: explicit > env var > placeholder (offline/mock)."""
    import os

    if explicit:
        return explicit
    if provider in ("openai", "openai_compatible", "ollama"):
        return os.environ.get("OPENAI_API_KEY", "") or "sk-no-key-offline"
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "") or "sk-ant-no-key-offline"
    return "sk-no-key-offline"


def _completion_model_config(llm: Any) -> dict[str, Any]:
    """Build a GraphRAG ModelConfig dict for the completion model.

    ``llm.provider == mock`` uses GraphRAG's built-in mock LLM (canned
    responses) so the pipeline runs fully offline.
    """
    if getattr(llm, "provider", None) == "mock" or str(llm.provider) == "LLMProvider.MOCK":
        from graphrag.index.operations.summarize_communities.community_reports_extractor import (
            CommunityReportResponse,
        )

        mock = CommunityReportResponse(
            title="Community report",
            summary="Summary of the community generated in offline/mock mode.",
            findings=[],
            rating=5.0,
            rating_explanation="Mock report (offline run).",
        )
        return {
            "type": "mock",
            "model": "mock-model",
            "model_provider": "mock",
            "mock_responses": [mock.model_dump_json()],
        }
    cfg: dict[str, Any] = {
        "model": llm.model,
        "model_provider": "openai"
        if str(llm.provider)
        in ("LLMProvider.OPENAI", "LLMProvider.OPENAI_COMPATIBLE", "openai", "openai_compatible")
        else "openai",
    }
    provider = (
        "openai"
        if str(llm.provider)
        in ("LLMProvider.OPENAI", "LLMProvider.OPENAI_COMPATIBLE", "openai", "openai_compatible")
        else ("anthropic" if "ANTHROPIC" in str(llm.provider) else "openai")
    )
    cfg["api_key"] = _resolve_api_key(llm.api_key, provider)
    if llm.base_url:
        cfg["api_base"] = llm.base_url
    return cfg


MOCK_EMBEDDING_DIM = 16


def _is_mock_mode(config: IndexConfig) -> bool:
    return (
        str(config.llm.provider) in ("LLMProvider.MOCK", "mock")
        or config.llm.provider.name == "MOCK"
    )


def _embedding_model_config(config: IndexConfig) -> dict[str, Any]:
    emb = config.embedding
    if _is_mock_mode(config):
        return {
            "type": "mock",
            "model": "mock-embedding",
            "model_provider": "mock",
            "mock_responses": [0.1 * (i + 1) for i in range(MOCK_EMBEDDING_DIM)],
        }
    cfg: dict[str, Any] = {
        "model": emb.model,
        "model_provider": emb.provider or "openai",
        "api_key": _resolve_api_key(emb.api_key, emb.provider or "openai"),
    }
    if emb.base_url:
        cfg["api_base"] = emb.base_url
    return cfg


def _vector_size_for(config: IndexConfig) -> int:
    emb = config.embedding
    if _is_mock_mode(config):
        return MOCK_EMBEDDING_DIM
    if emb.dimensions:
        return emb.dimensions
    model = emb.model.lower()
    if model.startswith("text-embedding-3-small"):
        return 1536
    if model.startswith("text-embedding-3-large"):
        return 3072
    if model.startswith("text-embedding-ada-002"):
        return 1536
    return 1536


def _register_load_code_graph() -> None:
    """Register the load_code_graph workflow with GraphRAG's PipelineFactory."""
    if "load_code_graph" in _CODE_WORKFLOWS:
        return
    _CODE_WORKFLOWS.add("load_code_graph")

    from graphrag.index.typing.context import PipelineRunContext
    from graphrag.index.typing.workflow import WorkflowFunctionOutput
    from graphrag.index.workflows.factory import PipelineFactory

    async def load_code_graph(config: Any, context: PipelineRunContext) -> WorkflowFunctionOutput:
        """Write the pre-built code graph tables into the output provider."""
        import graphrag  # noqa: F401 - ensure workflow registry is populated

        extra = context.state.get("additional_context", {})
        tables: GraphRagTables = extra["code_graphrag_tables"]
        provider = context.output_table_provider
        await provider.write_dataframe("documents", tables.documents)
        await provider.write_dataframe("text_units", tables.text_units)
        await provider.write_dataframe("entities", tables.entities)
        await provider.write_dataframe("relationships", tables.relationships)
        logger.info(
            "load_code_graph: wrote %d documents, %d text_units, %d entities, %d relationships",
            len(tables.documents),
            len(tables.text_units),
            len(tables.entities),
            len(tables.relationships),
        )
        return WorkflowFunctionOutput(result=None)

    PipelineFactory.register("load_code_graph", load_code_graph)


# The stock workflows we keep after our custom loader (verified v3.1.2 names).
_CODE_PIPELINE_WORKFLOWS = [
    "load_code_graph",
    "create_final_documents",
    "finalize_graph",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "generate_text_embeddings",
]


def build_graphrag_config(config: IndexConfig, output_dir: Path, tables: GraphRagTables) -> Any:
    """Assemble the GraphRagConfig for the code-graph pipeline run."""

    from graphrag.config import defaults as defs
    from graphrag.config.models.graph_rag_config import GraphRagConfig

    _register_load_code_graph()

    base = asdict(defs.graphrag_config_defaults)

    vector_size = _vector_size_for(config)
    base["output_storage"] = {"type": "file", "base_dir": str(output_dir / "output")}
    base["input_storage"] = {"type": "file", "base_dir": str(output_dir / "input")}
    base["update_output_storage"] = {"type": "file", "base_dir": str(output_dir / "update_output")}
    base["reporting"] = {"type": "file", "base_dir": str(output_dir / "logs")}
    base["cache"] = {
        "type": "json",
        "storage": {"type": "file", "base_dir": str(output_dir / "cache")},
    }
    base["vector_store"] = {
        "type": "lancedb",
        "db_uri": str(output_dir / config.vector_db_subdir),
        "vector_size": vector_size,
    }
    base["snapshots"] = {"graphml": True, "raw_graph": False, "embeddings": False}
    # Force use_lcc=False: GraphRAG's stable_lcc uppercases node names, which
    # would break the case-sensitive title->entity merge in create_communities
    # for code (e.g. "App.handle" vs "APP.HANDLE"). Documented in stable_lcc.py.
    if config.cluster.use_lcc:
        logger.warning(
            "cluster.use_lcc is True but is forced to False for code graphs "
            "(GraphRAG stable_lcc uppercases node names, breaking case-sensitive titles)."
        )
    base["cluster_graph"] = {
        "max_cluster_size": config.cluster.max_cluster_size,
        "use_lcc": False,
        "seed": config.cluster.seed,
    }
    base["extract_graph"] = {
        **base.get("extract_graph", {}),
        "entity_types": list(config.adapter.entity_types),
    }
    base["workflows"] = list(_CODE_PIPELINE_WORKFLOWS)
    base["completion_models"] = {
        "default_completion_model": _completion_model_config(config.llm),
    }
    base["embedding_models"] = {
        "default_embedding_model": _embedding_model_config(config),
    }

    return GraphRagConfig(**base)


def save_sidecar(
    graph: CodeKnowledgeGraph,
    tables: GraphRagTables,
    output_dir: Path,
    config: IndexConfig,
) -> None:
    """Persist the code-side metadata that GraphRAG tables cannot hold."""
    entities = []
    for e in graph.entities.values():
        entities.append(
            {
                "id": e.id,
                "name": e.name,
                "qualified_name": e.qualified_name,
                "type": e.type,
                "file": e.file,
                "start_line": e.start_line,
                "end_line": e.end_line,
                "location": e.location,
                "description": e.description,
                "language": e.language,
                "attributes": e.attributes,
                "graphrag_title": tables.entity_title_by_id.get(e.id),
            }
        )
    relations = []
    for r in graph.relations.values():
        relations.append(
            {
                "source": r.source_id,
                "target": r.target_id,
                "type": r.type,
                "description": r.description,
                "weight": r.weight,
            }
        )
    sidecar = {
        "name": config.name,
        "repo": graph.repo,
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
        "entities": entities,
        "relations": relations,
        "stats": graph.stats(),
    }
    (output_dir / "code_graphrag_index.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False)
    )
    logger.info(
        "Saved sidecar code_graphrag_index.json (%d entities, %d relations)",
        len(entities),
        len(relations),
    )


_REDACT_KEYS = ("api_key", "azure_deployment_name")


def _redact_model_configs(cfg: Any) -> dict[str, Any]:
    """Serialize the GraphRagConfig to a JSON-safe dict with secrets redacted."""
    data = cfg.model_dump(mode="json")
    for section in ("completion_models", "embedding_models"):
        for _mid, mc in (data.get(section) or {}).items():
            if isinstance(mc, dict):
                for k in _REDACT_KEYS:
                    if k in mc:
                        mc[k] = None
    # mock responses must survive redaction (needed for offline query)
    return data


def save_graphrag_settings(graphrag_config: Any, output_dir: Path, config: IndexConfig) -> None:
    """Persist the (redacted) GraphRagConfig so queries can rebuild it later."""
    data = _redact_model_configs(graphrag_config)
    data["_code_graphrag"] = {
        "name": config.name,
        "llm": config.llm.model_dump(),
        "embedding": config.embedding.model_dump(),
    }
    (output_dir / "graphrag_settings.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )
    logger.info("Saved graphrag_settings.json (api keys redacted)")


async def run_graphrag_indexing(
    config: IndexConfig,
    output_dir: Path,
    graph: CodeKnowledgeGraph,
    files: dict[str, FileContext],
) -> GraphRagTables:
    """Run the full GraphRAG indexing pipeline for the code graph (async)."""
    from graphrag.api import build_index

    tables = build_graphrag_tables(
        graph,
        files,
        max_text_unit_chars=config.adapter.max_text_unit_chars,
        include_config=config.adapter.include_config_entities,
        include_docs=config.adapter.include_doc_entities,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    graphrag_config = build_graphrag_config(config, output_dir, tables)
    # Persist BEFORE build_index: mock mock_responses are consumed during run.
    save_graphrag_settings(graphrag_config, output_dir, config)
    outputs = await build_index(
        config=graphrag_config,
        additional_context={"code_graphrag_tables": tables},
        verbose=config.verbose,
    )
    failed = [o for o in outputs if o.error is not None]
    if failed:
        detail = "; ".join(f"{o.workflow}: {o.error}" for o in failed)
        raise RuntimeError(f"GraphRAG indexing failed: {detail}")
    save_sidecar(graph, tables, output_dir, config)
    logger.info("GraphRAG indexing complete -> %s", output_dir)
    return tables
