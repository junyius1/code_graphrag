"""LangGraph workflow orchestrating the Code GraphRAG index build.

Nodes (verified against langgraph 1.2.x: node fns take (state), return a partial
state dict, edges use START/END):

    discover_files -> parse_source -> build_semantic_model -> [extract_semantics
    (optional LLM enrichment)] -> build_graph_data -> build_graphrag_index -> finalize

The state carries lightweight references (paths, counts) rather than the full
graph to keep checkpoints small; the in-memory artifacts are threaded via a
module-level run context so nodes can access the heavy objects without bloating
the LangGraph state schema.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from code_graphrag.config.models import IndexConfig
from code_graphrag.logging_setup import get_logger
from code_graphrag.semantic.builder import FileContext

logger = get_logger(__name__)


@dataclass
class _RunArtifacts:
    """Heavy per-run objects threaded between nodes without entering graph state."""

    config: IndexConfig | None = None
    output_dir: Path | None = None
    discovery: Any | None = None
    parse_result: Any | None = None
    files_ctx: dict[str, FileContext] = field(default_factory=dict)
    graph: Any | None = None
    tables: Any | None = None
    errors: list[str] = field(default_factory=list)


def _new_run_state(
    source: str,
    output: str,
    config: IndexConfig,
) -> _RunArtifacts:
    config.discovery.source = source
    config.output_dir = output
    return _RunArtifacts(config=config, output_dir=Path(output).resolve())


class IndexWorkflowState(TypedDict, total=False):
    source: str
    output: str
    files_discovered: int
    files_source: int
    files_parsed: int
    parse_failures: Annotated[list[str], operator.add]
    unsupported: Annotated[list[str], operator.add]
    entities: int
    relations: int
    entity_types: str
    status: str
    error: str | None


# A single module-level artifact holder per workflow invocation. Nodes are
# executed sequentially within one ainvoke, so this is safe and keeps the
# LangGraph state schema small and JSON-serializable.
_ARTIFACTS: _RunArtifacts | None = None


def _arts() -> _RunArtifacts:
    assert _ARTIFACTS is not None, "workflow not initialized"
    return _ARTIFACTS


# ---- nodes ---------------------------------------------------------------


async def discover_files(state: IndexWorkflowState) -> dict[str, Any]:
    from code_graphrag.discovery.discovery import discover_files as _discover
    from code_graphrag.parsing.parse import read_config_text, read_doc_text

    arts = _arts()
    disc = _discover(state["source"], arts.config)
    arts.discovery = disc
    for f in disc.sources:
        # source text is filled in by parse_source (it reads bytes once)
        arts.files_ctx[f.rel_path] = FileContext(
            rel_path=f.rel_path,
            kind="source",
            structure=None,
            text="",
            line_count=0,
        )
    for f in disc.docs:
        arts.files_ctx[f.rel_path] = FileContext(
            rel_path=f.rel_path,
            kind="doc",
            text=read_doc_text(disc, f.rel_path, arts.config.discovery.max_file_bytes),
            line_count=len(
                read_doc_text(disc, f.rel_path, arts.config.discovery.max_file_bytes).splitlines()
            ),
        )
    for f in disc.configs:
        arts.files_ctx[f.rel_path] = FileContext(
            rel_path=f.rel_path,
            kind="config",
            text=read_config_text(disc, f.rel_path, arts.config.discovery.max_file_bytes),
            line_count=len(
                read_config_text(
                    disc, f.rel_path, arts.config.discovery.max_file_bytes
                ).splitlines()
            ),
        )
    logger.info("discover_files: %d files (%d source)", len(disc.files), len(disc.sources))
    return {
        "files_discovered": len(disc.files),
        "files_source": len(disc.sources),
    }


async def parse_source(state: IndexWorkflowState) -> dict[str, Any]:
    from code_graphrag.parsing.parse import parse_files

    arts = _arts()
    pr = parse_files(arts.discovery, arts.config)
    arts.parse_result = pr
    for rel, ctx in arts.files_ctx.items():
        if ctx.kind == "source" and rel in pr.structures:
            ctx.structure = pr.structures[rel]
            ctx.text = pr.text_cache.get(rel, ctx.text)
            ctx.line_count = len(pr.text_cache.get(rel, "").splitlines())
    return {
        "files_parsed": len(pr.structures),
        "parse_failures": [f.file for f in pr.failures],
        "unsupported": pr.skipped_unsupported,
    }


async def build_semantic_model(state: IndexWorkflowState) -> dict[str, Any]:
    from code_graphrag.semantic.builder import CodeGraphBuilder

    arts = _arts()
    repo = Path(state["source"]).resolve().name or "repo"
    builder = CodeGraphBuilder(repo, arts.files_ctx)
    arts.graph = builder.build()
    stats = arts.graph.stats()
    return {
        "entities": stats["entities"],
        "relations": stats["relations"],
        "entity_types": ", ".join(
            f"{k}={v}" for k, v in sorted(stats["by_entity_type"].items(), key=lambda kv: -kv[1])
        ),
    }


async def extract_semantics(state: IndexWorkflowState) -> dict[str, Any]:
    """Optional LLM enrichment. Skipped when disabled or in mock mode (offline)."""
    arts = _arts()
    if not arts.config.enrichment.enabled:
        logger.info("extract_semantics: skipped (enrichment disabled)")
        return {"status": "enrichment-skipped"}
    if str(arts.config.llm.provider).endswith("MOCK"):
        logger.info("extract_semantics: skipped (mock mode is offline)")
        return {"status": "enrichment-skipped"}
    # Real LLM enrichment would attach richer descriptions to entities here.
    # Kept as an explicit, isolated step so it can be enabled per run.
    from code_graphrag.llm.enrichment import enrich_graph

    await enrich_graph(arts.graph, arts.files_ctx, arts.config)
    return {"status": "enrichment-done"}


async def build_graph_data(state: IndexWorkflowState) -> dict[str, Any]:
    """Materialize the GraphRAG input tables (deterministic)."""
    from code_graphrag.graphrag.tables import build_graphrag_tables

    arts = _arts()
    cfg = arts.config
    arts.tables = build_graphrag_tables(
        arts.graph,
        arts.files_ctx,
        max_text_unit_chars=cfg.adapter.max_text_unit_chars,
        include_config=cfg.adapter.include_config_entities,
        include_docs=cfg.adapter.include_doc_entities,
    )
    logger.info(
        "build_graph_data: %d units, %d entities, %d relationships",
        len(arts.tables.text_units),
        len(arts.tables.entities),
        len(arts.tables.relationships),
    )
    return {"status": "graph-data-built"}


async def build_graphrag_index(state: IndexWorkflowState) -> dict[str, Any]:
    from graphrag.api import build_index

    from code_graphrag.graphrag.adapter import (
        build_graphrag_config,
        save_graphrag_settings,
        save_sidecar,
    )

    arts = _arts()
    arts.output_dir.mkdir(parents=True, exist_ok=True)
    graphrag_config = build_graphrag_config(arts.config, arts.output_dir, arts.tables)
    # Persist BEFORE build_index: mock mock_responses are consumed during run.
    save_graphrag_settings(graphrag_config, arts.output_dir, arts.config)
    outputs = await build_index(
        config=graphrag_config,
        additional_context={"code_graphrag_tables": arts.tables},
        verbose=arts.config.verbose,
    )
    failed = [o for o in outputs if o.error is not None]
    if failed:
        detail = "; ".join(f"{o.workflow}: {o.error}" for o in failed)
        raise RuntimeError(f"GraphRAG indexing failed: {detail}")
    save_sidecar(arts.graph, arts.tables, arts.output_dir, arts.config)
    return {"status": "indexed"}


async def finalize(state: IndexWorkflowState) -> dict[str, Any]:
    arts = _arts()
    logger.info("finalize: index complete at %s", arts.output_dir)
    return {"status": "done"}


# ---- graph ---------------------------------------------------------------


def build_index_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(IndexWorkflowState)
    builder.add_node("discover_files", discover_files)
    builder.add_node("parse_source", parse_source)
    builder.add_node("build_semantic_model", build_semantic_model)
    builder.add_node("extract_semantics", extract_semantics)
    builder.add_node("build_graph_data", build_graph_data)
    builder.add_node("build_graphrag_index", build_graphrag_index)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "discover_files")
    builder.add_edge("discover_files", "parse_source")
    builder.add_edge("parse_source", "build_semantic_model")
    builder.add_edge("build_semantic_model", "extract_semantics")
    builder.add_edge("extract_semantics", "build_graph_data")
    builder.add_edge("build_graph_data", "build_graphrag_index")
    builder.add_edge("build_graphrag_index", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


async def run_index_workflow(
    source: str,
    output: str,
    config: IndexConfig | None = None,
) -> dict[str, Any]:
    """Run the full index pipeline and return the final state."""
    global _ARTIFACTS
    config = config or IndexConfig()
    _ARTIFACTS = _new_run_state(source, output, config)
    graph = build_index_graph()
    result = await graph.ainvoke(
        {"source": source, "output": output, "status": "starting", "error": None},
        {"configurable": {"thread_id": "code-graphrag-index"}},
    )
    return result
