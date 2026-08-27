"""Code GraphRAG command line interface (Typer).

Examples:
    code-graphrag index --source /path/to/repo --output /path/to/index
    code-graphrag query --index /path/to/index --question "Where is build_index implemented?"
    code-graphrag query --index /path/to/index --question "What does UserService register?" --mode local
    code-graphrag inspect --index /path/to/index --entity UserService
"""

from __future__ import annotations

import json

import typer

from code_graphrag.config.models import (
    IndexConfig,
    LLMProvider,
)
from code_graphrag.logging_setup import get_logger, setup_logging

app = typer.Typer(
    name="code-graphrag",
    help="Code GraphRAG: deterministic code knowledge graphs + Microsoft GraphRAG.",
    no_args_is_help=True,
)
logger = get_logger("code_graphrag.cli")


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #


def _build_index_config(
    source: str,
    output: str,
    name: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_api_key: str | None,
    llm_base_url: str | None,
    emb_model: str | None,
    emb_api_key: str | None,
    emb_base_url: str | None,
    emb_dimensions: int | None,
    enrich: bool,
    max_text_unit_chars: int | None,
    max_cluster_size: int | None,
    extra_exclude: list[str] | None,
    verbose: bool,
    local: bool = False,
    local_model: str | None = None,
    local_base_url: str | None = None,
) -> IndexConfig:
    import os

    cfg = IndexConfig()
    cfg.discovery.source = source
    cfg.output_dir = output
    cfg.verbose = verbose
    if name:
        cfg.name = name
    # --local (or LLM_MODEL_ID / LOCAL_LLM_BASE_URL env): target a local
    # OpenAI-compatible server (vLLM/Ollama); model written as local:/path.
    local_model = local_model or os.environ.get("LLM_MODEL_ID")
    local_base_url = local_base_url or os.environ.get("LOCAL_LLM_BASE_URL")
    if local or local_model or local_base_url:
        if not local_base_url:
            typer.echo(
                "Error: --local requires a base URL: pass --local-base-url or set "
                "LOCAL_LLM_BASE_URL (e.g. http://localhost:8000/v1)",
                err=True,
            )
            raise typer.Exit(code=2)
        cfg.llm.provider = LLMProvider.OPENAI_COMPATIBLE
        cfg.llm.base_url = local_base_url
        if local_model:
            cfg.llm.model = (
                local_model if local_model.startswith("local:") else f"local:{local_model}"
            )
        if not cfg.llm.api_key:
            cfg.llm.api_key = "sk-no-key-offline"
    if llm_provider:
        cfg.llm.provider = LLMProvider(llm_provider.replace("-", "_"))
    if llm_model:
        cfg.llm.model = llm_model
    if llm_api_key:
        cfg.llm.api_key = llm_api_key
    if llm_base_url:
        cfg.llm.base_url = llm_base_url
    if emb_model:
        cfg.embedding.model = emb_model
    if emb_api_key:
        cfg.embedding.api_key = emb_api_key
    if emb_base_url:
        cfg.embedding.base_url = emb_base_url
    elif (
        cfg.llm.base_url
        and cfg.llm.provider is LLMProvider.OPENAI_COMPATIBLE
        and not cfg.embedding.api_key
    ):
        # point embeddings at the local server too; the adapter detects a
        # missing /v1/embeddings and falls back to local hashing
        cfg.embedding.base_url = cfg.llm.base_url
    if emb_dimensions:
        cfg.embedding.dimensions = emb_dimensions
    if enrich:
        cfg.enrichment.enabled = True
    if max_text_unit_chars:
        cfg.adapter.max_text_unit_chars = max_text_unit_chars
    if max_cluster_size:
        cfg.cluster.max_cluster_size = max_cluster_size
    if extra_exclude:
        cfg.discovery.extra_exclude = extra_exclude
    return cfg


# --------------------------------------------------------------------------- #
# index command
# --------------------------------------------------------------------------- #


@app.command()
def index(
    source: str = typer.Option(..., "--source", "-s", help="Source repository directory to index."),
    output: str = typer.Option(
        ..., "--output", "-o", help="Directory to write the GraphRAG index."
    ),
    name: str | None = typer.Option(None, "--name", help="Index name."),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="openai|openai_compatible|anthropic|ollama|mock"
    ),
    llm_model: str | None = typer.Option(None, "--llm-model", help="Completion model name."),
    llm_api_key: str | None = typer.Option(
        None, "--llm-api-key", envvar="CODE_GRAPHRAG_LLM_API_KEY", help="LLM API key (or env)."
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        envvar="CODE_GRAPHRAG_LLM_BASE_URL",
        help="LLM base URL (OpenAI-compatible).",
    ),
    emb_model: str | None = typer.Option(None, "--emb-model", help="Embedding model name."),
    emb_api_key: str | None = typer.Option(
        None,
        "--emb-api-key",
        envvar="CODE_GRAPHRAG_EMB_API_KEY",
        help="Embedding API key (or env).",
    ),
    emb_base_url: str | None = typer.Option(
        None, "--emb-base-url", envvar="CODE_GRAPHRAG_EMB_BASE_URL", help="Embedding base URL."
    ),
    emb_dimensions: int | None = typer.Option(
        None, "--emb-dimensions", help="Embedding dimensions."
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Use a local OpenAI-compatible LLM (vLLM/Ollama); reads LLM_MODEL_ID / LOCAL_LLM_BASE_URL env.",
    ),
    local_model: str | None = typer.Option(
        None,
        "--local-model",
        envvar="LLM_MODEL_ID",
        help="Local model id, e.g. local:/mnt/D/models/Qwen3.8-27B (env: LLM_MODEL_ID).",
    ),
    local_base_url: str | None = typer.Option(
        None,
        "--local-base-url",
        envvar="LOCAL_LLM_BASE_URL",
        help="Local server base URL, e.g. http://localhost:8000/v1 (env: LOCAL_LLM_BASE_URL).",
    ),
    enrich: bool = typer.Option(False, "--enrich", help="Enable optional LLM semantic enrichment."),
    mock: bool = typer.Option(
        False, "--mock", help="Fully offline mock LLM + embeddings (canned summaries)."
    ),
    max_text_unit_chars: int | None = typer.Option(
        None, "--max-text-unit-chars", help="Max chars per semantic text unit."
    ),
    max_cluster_size: int | None = typer.Option(
        None, "--max-cluster-size", help="Leiden max cluster size."
    ),
    extra_exclude: list[str] | None = typer.Option(
        None, "--exclude", help="Extra path segments to exclude (repeatable)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Build a Code GraphRAG index from a source directory."""
    setup_logging("DEBUG" if verbose else "INFO")
    cfg = _build_index_config(
        source,
        output,
        name,
        llm_provider,
        llm_model,
        llm_api_key,
        llm_base_url,
        emb_model,
        emb_api_key,
        emb_base_url,
        emb_dimensions,
        enrich,
        max_text_unit_chars,
        max_cluster_size,
        extra_exclude,
        verbose,
        local=local,
        local_model=local_model,
        local_base_url=local_base_url,
    )
    if mock:
        cfg.llm.provider = LLMProvider.MOCK
    from code_graphrag.graphrag.local_llm import is_local_model, probe_local_server

    if is_local_model(cfg.llm.model) and cfg.llm.base_url:
        probe = probe_local_server(cfg.llm.base_url, cfg.llm.model)
        if not probe["reachable"]:
            typer.echo(
                f"Error: local LLM server not reachable at {cfg.llm.base_url} - is it running?",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(
            f"Local LLM: model={cfg.llm.model} base_url={cfg.llm.base_url} "
            f"server_models={len(probe['models'])} embeddings_endpoint={'yes' if probe['supports_embeddings'] else 'no (local hash fallback)'}",
            err=True,
        )
        if probe["warning"]:
            typer.echo(probe["warning"], err=True)
    typer.echo(f"Indexing {source} -> {output}")
    if cfg.llm.provider is LLMProvider.MOCK:
        typer.echo("NOTE: LLM provider is 'mock' (fully offline; canned summaries).")
    import asyncio

    from code_graphrag.workflow.index_workflow import run_index_workflow

    result = asyncio.run(run_index_workflow(source, output, cfg))
    if result.get("error"):
        raise typer.Exit(code=1)
    typer.echo("")
    typer.echo("Index build complete.")
    typer.echo(f"  files discovered : {result.get('files_discovered')}")
    typer.echo(f"  files source     : {result.get('files_source')}")
    typer.echo(f"  files parsed     : {result.get('files_parsed')}")
    typer.echo(f"  entities         : {result.get('entities')}")
    typer.echo(f"  relations        : {result.get('relations')}")
    typer.echo(f"  entity types     : {result.get('entity_types')}")
    typer.echo(f"  output dir       : {output}")


# --------------------------------------------------------------------------- #
# inspect command
# --------------------------------------------------------------------------- #


@app.command()
def inspect(
    index_dir: str = typer.Option(..., "--index", "-i", help="Index directory."),
    entity: str | None = typer.Option(
        None, "--entity", "-e", help="Filter/list entities matching this term."
    ),
    relations: bool = typer.Option(
        False, "--relations", "-r", help="Also print relations for matched entities."
    ),
    stats: bool = typer.Option(
        False, "--stats", help="Print entity/relation type stats (default when no --entity)."
    ),
) -> None:
    """Inspect the deterministic code knowledge graph in an index."""
    setup_logging("INFO")
    from code_graphrag.query.queries import CodeGraphQuery

    q = CodeGraphQuery(index_dir)
    if entity:
        from code_graphrag.query.queries import _find_entity

        matches = _find_entity(q.entities, entity)
        if not matches:
            typer.echo(f"No entities match '{entity}'.")
            raise typer.Exit(code=1)
        for e in matches[:40]:
            typer.echo(f"{e['type']:14s} {e['qualified_name']:50s} @ {e['location']}")
        if relations:
            for e in matches[:5]:
                typer.echo("")
                typer.echo(f"-- relations for {e['qualified_name']} --")
                for r in q.relations:
                    if r["source"] == e["id"] or r["target"] == e["id"]:
                        src = q._by_id.get(r["source"], {}).get("qualified_name", r["source"])
                        tgt = q._by_id.get(r["target"], {}).get("qualified_name", r["target"])
                        direction = "->" if r["source"] == e["id"] else "<-"
                        typer.echo(f"   {src} -{r['type']}{direction} {tgt}")
    else:
        typer.echo(q.overview())
        typer.echo("")
        typer.echo("Top entities by type:")
        for e in q.entities[:200]:
            typer.echo(f"   {e['type']:14s} {e['qualified_name']}")


# --------------------------------------------------------------------------- #
# query command
# --------------------------------------------------------------------------- #


@app.command()
def query(
    index_dir: str = typer.Option(..., "--index", "-i", help="Index directory."),
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="The question to answer (omit with --agent for interactive mode).",
    ),
    agent: bool = typer.Option(
        False,
        "--agent",
        help="Use the LangGraph Query Agent (tool-calling, multi-round retrieval).",
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="openai|openai_compatible|anthropic|ollama|mock (agent mode)."
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Completion model name (agent mode; supports local:/path)."
    ),
    llm_api_key: str | None = typer.Option(
        None, "--llm-api-key", envvar="CODE_GRAPHRAG_LLM_API_KEY", help="LLM API key (or env)."
    ),
    llm_base_url: str | None = typer.Option(
        None, "--llm-base-url", envvar="CODE_GRAPHRAG_LLM_BASE_URL", help="LLM base URL."
    ),
    local: bool = typer.Option(
        False, "--local", help="Local LLM; reads LLM_MODEL_ID / LOCAL_LLM_BASE_URL env."
    ),
    local_model: str | None = typer.Option(
        None, "--local-model", envvar="LLM_MODEL_ID", help="Local model id (env: LLM_MODEL_ID)."
    ),
    local_base_url: str | None = typer.Option(
        None, "--local-base-url", envvar="LOCAL_LLM_BASE_URL", help="Local server base URL."
    ),
    max_iterations: int = typer.Option(6, "--max-iterations", help="Max agent retrieval rounds."),
    mock: bool = typer.Option(
        False, "--mock", help="Fully offline mock LLM (agent mode, canned answers)."
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Interactive query REPL (agent mode; no --question needed).",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="auto|where|callers|callees|imports|imported_by|inherits|path|config|files|local|global|drift|basic.",
    ),
    search_type: str = typer.Option(
        "local",
        "--search-type",
        help="GraphRAG search type for LLM modes (local|global|drift|basic).",
    ),
    community_level: int = typer.Option(
        2, "--community-level", help="Community level for GraphRAG search."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Answer a question about the indexed source repository.

    With ``--agent``, a LangGraph Query Agent performs multi-round retrieval
    (question analysis -> controlled GraphRAG/code-graph tools -> budgeted
    evidence -> grounded answer). Without ``--question`` (or with
    ``--interactive``) it starts an interactive REPL.
    """
    setup_logging("INFO" if not verbose else "DEBUG")

    # ---- agent mode ------------------------------------------------------
    if agent or interactive:
        import os

        from code_graphrag.config.models import LLMConfig, LLMProvider
        from graphrag_query import query_repl, run_query_agent

        cfg = LLMConfig()
        local_model = local_model or os.environ.get("LLM_MODEL_ID")
        local_base_url = local_base_url or os.environ.get("LOCAL_LLM_BASE_URL")
        if local or local_model or local_base_url:
            if not local_base_url:
                typer.echo(
                    "Error: --local requires --local-base-url or LOCAL_LLM_BASE_URL",
                    err=True,
                )
                raise typer.Exit(code=2)
            cfg.provider = LLMProvider.OPENAI_COMPATIBLE
            cfg.base_url = local_base_url
            if local_model:
                cfg.model = (
                    local_model if local_model.startswith("local:") else f"local:{local_model}"
                )
            if not cfg.api_key:
                cfg.api_key = "sk-no-key-offline"
        if mock:
            cfg.provider = LLMProvider.MOCK
        if llm_provider:
            cfg.provider = LLMProvider(llm_provider.replace("-", "_"))
        if llm_model:
            cfg.model = llm_model
        if llm_api_key:
            cfg.api_key = llm_api_key
        if llm_base_url:
            cfg.base_url = llm_base_url

        if not question:
            query_repl(index_dir, llm_config=cfg, max_iterations=max_iterations, verbose=verbose)
            return

        typer.echo(f"[Query Agent] {question}", err=True)
        res = run_query_agent(
            index_dir,
            question,
            llm_config=cfg,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        if res.error:
            typer.echo(f"Query agent failed: {res.error}", err=True)
            raise typer.Exit(code=1)
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "question": question,
                        "source": "query_agent",
                        "answer": res.answer,
                        "analysis": res.analysis,
                        "tool_calls": res.tool_calls,
                        "evidence": res.evidence,
                        "trace": res.trace,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo("")
            typer.echo(res.answer)
            typer.echo("")
            typer.echo(
                f"(source: query_agent | tools: {len(res.tool_calls)} | evidence: {res.evidence.get('items')})"
            )
        return

    if question is None:
        typer.echo("Error: --question is required (or use --agent/--interactive).", err=True)
        raise typer.Exit(code=2)

    # ---- classic mode (deterministic routing + GraphRAG search) -----------
    from code_graphrag.query.graphrag_search import graphrag_search
    from code_graphrag.query.queries import CodeGraphQuery
    from code_graphrag.query.routing import (
        GRAPHRAAG_MODES,
        detect_deterministic_mode,
        extract_path_symbols,
        extract_symbol,
    )

    q = CodeGraphQuery(index_dir)
    detected = mode or detect_deterministic_mode(question)

    # deterministic path (no LLM)
    if detected in (
        "where",
        "callers",
        "callees",
        "imports",
        "imported_by",
        "inherits",
        "config",
        "files",
        "path",
    ):
        if detected == "path":
            a, b = extract_path_symbols(question)
            result = q.call_path(
                a or extract_symbol(question, detected),
                b or extract_symbol(question, detected),
            )
        else:
            handlers = {
                "where": q.where_is,
                "callers": q.callers,
                "callees": q.callees,
                "imports": q.imports_of,
                "imported_by": q.imported_by,
                "inherits": q.inherits,
                "config": q.config_affects,
                "files": q.files_for,
            }
            result = handlers[detected](extract_symbol(question, detected))
        _emit(question, result, json_output, source=f"code_graph:{detected}")
        return

    # LLM / GraphRAG semantic search
    st = search_type if detected is None else detected
    if st not in GRAPHRAAG_MODES:
        st = "local"
    typer.echo(f"[GraphRAG {st} search] {question}", err=True)
    res = graphrag_search(index_dir, question, search_type=st, community_level=community_level)
    if res.error:
        typer.echo(f"GraphRAG search failed: {res.error}", err=True)
        raise typer.Exit(code=1)
    _emit(
        question,
        type(
            "_R",
            (),
            {
                "answer": res.answer,
                "entities": [],
                "relations": [],
                "path": [],
                "reasoning": res.reasoning,
            },
        )(),
        json_output,
        source=f"graphrag:{st}",
    )


@app.command()
def explore(
    index_dir: str = typer.Option(..., "--index", "-i", help="Index directory."),
    question: str = typer.Option(..., "--question", "-q", help="Open-ended question to explore."),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Source repo root for the agent's filesystem tools."
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="openai|openai_compatible|anthropic|ollama|mock"
    ),
    llm_model: str | None = typer.Option(None, "--llm-model", help="Completion model name."),
    llm_api_key: str | None = typer.Option(
        None, "--llm-api-key", envvar="CODE_GRAPHRAG_LLM_API_KEY", help="LLM API key (or env)."
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        envvar="CODE_GRAPHRAG_LLM_BASE_URL",
        help="LLM base URL (OpenAI-compatible).",
    ),
    mock: bool = typer.Option(False, "--mock", help="Fully offline mock LLM (canned answers)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Run an open-ended code-exploration agent (Deep Agents) over the index."""
    setup_logging("DEBUG" if verbose else "INFO")
    from code_graphrag.agents import run_code_agent
    from code_graphrag.config.models import LLMConfig, LLMProvider

    cfg = LLMConfig()
    if mock:
        cfg.provider = LLMProvider.MOCK
    if llm_provider:
        cfg.provider = LLMProvider(llm_provider.replace("-", "_"))
    if llm_model:
        cfg.model = llm_model
    if llm_api_key:
        cfg.api_key = llm_api_key
    if llm_base_url:
        cfg.base_url = llm_base_url
    if cfg.provider is LLMProvider.MOCK:
        typer.echo("NOTE: LLM provider is 'mock' (offline; canned answers).", err=True)
    typer.echo(f"[Deep Agents explore] {question}", err=True)
    answer = run_code_agent(index_dir, question, source_dir=source, llm_config=cfg)
    typer.echo("")
    typer.echo(answer)


def _emit(question: str, result, json_output: bool, source: str) -> None:
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "question": question,
                    "source": source,
                    "answer": result.answer,
                    "entities": [
                        {
                            "name": e.get("name"),
                            "qualified_name": e.get("qualified_name"),
                            "type": e.get("type"),
                            "file": e.get("file"),
                            "location": e.get("location"),
                        }
                        for e in result.entities
                    ],
                    "path": result.path,
                    **({"reasoning": result.reasoning} if getattr(result, "reasoning", "") else {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        typer.echo("")
        typer.echo(result.answer)
        typer.echo("")
        typer.echo(f"(source: {source})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
