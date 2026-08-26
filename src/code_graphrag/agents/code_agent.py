"""Build a Deep Agents "code exploration" agent over a Code GraphRAG index.

The agent gets:
- the built-in Deep Agents filesystem tools (ls / read_file / glob / grep /
  write_file / edit_file / execute) rooted at the *source* repository, so it can
  read actual code;
- custom tools that query the deterministic code knowledge graph
  (where is X, callers/callees, imports, inheritance, call paths, files,
  config impact, overview) and the GraphRAG semantic index (local/global
  search);
- a ``task`` tool to delegate to an optional "architecture analyst" subagent.

This is intentionally an *optional* layer on top of the deterministic CLI and
LangGraph workflow — it requires an LLM (mock mode works offline but returns
canned text, so it is mainly useful for a real model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from code_graphrag.config.models import LLMConfig
from code_graphrag.logging_setup import get_logger
from code_graphrag.query.queries import CodeGraphQuery

logger = get_logger(__name__)


def build_code_tools(index_dir: str | Path, source_dir: str | Path | None = None) -> list[Any]:
    """Create LangChain tools backed by a built index (+ optional source root)."""
    index_dir = Path(index_dir)
    q = CodeGraphQuery(index_dir)

    @tool
    def code_overview() -> str:
        """Return an overview of the indexed repository: entity/relation counts by type."""
        return q.overview().answer

    @tool
    def where_is(symbol: str) -> str:
        """Find where a symbol (function/class/method) is implemented, with file:line."""
        return q.where_is(symbol).answer

    @tool
    def callers(symbol: str) -> str:
        """List the direct callers of a function/method (incoming CALLS)."""
        return q.callers(symbol).answer

    @tool
    def callees(symbol: str) -> str:
        """List what a function/method calls (outgoing CALLS)."""
        return q.callees(symbol).answer

    @tool
    def imports_of(module: str) -> str:
        """List the modules/files a module imports (outgoing IMPORTS)."""
        return q.imports_of(module).answer

    @tool
    def imported_by(module: str) -> str:
        """List the modules/files that import a module (incoming IMPORTS)."""
        return q.imported_by(module).answer

    @tool
    def inheritance(klass: str) -> str:
        """Show the superclass and subclasses of a class (INHERITS both directions)."""
        return q.inherits(klass).answer

    @tool
    def call_path(from_symbol: str, to_symbol: str) -> str:
        """Shortest call-chain path from one function to another (CALLS edges)."""
        return q.call_path(from_symbol, to_symbol).answer

    @tool
    def related_files(symbol: str) -> str:
        """List the files involved with a symbol, weighted by relation degree."""
        return q.files_for(symbol).answer

    @tool
    def config_impact(config_key: str) -> str:
        """Find code affected by a configuration key/file."""
        return q.config_affects(config_key).answer

    @tool
    def graphrag_local_search(query: str) -> str:
        """Semantic local search over the GraphRAG index (entities + context)."""
        from code_graphrag.query.graphrag_search import graphrag_search

        r = graphrag_search(index_dir, query, search_type="local")
        return r.error or r.answer

    @tool
    def graphrag_global_search(query: str) -> str:
        """Semantic global (map-reduce over communities) search over the index."""
        from code_graphrag.query.graphrag_search import graphrag_search

        r = graphrag_search(index_dir, query, search_type="global")
        return r.error or r.answer

    tools = [
        code_overview,
        where_is,
        callers,
        callees,
        imports_of,
        imported_by,
        inheritance,
        call_path,
        related_files,
        config_impact,
        graphrag_local_search,
        graphrag_global_search,
    ]
    if source_dir:
        logger.info("Code agent will also see filesystem rooted at %s", source_dir)
    return tools


_SYSTEM_PROMPT = """You are a code-exploration assistant for a specific source repository.

You can query a pre-built deterministic code knowledge graph (precise: where-is,
callers/callees, imports, inheritance, call paths, related files, config impact)
and a GraphRAG semantic index (fuzzy: "how does X work", "what is this repo
about"). You also have filesystem tools (ls, read_file, glob, grep) rooted at the
source repository to read the actual code when you need to confirm details.

Strategy:
1. Start with `code_overview` to learn the shape of the repo.
2. For precise "where/who/what" questions, prefer the deterministic graph tools
   (they cite file:line and never hallucinate).
3. For open-ended semantic questions, use `graphrag_local_search` first; if the
   answer is too narrow, escalate to `graphrag_global_search`.
4. Read the actual files (read_file / glob / grep) to verify before making a
   final claim about behavior.

Always ground your answer in the code. Cite `file:line` for concrete facts. If
the index does not contain the symbol, say so rather than guessing."""


def create_code_agent(
    index_dir: str | Path,
    source_dir: str | Path | None = None,
    llm_config: LLMConfig | None = None,
    model: str | Any | None = None,
) -> Any:
    """Build (but do not run) the Deep Agents code-exploration agent.

    Args:
        index_dir: Directory holding a built Code GraphRAG index.
        source_dir: Optional source repo root for the filesystem tools.
        llm_config: LLMConfig used to build the chat model (mock works offline).
        model: Optional explicit model (str "provider:model" or BaseChatModel)
            that overrides ``llm_config``.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from code_graphrag.llm.models import create_chat_model

    tools = build_code_tools(index_dir, source_dir)

    if model is not None:
        resolved_model: Any = model
    else:
        cfg = llm_config or LLMConfig()
        resolved_model = create_chat_model(cfg)

    backend = FilesystemBackend(
        root_dir=Path(source_dir) if source_dir else None, virtual_mode=True
    )

    subagents = [
        {
            "name": "architecture-analyst",
            "description": (
                "Analyzes high-level architecture: module boundaries, data flow, "
                "layering, and cross-cutting concerns. Use for 'how is this "
                "organized' / 'what are the abstractions' questions."
            ),
            "system_prompt": (
                "You are a senior software architect. Use the code graph tools to "
                "map modules and their dependencies, then describe the "
                "architecture, key abstractions, and data flow. Be concrete and "
                "cite files."
            ),
        }
    ]

    return create_deep_agent(
        model=resolved_model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        backend=backend,
        subagents=subagents,
    )


def run_code_agent(
    index_dir: str | Path,
    question: str,
    source_dir: str | Path | None = None,
    llm_config: LLMConfig | None = None,
    model: str | Any | None = None,
) -> str:
    """Run one question through the code agent and return the final answer text."""
    agent = create_code_agent(index_dir, source_dir=source_dir, llm_config=llm_config, model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    ai = [m for m in messages if getattr(m, "type", None) == "ai"]
    return ai[-1].content if ai else ""
