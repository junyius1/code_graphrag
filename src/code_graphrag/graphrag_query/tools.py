"""Controlled retrieval tools for the Query Agent.

The LLM chooses tools via standard LangChain tool calling — it never writes
raw SQL / pandas queries. Each tool validates its arguments, hits the
pre-built index (deterministic sidecar graph, GraphRAG parquet tables, or
Microsoft GraphRAG's own search APIs), returns a compact structured result,
and records an ``EvidenceItem`` for the final synthesis step.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from code_graphrag.graphrag_query.context import ContextManager, EvidenceItem
from code_graphrag.graphrag_query.data_access import QueryData
from code_graphrag.graphrag_query.trace import Tracer


def _result(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _sym(e: dict) -> dict:
    """Compact symbol record for LLM consumption (always citation-bearing)."""
    return {
        "name": e.get("qualified_name") or e.get("name"),
        "type": e.get("type"),
        "file": e.get("file"),
        "location": e.get("location"),
        "description": (e.get("description") or "")[:300],
    }


def build_tools(
    data: QueryData,
    ctx: ContextManager,
    tracer: Tracer,
    on_call: Callable[[str, dict, int, str], None] | None = None,
) -> list[Any]:
    """Build the agent's tool list bound to ``data`` / ``ctx`` / ``tracer``."""

    def record(name: str, args: dict, items: list[EvidenceItem], preview: str) -> None:
        ctx.add_many(items)
        tracer.tool_result(name, len(items), preview)
        if on_call:
            on_call(name, args, len(items), preview)

    @tool
    def search_code_symbol(symbol: str, symbol_type: str = "") -> str:
        """Locate a code symbol (function/method/class) in the index.

        Returns its file, line range, docstring, and a summary of its
        callers/callees/definitions. Prefer this for "where is X
        implemented?" questions.

        Args:
            symbol: the symbol name, e.g. "build_index" or "App.handle".
            symbol_type: optional filter: function|method|class|api.
        """
        types = (
            (symbol_type.lower(),)
            if symbol_type in ("function", "method", "class", "api")
            else None
        )
        matches = data.find_sidecar_entities(symbol, types)
        if not matches:
            record(
                "search_code_symbol",
                {"symbol": symbol, "symbol_type": symbol_type},
                [],
                f"no match for '{symbol}'",
            )
            return _result(
                {"found": False, "symbol": symbol, "note": "symbol not found in the index"}
            )
        out = []
        items = []
        for e in matches[:5]:
            entry = _sym(e)
            entry["callers"] = (
                data.graph._neighbors(e["id"], "CALLS", "in")[:5]
                and [n["qualified_name"] for n in data.graph._neighbors(e["id"], "CALLS", "in")[:5]]
                or []
            )
            entry["callees"] = [
                n["qualified_name"] for n in data.graph._neighbors(e["id"], "CALLS", "out")[:5]
            ]
            out.append(entry)
            items.append(
                EvidenceItem(
                    source="search_code_symbol",
                    kind="symbol",
                    title=e["qualified_name"],
                    content=f"{e['qualified_name']} [{e['type']}] @ {e['location']}\n{e.get('description', '')}",
                    weight=2.0,
                    loc=e["location"],
                )
            )
        record(
            "search_code_symbol",
            {"symbol": symbol, "symbol_type": symbol_type},
            items,
            out[0]["location"],
        )
        return _result({"found": True, "symbol": symbol, "results": out})

    @tool
    def search_entity(name: str, entity_type: str = "") -> str:
        """Find an entity in the GraphRAG knowledge graph (by title/name).

        Returns type, semantic description, degree/frequency, and source
        location when available. Use for concepts beyond plain code symbols
        (modules, files, configurations, documentation).

        Args:
            name: entity name or qualified name.
            entity_type: optional filter, e.g. module|file|function|class|configuration.
        """
        types = (entity_type.lower(),) if entity_type else None
        matches = data.find_entities(name, types)
        if not matches:
            record(
                "search_entity",
                {"name": name, "entity_type": entity_type},
                [],
                f"no match for '{name}'",
            )
            return _result({"found": False, "name": name})
        out, items = [], []
        for ent in matches[:5]:
            full = data.entity_full(ent["title"]) or ent
            entry = {
                "title": ent["title"],
                "type": ent["type"],
                "description": (ent.get("description") or "")[:400],
                "degree": ent.get("degree"),
                "frequency": ent.get("frequency"),
            }
            if full.get("location"):
                entry["location"] = full["location"]
            if full.get("semantic_description"):
                entry["semantic_description"] = full["semantic_description"][:400]
            out.append(entry)
            items.append(
                EvidenceItem(
                    source="search_entity",
                    kind="entity",
                    title=ent["title"],
                    content=f"{ent['title']} [{ent['type']}]\n{ent.get('description', '')}",
                    weight=1.5,
                    loc=full.get("location", ""),
                )
            )
        record("search_entity", {"name": name, "entity_type": entity_type}, items, out[0]["title"])
        return _result({"found": True, "name": name, "results": out})

    @tool
    def search_relationship(source: str, target: str, relationship_type: str = "") -> str:
        """Find relations between two named entities (e.g. who calls what).

        Args:
            source: name of the source entity.
            target: name of the target entity.
            relationship_type: optional filter, e.g. CALLS|IMPORTS|INHERITS|DEFINES|CONTAINS.
        """
        rels = data.relations_between(source, target, relationship_type or None)
        items = [
            EvidenceItem(
                source="search_relationship",
                kind="relationship",
                title=f"{source} -> {target}",
                content=(
                    f"{r['source_entity']['qualified_name']} --[{r['type']}]--> "
                    f"{r['target_entity']['qualified_name']} @ "
                    f"{r['target_entity'].get('location', '')}\n{r.get('description', '')}"
                ),
                weight=2.0,
                loc=r["target_entity"].get("location", ""),
            )
            for r in rels
        ]
        payload = {
            "found": bool(rels),
            "source": source,
            "target": target,
            "results": [
                {
                    "from": r["source_entity"]["qualified_name"],
                    "to": r["target_entity"]["qualified_name"],
                    "type": r["type"],
                    "description": r.get("description"),
                    "location": r["target_entity"].get("location"),
                }
                for r in rels
            ],
        }
        record(
            "search_relationship",
            {"source": source, "target": target, "relationship_type": relationship_type},
            items,
            payload["results"][0]["to"] if rels else "none",
        )
        return _result(payload)

    @tool
    def callers(symbol: str) -> str:
        """List the direct callers of a function/method (incoming CALLS edges)."""
        r = data.graph.callers(symbol)
        items = [
            EvidenceItem(
                source="callers",
                kind="graph",
                title=f"callers({symbol})",
                content=r.answer,
                weight=2.0,
            )
        ]
        record("callers", {"symbol": symbol}, items, _preview(r.answer))
        return _result(
            {"symbol": symbol, "answer": r.answer, "entities": [_sym(e) for e in r.entities]}
        )

    @tool
    def callees(symbol: str) -> str:
        """List what a function/method calls (outgoing CALLS edges)."""
        r = data.graph.callees(symbol)
        items = [
            EvidenceItem(
                source="callees",
                kind="graph",
                title=f"callees({symbol})",
                content=r.answer,
                weight=2.0,
            )
        ]
        record("callees", {"symbol": symbol}, items, _preview(r.answer))
        return _result(
            {"symbol": symbol, "answer": r.answer, "entities": [_sym(e) for e in r.entities]}
        )

    @tool
    def imports_of(symbol: str) -> str:
        """List the modules a module/file imports (outgoing IMPORTS)."""
        r = data.graph.imports_of(symbol)
        items = [
            EvidenceItem(
                source="imports_of",
                kind="graph",
                title=f"imports_of({symbol})",
                content=r.answer,
                weight=1.5,
            )
        ]
        record("imports_of", {"symbol": symbol}, items, _preview(r.answer))
        return _result(
            {"symbol": symbol, "answer": r.answer, "entities": [_sym(e) for e in r.entities]}
        )

    @tool
    def imported_by(symbol: str) -> str:
        """List the modules/files that import a module (incoming IMPORTS)."""
        r = data.graph.imported_by(symbol)
        items = [
            EvidenceItem(
                source="imported_by",
                kind="graph",
                title=f"imported_by({symbol})",
                content=r.answer,
                weight=1.5,
            )
        ]
        record("imported_by", {"symbol": symbol}, items, _preview(r.answer))
        return _result(
            {"symbol": symbol, "answer": r.answer, "entities": [_sym(e) for e in r.entities]}
        )

    @tool
    def inheritance(klass: str) -> str:
        """Show the superclass(es) and subclass(es) of a class (INHERITS)."""
        r = data.graph.inherits(klass)
        items = [
            EvidenceItem(
                source="inheritance",
                kind="graph",
                title=f"inheritance({klass})",
                content=r.answer,
                weight=1.5,
            )
        ]
        record("inheritance", {"klass": klass}, items, _preview(r.answer))
        return _result(
            {"class": klass, "answer": r.answer, "entities": [_sym(e) for e in r.entities]}
        )

    @tool
    def fetch_source_snippets(symbol: str, max_snippets: int = 3) -> str:
        """Fetch the actual source code of a symbol (from its text units).

        Returns raw code with file and line range. Use after locating a
        symbol when the answer needs concrete code (not just a summary).

        Args:
            symbol: qualified or simple symbol name (e.g. "App.handle").
            max_snippets: how many code snippets to return (default 3).
        """
        matches = data.find_sidecar_entities(symbol) or data.find_entities(symbol)
        if not matches:
            record("fetch_source_snippets", {"symbol": symbol}, [], f"no match for '{symbol}'")
            return _result({"found": False, "symbol": symbol})
        snippets = []
        items = []
        for m in matches[:2]:
            title = m.get("graphrag_title") or m.get("title") or m["qualified_name"]
            for s in data.source_snippets_for(title, max_snippets=max_snippets):
                text = s.text
                if len(text) > 2000:
                    text = text[:2000] + "\n... [snip truncated]"
                snippets.append(
                    {
                        "symbol": title,
                        "file": s.file,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "source": text,
                    }
                )
                items.append(
                    EvidenceItem(
                        source="fetch_source_snippets",
                        kind="source",
                        title=f"{title}@{s.file}:{s.start_line}-{s.end_line}",
                        content=text,
                        weight=2.0,
                        loc=f"{s.file}:{s.start_line}-{s.end_line}",
                    )
                )
        if not snippets:
            # fallback: text search for the symbol in code units
            for s in data.search_source(symbol, limit=max_snippets):
                text = s.text[:2000]
                snippets.append(
                    {
                        "symbol": s.symbol,
                        "file": s.file,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "source": text,
                    }
                )
                items.append(
                    EvidenceItem(
                        source="fetch_source_snippets",
                        kind="source",
                        title=f"{s.file}:{s.start_line}-{s.end_line}",
                        content=text,
                        weight=1.5,
                        loc=f"{s.file}:{s.start_line}-{s.end_line}",
                    )
                )
        record(
            "fetch_source_snippets",
            {"symbol": symbol},
            items,
            snippets[0]["file"] if snippets else "none",
        )
        return _result({"found": bool(snippets), "symbol": symbol, "snippets": snippets})

    @tool
    def search_source(query: str, limit: int = 4) -> str:
        """Full-text search over the indexed source code.

        Args:
            query: a word/phrase that should appear in the code.
            limit: max snippets (default 4).
        """
        hits = data.search_source(query, limit=limit)
        items = [
            EvidenceItem(
                source="search_source",
                kind="source",
                title=f"{h.file}:{h.start_line}-{h.end_line}",
                content=h.text[:2000],
                weight=1.2,
                loc=f"{h.file}:{h.start_line}-{h.end_line}",
            )
            for h in hits
        ]
        record("search_source", {"query": query}, items, hits[0].file if hits else "none")
        return _result(
            {
                "found": bool(hits),
                "query": query,
                "results": [
                    {
                        "file": h.file,
                        "symbol": h.symbol,
                        "location": f"{h.file}:{h.start_line}-{h.end_line}",
                        "source": h.text[:1500],
                    }
                    for h in hits
                ],
            }
        )

    @tool
    def search_documentation(query: str, limit: int = 4) -> str:
        """Search the indexed documentation (markdown/RST) for a phrase.

        Args:
            query: word/phrase expected in the docs.
            limit: max results (default 4).
        """
        hits = data.search_documentation(query, limit=limit)
        items = [
            EvidenceItem(
                source="search_documentation",
                kind="doc",
                title=f"{h['document']}#{h['section']}",
                content=h["content"][:2000],
                weight=1.2,
                loc=h["document"],
            )
            for h in hits
        ]
        record(
            "search_documentation", {"query": query}, items, hits[0]["document"] if hits else "none"
        )
        return _result(
            {
                "found": bool(hits),
                "query": query,
                "results": [
                    {
                        "document": h["document"],
                        "section": h["section"],
                        "content": h["content"][:1500],
                    }
                    for h in hits
                ],
            }
        )

    @tool
    def local_graph_search(question: str) -> str:
        """Semantic local search (Microsoft GraphRAG) over entities + context.

        Uses an LLM to build a focused answer from the nearest entities and
        their text units. Good for "what does X do / how does X work" once the
        key symbols are known.

        Args:
            question: a self-contained sub-question in natural language.
        """
        r = data.local_search(question)
        items = [
            EvidenceItem(
                source="local_graph_search",
                kind="graphrag",
                title=question[:60],
                content=r.answer[:4000],
                weight=1.5,
            )
        ]
        record("local_graph_search", {"question": question}, items, _preview(r.answer))
        return _result({"question": question, "answer": r.answer[:4000]})

    @tool
    def global_graph_search(question: str, community_level: int = 2) -> str:
        """Semantic global search (Microsoft GraphRAG, map-reduce over
        community reports) for whole-repo / architectural questions.

        Slower than local search. Use for "what is the architecture / overall
        flow / how is X organized" questions.

        Args:
            question: a self-contained sub-question in natural language.
            community_level: community hierarchy level to map over (default 2).
        """
        r = data.global_search(question, community_level=community_level)
        items = [
            EvidenceItem(
                source="global_graph_search",
                kind="graphrag",
                title=question[:60],
                content=r.answer[:4000],
                weight=1.5,
            )
        ]
        record(
            "global_graph_search",
            {"question": question, "community_level": community_level},
            items,
            _preview(r.answer),
        )
        return _result({"question": question, "answer": r.answer[:4000]})

    @tool
    def code_overview() -> str:
        """Overview of the indexed repository: entity/relation counts by type.

        Useful as a first step to learn the shape of the codebase.
        """
        ov = data.overview()
        record("code_overview", {}, [], _preview(ov))
        return _result({"overview": ov})

    return [
        search_code_symbol,
        search_entity,
        search_relationship,
        callers,
        callees,
        imports_of,
        imported_by,
        inheritance,
        fetch_source_snippets,
        search_source,
        search_documentation,
        local_graph_search,
        global_graph_search,
        code_overview,
    ]


def _preview(text: str) -> str:
    text = str(text).replace("\n", " ")
    return text[:160]
