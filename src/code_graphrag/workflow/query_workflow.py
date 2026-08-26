"""LangGraph workflow for answering questions over a Code GraphRAG index.

Nodes:
    classify -> answer -> format

``classify`` maps the question to either a deterministic code-graph query
(precise, no LLM) or a GraphRAG semantic search. ``answer`` dispatches, and
``format`` produces the final answer string. This keeps the two retrieval
strategies behind one graph and makes the routing inspectable/overridable.
"""

from __future__ import annotations

from typing import Any, TypedDict

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


class QueryWorkflowState(TypedDict, total=False):
    index_dir: str
    question: str
    forced_mode: str | None
    route: str
    answer: str
    entities: list[dict]
    path: list[str]
    source: str
    error: str | None


async def classify(state: QueryWorkflowState) -> dict[str, Any]:
    from code_graphrag.query.routing import detect_deterministic_mode

    forced = state.get("forced_mode")
    if forced:
        return {"route": forced}
    return {"route": detect_deterministic_mode(state["question"]) or "local"}


async def answer(state: QueryWorkflowState) -> dict[str, Any]:
    from code_graphrag.query.graphrag_search import agraphrag_search
    from code_graphrag.query.queries import CodeGraphQuery
    from code_graphrag.query.routing import (
        GRAPHRAAG_MODES,
        extract_path_symbols,
        extract_symbol,
    )

    index_dir = state["index_dir"]
    question = state["question"]
    route = state["route"]

    if route in (
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
        q = CodeGraphQuery(index_dir)
        handlers = {
            "where": lambda s: q.where_is(s),
            "callers": lambda s: q.callers(s),
            "callees": lambda s: q.callees(s),
            "imports": lambda s: q.imports_of(s),
            "imported_by": lambda s: q.imported_by(s),
            "inherits": lambda s: q.inherits(s),
            "config": lambda s: q.config_affects(s),
            "files": lambda s: q.files_for(s),
        }
        if route == "path":
            a, b = extract_path_symbols(question)
            sym = extract_symbol(question, route)
            res = q.call_path(a or sym, b or sym)
        else:
            res = handlers[route](extract_symbol(question, route))
        return {
            "answer": res.answer,
            "entities": res.entities,
            "path": res.path,
            "source": f"code_graph:{route}",
        }

    # GraphRAG semantic search
    st = route if route in GRAPHRAAG_MODES else "local"
    res = await agraphrag_search(index_dir, question, search_type=st)
    if res.error:
        return {"answer": "", "source": f"graphrag:{st}", "error": res.error}
    return {"answer": res.answer, "source": f"graphrag:{st}"}


async def format(state: QueryWorkflowState) -> dict[str, Any]:
    if state.get("error"):
        return {"answer": f"Query failed: {state['error']}", "error": state["error"]}
    answer = state.get("answer", "")
    footer = f"\n\n[source: {state.get('source', 'unknown')}]"
    return {"answer": answer + footer}


def build_query_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(QueryWorkflowState)
    builder.add_node("classify", classify)
    builder.add_node("answer", answer)
    builder.add_node("format", format)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "answer")
    builder.add_edge("answer", "format")
    builder.add_edge("format", END)
    return builder.compile()


async def run_query_workflow(
    index_dir: str,
    question: str,
    forced_mode: str | None = None,
) -> dict[str, Any]:
    graph = build_query_graph()
    return await graph.ainvoke(
        {
            "index_dir": index_dir,
            "question": question,
            "forced_mode": forced_mode,
            "route": "",
            "answer": "",
            "entities": [],
            "path": [],
            "source": "",
            "error": None,
        },
        {"configurable": {"thread_id": "code-graphrag-query"}},
    )
