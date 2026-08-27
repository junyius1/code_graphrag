"""Tests for the LangGraph Query Agent (graphrag_query package).

Runs fully offline against the session-scoped mock index (tests/sample_project
-> mock LLM + embeddings). A ``ScriptedChatModel`` drives the agent's
tool-calling loop deterministically so multi-round, duplicate-call and
iteration-cap behavior are testable without a real model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from graphrag_query.agent import (
    _tool_signature,
    heuristic_analysis,
    run_query_agent,
)
from graphrag_query.context import ContextManager, EvidenceItem
from graphrag_query.data_access import QueryData
from graphrag_query.trace import Tracer

# --------------------------------------------------------------------------- #
# stub model
# --------------------------------------------------------------------------- #


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model: replays a script of agent steps.

    Each step is either:
    - a list of (tool_name, args) tuples -> emitted as tool_calls; or
    - a string -> emitted as the final text answer (no tool calls).
    Steps are consumed one per model invocation (agent node and synthesize
    both count; the last step should be the synthesis answer text).
    """

    script: list[Any]
    calls: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("scripted model: no structured output")

    @property
    def _identifying_params(self) -> dict:
        return {"script_len": len(self.script)}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        step = self.script.pop(0) if self.script else "Done."
        if isinstance(step, list):
            calls = [
                {
                    "name": name,
                    "args": args,
                    "id": f"call_{len(self.calls)}_{i}",
                    "type": "tool_call",
                }
                for i, (name, args) in enumerate(step)
            ]
            msg = AIMessage(content="", tool_calls=calls)
        else:
            msg = AIMessage(content=step)
        self.calls.append(msg)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _mock_index() -> str:
    return "mock"


# --------------------------------------------------------------------------- #
# QueryData (real index data access)
# --------------------------------------------------------------------------- #


def test_querydata_load(built_index):
    d = QueryData.load(built_index)
    assert len(d.entities_df) > 0
    assert len(d.sidecar_entities) == len(d.entities_df)
    # every sidecar graphrag_title maps to a parquet entity
    for e in d.sidecar_entities[:20]:
        t = e.get("graphrag_title") or e["qualified_name"]
        assert t in set(d.entities_df["title"].tolist())


def test_querydata_find_and_snippets(built_index):
    d = QueryData.load(built_index)
    matches = d.find_sidecar_entities("handle")
    assert matches and matches[0]["qualified_name"].endswith("handle")
    best = matches[0]
    title = best.get("graphrag_title") or best["qualified_name"]
    snips = d.source_snippets_for(title)
    assert snips, "expected at least one source snippet"
    assert snips[0].file == best["file"]
    assert snips[0].start_line is not None
    # source search + doc search
    assert d.search_source("handle")
    docs = d.search_documentation("service")
    assert isinstance(docs, list)


def test_querydata_relationships(built_index):
    d = QueryData.load(built_index)
    rels = d.relations_between("App", "handle")
    assert rels, "expected a DEFINES relation between App and App.handle"
    assert rels[0]["type"] == "DEFINES"
    assert rels[0]["target_entity"]["location"].startswith("app.py:")


# --------------------------------------------------------------------------- #
# ContextManager
# --------------------------------------------------------------------------- #


def test_context_dedup_and_budget():
    ctx = ContextManager(budget_chars=500)
    a = EvidenceItem(source="t", kind="symbol", title="X", content="A" * 100, weight=2)
    b = EvidenceItem(source="t", kind="symbol", title="X", content="A" * 100, weight=1)
    c = EvidenceItem(source="t", kind="source", title="Y", content="B" * 400, weight=1)
    assert ctx.add(a)
    assert not ctx.add(b)  # duplicate (same kind+title+content)
    assert ctx.add(c)
    assert ctx.summary()["duplicates"] == 1
    rendered = ctx.rendered()
    # budget respected (allow marker overhead)
    assert len(rendered) <= 500 + 80
    assert "truncated" in rendered or len(rendered) <= 500


# --------------------------------------------------------------------------- #
# heuristic analysis
# --------------------------------------------------------------------------- #


def test_heuristic_analysis_types():
    assert (
        heuristic_analysis("Where is build_index implemented?")["question_type"] == "code_location"
    )
    assert heuristic_analysis("Who calls build_index?")["question_type"] == "callers"
    assert heuristic_analysis("What does build_index call?")["question_type"] == "callees"
    assert (
        heuristic_analysis("What does the indexing module import?")["question_type"]
        == "module_dependency"
    )
    assert heuristic_analysis("What class inherits from Base?")["question_type"] == "inheritance"
    assert (
        heuristic_analysis("What is the overall architecture?")["question_type"] == "architecture"
    )
    sym = heuristic_analysis("Where is build_index implemented?")["symbols"]
    assert "build_index" in sym


# --------------------------------------------------------------------------- #
# agent end-to-end (offline, scripted model)
# --------------------------------------------------------------------------- #


def _run(built_index, script, max_iterations=6, question="App.handle 是在哪里实现的？"):
    model = ScriptedChatModel(script=script)
    res = run_query_agent(
        built_index, question, model=model, max_iterations=max_iterations, verbose=False
    )
    assert res.error is None, res.error
    return res


def test_agent_code_location(built_index):
    # locate, then fetch source, then finalize
    res = _run(
        built_index,
        [
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            [("fetch_source_snippets", {"symbol": "handle", "max_snippets": 2})],
            "App.handle is defined in app.py:18-26.",
            "App.handle is implemented in app.py lines 18-26.",
        ],
    )
    assert [c["name"] for c in res.tool_calls] == ["search_code_symbol", "fetch_source_snippets"]
    assert res.analysis.get("question_type") == "code_location"
    assert "app.py" in res.answer
    assert res.evidence["items"] >= 1
    # trace contains the structured events (no raw CoT)
    steps = [e["step"] for e in res.trace]
    assert (
        "question" in steps and "analysis" in steps and "tool_call" in steps and "answer" in steps
    )


def test_agent_callers(built_index):
    res = _run(
        built_index,
        [
            [("callers", {"symbol": "handle_register"})],
            "The caller is App.handle at app.py:18-26.",
            "App.handle (app.py:18-26) calls App.handle_register (app.py:28-31).",
        ],
        question="谁调用了 App.handle_register？",
    )
    assert [c["name"] for c in res.tool_calls] == ["callers"]
    assert res.analysis.get("question_type") == "callers"


def test_agent_callees_and_multi_round(built_index):
    res = _run(
        built_index,
        [
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            [("callees", {"symbol": "handle"})],
            [("fetch_source_snippets", {"symbol": "handle", "max_snippets": 1})],
            "It calls handle_register/handle_lookup/handle_notify.",
            "App.handle (app.py:18-26) calls three handlers in app.py.",
        ],
        question="App.handle 会调用哪些函数？",
    )
    names = [c["name"] for c in res.tool_calls]
    assert names == ["search_code_symbol", "callees", "fetch_source_snippets"]
    assert res.evidence["items"] >= 2


def test_agent_module_dependency(built_index):
    res = _run(
        built_index,
        [
            [("imports_of", {"symbol": "app"})],
            "app imports models and services.",
            "Module app imports: models, services (app.py).",
        ],
        question="app 模块依赖哪些模块？",
    )
    assert [c["name"] for c in res.tool_calls] == ["imports_of"]
    assert res.analysis.get("question_type") == "module_dependency"


def test_agent_architecture_uses_global(built_index):
    res = _run(
        built_index,
        [
            [("code_overview", {})],
            [
                (
                    "global_graph_search",
                    {"question": "what is the architecture?", "community_level": 1},
                )
            ],
            "The repo is a small HTTP service.",
            "The sample project is a minimal HTTP user service (app.py + services.py).",
        ],
        question="这个项目的整体架构是什么？",
    )
    names = [c["name"] for c in res.tool_calls]
    assert "global_graph_search" in names
    assert res.analysis.get("question_type") == "architecture"


def test_agent_documentation(built_index):
    res = _run(
        built_index,
        [
            [("search_documentation", {"query": "service", "limit": 2})],
            "The README documents the service.",
            "Documented in README.md (see Overview / Modules sections).",
        ],
        question="这个 service 的官方文档在哪里？",
    )
    assert [c["name"] for c in res.tool_calls] == ["search_documentation"]
    assert res.analysis.get("question_type") == "documentation"


def test_agent_no_result_question(built_index):
    res = _run(
        built_index,
        [
            [("search_code_symbol", {"symbol": "xyz_not_exist", "symbol_type": ""})],
            [("search_source", {"query": "xyz_not_exist", "limit": 2})],
            "Not found.",
            "No sufficient evidence in the index for 'xyz_not_exist'; the symbol was not found.",
        ],
        question="xyz_not_exist 是什么？",
    )
    assert "not found" in res.answer.lower() or "no sufficient evidence" in res.answer.lower()
    # tool results recorded as evidence of absence
    assert res.evidence["items"] >= 0


def test_agent_duplicate_calls_not_rerun(built_index):
    # the model repeats the exact same tool call 3 times, then answers
    res = _run(
        built_index,
        [
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            "Repeated calls return the cached result.",
            "App.handle is defined in app.py:18-26 (verified once; calls 2-3 were duplicates).",
        ],
    )
    assert len(res.tool_calls) == 3
    # exactly one distinct signature
    sigs = {_tool_signature(c["name"], c["args"]) for c in res.tool_calls}
    assert len(sigs) == 1
    # duplicate events recorded in the trace
    steps = [e["step"] for e in res.trace]
    assert steps.count("duplicate_skip") >= 1


def test_agent_iteration_cap_stops(built_index):
    # model never stops calling tools; the cap must force a final answer
    res = _run(
        built_index,
        [
            [("search_code_symbol", {"symbol": "handle", "symbol_type": ""})],
            [("search_code_symbol", {"symbol": "handle", "symbol_type": "function"})],
            [("search_entity", {"name": "app", "entity_type": ""})],
            "Forced final: App.handle is in app.py:18-26.",
        ],
        max_iterations=2,
    )
    # the loop terminated despite the model never voluntarily stopping
    assert res.error is None
    assert res.answer
    assert len(res.tool_calls) <= 3


def test_agent_unknown_tool_gets_feedback(built_index):
    # an invalid tool name must not crash the loop; the model gets a canned
    # ToolMessage and can conclude.
    res = _run(
        built_index,
        [
            [("no_such_tool", {"x": 1})],
            "The tool does not exist; falling back to the index overview.",
            "No sufficient evidence for this question.",
        ],
        question="random nonsense question?",
    )
    assert res.error is None
    assert res.answer


def test_agent_context_budget_bounded(built_index):
    # many source snippets -> the synthesis prompt must stay within budget
    model = ScriptedChatModel(
        script=[
            [("search_source", {"query": "handle", "limit": 4})],
            [("fetch_source_snippets", {"symbol": "handle", "max_snippets": 4})],
            "OK",
        ]
    )
    from graphrag_query.agent import build_query_graph
    from graphrag_query.context import ContextManager

    d = QueryData.load(built_index)
    ctx = ContextManager(budget_chars=1200)
    tracer = Tracer(verbose=False)
    graph = build_query_graph(d, model, ctx, tracer, max_iterations=4)
    from langchain_core.messages import HumanMessage

    graph.invoke({"messages": [HumanMessage(content="where is handle?")]})
    rendered = ctx.rendered()
    assert len(rendered) <= 1200 + 80


def test_tool_signatures():
    a = _tool_signature("t", {"x": 1, "y": 2})
    b = _tool_signature("t", {"y": 2, "x": 1})
    c = _tool_signature("t", {"x": 1, "y": 3})
    assert a == b
    assert a != c


def test_tracer_no_cot_leak(built_index):
    res = _run(
        built_index,
        [
            [("code_overview", {})],
            "OK",
            "Sample service overview recorded.",
        ],
    )
    blob = str(res.trace).lower()
    # only structured events, bounded payload
    for ev in res.trace:
        assert len(ev["payload"]) <= 600
        assert ev["step"] in {
            "question",
            "analysis",
            "tool_call",
            "tool_result",
            "duplicate_skip",
            "synthesis",
            "answer",
            "error",
        }
    assert "reasoning_content" not in blob
