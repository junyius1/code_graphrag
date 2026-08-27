"""LangGraph Query Agent over a built Code GraphRAG index.

Workflow (explicit StateGraph, LangGraph 1.x):

    START
      -> analyze        : LLM classifies the question (structured output);
                          a deterministic heuristic is the fallback (offline /
                          when the model lacks structured output)
      -> agent          : LLM with tools bound (standard tool calling)
      -> pre_tools      : duplicate-call guard (same tool + args -> short
                          "already done" message instead of re-running)
      -> tools          : ToolNode executes the chosen controlled tools
      -> agent          : (loop until the LLM stops calling tools)
      -> synthesize     : budgeted evidence -> grounded final answer
      -> END

Context never grows unbounded: tools register ``EvidenceItem`` records in a
``ContextManager`` that de-duplicates and enforces a character budget, and only
the budgeted evidence + a compact tool trail reach the final synthesis prompt.
The trace (``Tracer``) records the analysis, chosen tools, args and result
summaries — never the model's hidden chain-of-thought.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from code_graphrag.config.models import LLMConfig
from code_graphrag.graphrag_query.context import ContextManager
from code_graphrag.graphrag_query.data_access import QueryData
from code_graphrag.graphrag_query.trace import Tracer
from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

QUESTION_TYPES = (
    "code_location",
    "callers",
    "callees",
    "module_dependency",
    "inheritance",
    "call_path",
    "config_impact",
    "feature_explanation",
    "local_understanding",
    "architecture",
    "documentation",
    "unknown",
)

_QUESTION_TYPE_RE = {
    "callers": re.compile(
        r"\b(callers?|who calls|which (?:cli )?command|calls? (?:this|it))\b|谁调?用|被谁调用|调用方",
        re.I,
    ),
    "callees": re.compile(
        r"\b(callees?|calls? (?:what|which)|does .{0,40} call)\b|调用(了|哪些|什么)",
        re.I,
    ),
    "module_dependency": re.compile(
        r"\b(imports?|depend|depends on|dependencies|requires)\b|依赖(哪些|了|什么)?|引用了",
        re.I,
    ),
    "inheritance": re.compile(
        r"\b(inherit\w*|subclass\w*|extends?\b|derived from|superclass)\b|继承",
        re.I,
    ),
    "call_path": re.compile(
        r"\b(call (?:path|chain)|from \w+ (?:to|via) \w+)\b|调用链|调用路径",
        re.I,
    ),
    "config_impact": re.compile(
        r"\b(config(?:uration)? (?:affect|impact)|impact of (?:modifying|changing)|what code is affected)\b|配置.{0,10}影响|修改.{0,30}影响|影响哪些代码",
        re.I,
    ),
    "documentation": re.compile(
        r"\b(doc(?:umentation)?|readme|where is .{0,40} documented|official)\b|文档|说明书",
        re.I,
    ),
    "code_location": re.compile(
        r"\b(where|defined|implemented|located|declared|in which file)\b|在哪|定义在|实现在|实现位置|哪个文件|哪里",
        re.I,
    ),
    "architecture": re.compile(
        r"\b(architecture|overall|end to end|end-to-end|how (?:is|does) .{0,40} (?:work|build|structured))\b|架构|整体|流程|工作原理",
        re.I,
    ),
    "feature_explanation": re.compile(
        r"\b(how (?:does|do|is)|what does|why|explain|describe)\b|是什么|干什么|作用",
        re.I,
    ),
}


_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "is",
    "are",
    "was",
    "with",
    "by",
    "it",
    "this",
    "that",
    "as",
    "be",
    "do",
    "does",
    "did",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "file",
    "line",
    "function",
    "class",
    "method",
    "module",
    "code",
    "command",
    "cli",
    "repo",
}


def heuristic_analysis(question: str) -> dict[str, Any]:
    """Deterministic question analysis (fallback when no usable LLM)."""
    qtype = "unknown"
    for cand in (
        "callers",
        "callees",
        "module_dependency",
        "inheritance",
        "call_path",
        "config_impact",
        "documentation",
    ):
        if _QUESTION_TYPE_RE[cand].search(question):
            qtype = cand
            break
    if qtype == "unknown" and _QUESTION_TYPE_RE["code_location"].search(question):
        qtype = "code_location"
    elif qtype == "unknown" and _QUESTION_TYPE_RE["architecture"].search(question):
        qtype = "architecture"
    elif qtype == "unknown" and _QUESTION_TYPE_RE["feature_explanation"].search(question):
        qtype = "feature_explanation"
    tokens = [t for t in _SYMBOL_RE.findall(question) if t.lower() not in _STOPWORDS and len(t) > 2]
    symbols = [t for t in tokens if ("_" in t or "." in t or re.search(r"[a-z][A-Z]", t))][:5]
    if not symbols:
        symbols = tokens[:3]
    return {
        "question_type": qtype,
        "symbols": symbols,
        "entities": [],
        "confidence": 0.5,
        "rationale": "heuristic routing (no structured LLM available)",
    }


def _analysis_model(model: Any, tracer: Tracer):
    """Bind a question-analysis structured-output model, if the LLM supports it."""
    try:
        from pydantic import BaseModel, Field

        class _QuestionAnalysis(BaseModel):
            question_type: str = Field(description="one of: " + ", ".join(QUESTION_TYPES))
            symbols: list[str] = Field(
                description="code symbols mentioned (functions/classes/modules), as written"
            )
            entities: list[str] = Field(
                description="non-symbol entities (files, configs, doc topics)"
            )
            confidence: float = Field(description="0..1 confidence in the classification")
            rationale: str = Field(description="one-sentence structured justification")

        bound = model.with_structured_output(_QuestionAnalysis)

        def _run(question: str) -> dict[str, Any] | None:
            out = bound.invoke(
                [
                    SystemMessage(
                        content=(
                            "Classify a question about a source repository. Choose exactly one "
                            f"question_type from: {', '.join(QUESTION_TYPES)}. Extract code symbols "
                            "exactly as written. Be terse."
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            if out is None:
                return None
            data = out if isinstance(out, dict) else out.model_dump()
            if data.get("question_type") not in QUESTION_TYPES:
                data["question_type"] = "unknown"
            data.setdefault("symbols", [])
            data.setdefault("entities", [])
            data.setdefault("confidence", 0.5)
            data.setdefault("rationale", "")
            return data

        return _run
    except Exception as exc:  # noqa: BLE001
        logger.debug("structured analysis unavailable (%s); using heuristics", exc)
        return None


_ANALYZE_SYSTEM = """You are a code-knowledge retrieval agent for a specific source repository.
You answer questions ONLY from the evidence your tools return — never from your own
knowledge of the codebase. If the evidence is insufficient, say exactly that.

Tool selection guidance:
- "where is X defined/implemented?"        -> search_code_symbol, then fetch_source_snippets
- "who calls X?" / "which command calls X" -> callers(X), then search_code_symbol on the caller
- "what does X call?"                      -> callees(X)
- "what does module M import / depend on"  -> imports_of(M) / imported_by(M)
- inheritance questions                    -> inheritance(C)
- call path A -> B                         -> search_relationship(A, B, CALLS)
- "what does X do / how does it work"      -> search_code_symbol + search_entity, then
  local_graph_search with a focused sub-question
- whole-repo / architecture / flow         -> global_graph_search (slow) and/or
  code_overview + imports_of for the main modules
- documentation questions                  -> search_documentation, then search_entity
- config impact                            -> search_entity(config) + search_relationship

Rules:
1. Prefer the precise code-graph tools (search_code_symbol/callers/callees) over
   LLM-based searches when a concrete symbol is named.
2. You may call tools several times to build the full picture (e.g. locate a
   symbol, then its callers, then their source). Repeating a tool call with the
   same arguments returns the cached result — do not loop; move on or answer.
3. Once the evidence is sufficient, stop calling tools and state your final
   answer in a short structured summary (this intermediate summary is used to
   guide final synthesis, so cite file:line locations you already have).
4. If searches return no results, report that plainly; do not guess."""


def _memoize_tool(tool: BaseTool, seen: set, tracer: Tracer) -> BaseTool:
    """Wrap a tool so repeated identical calls return the cached result.

    Read-only retrieval tools are idempotent; memoization makes the agent
    loop robust against repeated calls (no re-execution, no unbounded growth)
    while keeping the message protocol valid (every call gets a result).
    """

    cache: dict[str, str] = {}

    def _run(**kwargs: Any) -> str:
        sig = _tool_signature(tool.name, kwargs)
        if sig in cache:
            tracer.duplicate(tool.name, kwargs)
            cached = cache[sig]
            return "[duplicate call - returning the cached result from the earlier call] " + cached
        seen.add(sig)
        result = tool.invoke(kwargs)
        content = result if isinstance(result, str) else str(result)
        cache[sig] = content
        return content

    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _tool_signature(name: str, args: dict[str, Any]) -> str:
    return name + "::" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def build_query_graph(
    data: QueryData,
    model: Any,
    ctx: ContextManager,
    tracer: Tracer,
    max_iterations: int = 6,
) -> Any:
    """Compile the query agent StateGraph for one run (fresh dedup state)."""
    seen_sigs: set[str] = set()
    raw_tools = _build_tools_cached(data, ctx, tracer)
    tools = [_memoize_tool(t, seen_sigs, tracer) for t in raw_tools]
    tool_node = ToolNode(tools)
    tool_names = {t.name for t in tools}
    analysis_fn = _analysis_model(model, tracer)

    def analyze(state: dict) -> dict:
        question = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            str(state["messages"][-1].content),
        )
        tracer.question(question)
        analysis = None
        if analysis_fn is not None:
            try:
                analysis = analysis_fn(question)
            except Exception as exc:  # noqa: BLE001
                logger.warning("analysis LLM failed (%s); falling back to heuristics", exc)
        if not analysis:
            analysis = heuristic_analysis(question)
        tracer.analysis(analysis)
        return {"analysis": analysis, "plan": _plan_hint(analysis)}

    def agent(state: dict) -> dict:
        iteration = state.get("iteration", 0) + 1
        forced = iteration > max_iterations
        forced_final = state.get("forced_final", False) or forced
        # some servers (vLLM) require exactly one leading system message
        plan = state.get("plan") or ""
        system = _ANALYZE_SYSTEM if not plan else _ANALYZE_SYSTEM + "\n\n" + plan
        msgs = [SystemMessage(content=system)] + list(state["messages"])
        if forced and not state.get("forced_final", False):
            msgs = msgs + [
                HumanMessage(
                    content=(
                        "Iteration budget exhausted. Stop calling tools and produce your "
                        "final answer now from the evidence gathered so far."
                    )
                )
            ]
        use_model = model if forced else model.bind_tools(tools)
        out = use_model.invoke(msgs)
        for c in out.tool_calls or []:
            tracer.tool_call(c["name"], c["args"])
        return {
            "iteration": iteration,
            "forced_final": forced_final,
            "tool_calls": list(state.get("tool_calls", []))
            + [{"name": c["name"], "args": c["args"]} for c in (out.tool_calls or [])],
            "messages": [out],
        }

    def pre_tools(state: dict) -> dict:
        """Log tool calls; unknown names get a canned ToolMessage.

        Duplicates are not an error: every tool is memoized, so a repeated
        call returns its cached result cheaply (idempotent, read-only tools).
        """
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return {}
        seen = set(state.get("seen_calls", []))
        out_msgs: list[ToolMessage] = []
        for c in last_ai.tool_calls:
            sig = _tool_signature(c["name"], c["args"])
            if c["name"] not in tool_names:
                out_msgs.append(
                    ToolMessage(
                        content=f"Tool '{c['name']}' does not exist. Available tools: "
                        + ", ".join(sorted(tool_names)),
                        tool_call_id=c["id"],
                        name=c["name"],
                    )
                )
            elif sig in seen:
                tracer.duplicate(c["name"], c["args"])
            seen.add(sig)
        return {"seen_calls": list(seen), **({"messages": out_msgs} if out_msgs else {})}

    def route_after_agent(state: dict) -> str:
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if last_ai is None or not getattr(last_ai, "tool_calls", None):
            return "synthesize"
        return "pre_tools"

    def route_after_pre(state: dict) -> str:
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls),
            None,
        )
        return "tools" if last_ai is not None else "synthesize"

    def synthesize(state: dict) -> dict:
        question = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
        evidence = ctx.summary()
        tracer.synthesis(evidence)
        tool_trail = (
            "\n".join(
                f"- {tc['name']}({json.dumps(tc['args'], ensure_ascii=False, default=str)})"
                for tc in state["tool_calls"]
            )
            or "- (no tools called)"
        )
        synthesis_messages = [
            SystemMessage(
                content=(
                    "You are writing the final answer to a question about a source repository. "
                    "Base the answer ONLY on the EVIDENCE below. Requirements: "
                    "1) every concrete claim must cite the evidence (file:line where the "
                    "evidence has a location); 2) do not invent code, files, or behavior; "
                    "3) if the evidence is insufficient for part of the question, state "
                    "'no sufficient evidence in the index' for that part; 4) structure: a "
                    "direct answer first, then the supporting locations/call chains, then "
                    "related docs if present. Keep it focused."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Question analysis: {json.dumps(state.get('analysis', {}), ensure_ascii=False)}\n\n"
                    f"Tools called by the retrieval agent:\n{tool_trail}\n\n"
                    f"EVIDENCE:\n{ctx.rendered()}\n\n"
                    "Write the final answer now."
                )
            ),
        ]
        out = model.invoke(synthesis_messages)
        answer = out.content if isinstance(out.content, str) else str(out.content)
        if not answer.strip() or getattr(out, "tool_calls", None):
            # the model did not produce a final text (e.g. mock model emitting a
            # tool call): deterministically summarize the gathered evidence so
            # the loop stays testable offline.
            if ctx.items:
                parts = [
                    f"- {it.title}" + (f" [{it.loc}]" if it.loc else "")
                    for it in sorted(ctx.items, key=lambda x: -x.weight)[:10]
                ]
                NL = chr(10)
                answer = (
                    "(offline/mock synthesis) Evidence gathered by the agent:"
                    + NL
                    + NL.join(parts)
                    + NL
                    + "A real LLM model produces the full grounded answer here."
                )
            else:
                answer = (
                    "No sufficient evidence found in the index for this question. "
                    "Try a more specific symbol or check `code-graphrag inspect -i <index>`."
                )
        tracer.answer(answer)
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    g = StateGraph(_QueryState)
    g.add_node("analyze", analyze)
    g.add_node("agent", agent)
    g.add_node("pre_tools", pre_tools)
    g.add_node("tools", tool_node)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "agent")
    g.add_conditional_edges(
        "agent", route_after_agent, {"synthesize": "synthesize", "pre_tools": "pre_tools"}
    )
    g.add_conditional_edges(
        "pre_tools",
        route_after_pre,
        {"tools": "tools", "agent": "agent", "synthesize": "synthesize"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("synthesize", END)
    return g.compile()


class _QueryState(TypedDict, total=False):
    """LangGraph state channels for one query-agent run."""

    messages: Annotated[list[BaseMessage], add_messages]
    analysis: dict
    plan: str
    iteration: int
    forced_final: bool
    tool_calls: list
    seen_calls: list
    final_answer: str


def _plan_hint(analysis: dict[str, Any]) -> str:
    """Short retrieval-plan hint derived from the analysis (logged, not CoT)."""
    hints = {
        "code_location": "Suggested plan: search_code_symbol -> fetch_source_snippets",
        "callers": "Suggested plan: callers(symbol) -> search_code_symbol on the top caller",
        "callees": "Suggested plan: callees(symbol)",
        "module_dependency": "Suggested plan: imports_of / imported_by for the module",
        "inheritance": "Suggested plan: inheritance(class)",
        "call_path": "Suggested plan: search_relationship(from, to, CALLS) or callees/callers chain",
        "config_impact": "Suggested plan: search_entity(config) -> search_relationship on the hits",
        "feature_explanation": "Suggested plan: search_code_symbol -> local_graph_search (focused sub-question)",
        "local_understanding": "Suggested plan: search_code_symbol -> fetch_source_snippets -> local_graph_search",
        "architecture": "Suggested plan: code_overview -> global_graph_search (community_level 1-2)",
        "documentation": "Suggested plan: search_documentation -> search_entity",
        "unknown": "Suggested plan: search_code_symbol / search_entity on the extracted symbols; escalate to local_graph_search",
    }
    qtype = analysis.get("question_type", "unknown")
    symbols = ", ".join(analysis.get("symbols") or []) or "(none extracted)"
    return f"Question analysis: type={qtype}, symbols=[{symbols}].\n" + hints.get(
        qtype, hints["unknown"]
    )


def _build_tools_cached(data: QueryData, ctx: ContextManager, tracer: Tracer) -> list[BaseTool]:
    from code_graphrag.graphrag_query.tools import build_tools

    return build_tools(data, ctx, tracer)


@dataclass
class AgentResult:
    question: str
    answer: str
    analysis: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _resolve_local(cfg: LLMConfig) -> LLMConfig:
    """Translate `local:<path>` model ids to a usable OpenAI-compatible cfg."""
    from code_graphrag.graphrag.local_llm import is_local_model, normalize_model_id

    if is_local_model(cfg.model):
        cfg = cfg.model_copy(
            update={
                "provider": "openai_compatible",
                "model": normalize_model_id(cfg.model),
            }
        )
        if not cfg.api_key:
            cfg.api_key = "sk-no-key-offline"
    return cfg


def run_query_agent(
    index_dir: str,
    question: str,
    llm_config: LLMConfig | None = None,
    model: Any | None = None,
    max_iterations: int = 6,
    budget_chars: int | None = None,
    verbose: bool = False,
    interactive: bool = False,
) -> AgentResult:
    """Run one question through the Query Agent and return the result.

    Args:
        index_dir: directory of a built Code GraphRAG index.
        question: natural-language question.
        llm_config: LLM configuration (supports `local:` model ids).
        model: explicit chat model override (tests / callers).
        max_iterations: cap on agent->tool rounds (loop protection).
        budget_chars: evidence budget for the synthesis prompt.
        verbose: print the structured trace to stderr.
        interactive: keep the model/tools alive for a follow-up REPL
            (used by the CLI; ignored by `run_query_agent` callers).
    """
    from code_graphrag.llm.models import create_chat_model

    cfg = _resolve_local(llm_config or LLMConfig())
    chat_model = model if model is not None else create_chat_model(cfg)
    data = QueryData.load(index_dir)
    ctx = ContextManager(budget_chars=budget_chars or ContextManager().budget_chars)
    tracer = Tracer(verbose=verbose)
    graph = build_query_graph(data, chat_model, ctx, tracer, max_iterations=max_iterations)

    try:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as exc:  # noqa: BLE001
        logger.exception("query agent failed")
        tracer._emit("error", str(exc))
        return AgentResult(
            question=question,
            answer="",
            error=str(exc),
            evidence=ctx.summary(),
            trace=tracer.to_dict(),
        )
    answer = result.get("final_answer") or ""
    if not answer:
        ai = [m for m in result.get("messages", []) if isinstance(m, AIMessage) and m.content]
        answer = ai[-1].content if ai else "(agent produced no answer)"
    return AgentResult(
        question=question,
        answer=answer,
        analysis=result.get("analysis", {}),
        tool_calls=result.get("tool_calls", []),
        evidence=ctx.summary(),
        trace=tracer.to_dict(),
    )
