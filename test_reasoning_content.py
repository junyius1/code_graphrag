#!/usr/bin/env python3
"""Probe whether a local vLLM-served Qwen3 model fills
``additional_kwargs["reasoning_content"]`` when called through LangChain /
LangGraph.

What this script checks, in five layers:

  A. RAW OpenAI client              -> ground truth: what does the vLLM server
                                       actually return? (field name, present?)
  B. LangChain ChatOpenAI.invoke    -> does vanilla langchain_openai surface
                                       the reasoning into additional_kwargs?
  C. LangGraph StateGraph node      -> same, inside a compiled LangGraph graph
                                       (the repo's real usage path).
  D. LangChain streaming            -> same, on stream chunks.
  E. LangChain + LangGraph AFTER    -> after install_vllm_reasoning_patch(),
     install_vllm_reasoning_patch()  does reasoning land in
                                       additional_kwargs["reasoning_content"]?

Run:
    VLLM_BASE_URL=http://localhost:8000/v1 \
    VLLM_MODEL=/mnt/D/models/Qwen3.8-27B \
    python test_reasoning_content.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "/mnt/D/models/Qwen3.8-27B")
PROMPT = "先简短思考，再用一句话回答：9+9等于几？"

# vLLM/Qwen3 chat template knob that turns on the thinking pass.
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


def _banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _verdict(value: object) -> str:
    return f"FILLED (len={len(value)})" if isinstance(value, str) and value else "NOT filled"


def _summarize_additional_kwargs(ai) -> None:
    ak = ai.additional_kwargs or {}
    print(f"  additional_kwargs keys   : {list(ak.keys())}")
    for key in ("reasoning_content", "reasoning"):
        print(f"    - {key:18s} present={key in ak!s:5s} value={_verdict(ak.get(key))}")


def a_raw_openai_client() -> None:
    """Ground truth: hit /v1/chat/completions directly, ignore LangChain."""
    _banner("A. RAW OPENAI CLIENT (ground truth, no LangChain)")
    import openai

    client = openai.OpenAI(base_url=BASE_URL, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=400,
        extra_body=EXTRA_BODY,
    )
    msg = resp.choices[0].message
    raw = msg.model_dump()
    print(f"  content            : {repr(msg.content)[:100]}")
    print(f"  raw message keys   : {list(raw.keys())}")
    for key in ("reasoning", "reasoning_content"):
        print(f"  raw {key:18s} : {_verdict(raw.get(key))}")
    print("  -> server emits reasoning via the 'reasoning' field (not 'reasoning_content')")


def _make_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key="EMPTY",
        temperature=0.1,
        max_tokens=400,
        extra_body=EXTRA_BODY,
    )


def b_langchain_direct() -> None:
    _banner("B. LANGCHAIN (vanilla ChatOpenAI.invoke)")
    ai = _make_llm().invoke([{"role": "user", "content": PROMPT}])
    print(f"  content: {repr(ai.content)[:100]}")
    _summarize_additional_kwargs(ai)
    print(f"  VERDICT: additional_kwargs['reasoning_content'] {_verdict(ai.additional_kwargs.get('reasoning_content'))}")


def _langgraph_invoke(llm, prompt: str):
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def chat(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()
    return app.invoke({"messages": [HumanMessage(content=prompt)]})["messages"][-1]


def c_langgraph() -> None:
    _banner("C. LANGGRAPH (vanilla StateGraph node invoking the model)")
    ai = _langgraph_invoke(_make_llm(), PROMPT)
    print(f"  content: {repr(ai.content)[:100]}")
    _summarize_additional_kwargs(ai)
    print(f"  VERDICT: additional_kwargs['reasoning_content'] {_verdict(ai.additional_kwargs.get('reasoning_content'))}")


def d_langchain_stream() -> None:
    _banner("D. LANGCHAIN STREAMING (vanilla .stream -> chunk.additional_kwargs)")
    llm = _make_llm()
    saw_rc, n_chunks, total = False, 0, 0
    for chunk in llm.stream([{"role": "user", "content": PROMPT}]):
        n_chunks += 1
        v = (chunk.additional_kwargs or {}).get("reasoning_content")
        if isinstance(v, str):
            saw_rc = True
            total += len(v)
    print(f"  chunks received: {n_chunks}; chunks with reasoning_content: {saw_rc}; total chars: {total}")
    print(f"  VERDICT: reasoning on stream chunks: {'YES' if saw_rc else 'NO (dropped by langchain_openai)'}")


def e_langchain_with_patch() -> None:
    """After install_vllm_reasoning_patch(): reasoning must be in additional_kwargs."""
    _banner("E. LANGCHAIN + LANGGRAPH AFTER install_vllm_reasoning_patch()")
    from code_graphrag.llm.vllm_reasoning_patch import install_vllm_reasoning_patch

    install_vllm_reasoning_patch()

    llm = _make_llm()
    ai = llm.invoke([{"role": "user", "content": PROMPT}])
    print("  [direct invoke]")
    print(f"  content: {repr(ai.content)[:100]}")
    _summarize_additional_kwargs(ai)
    print(f"  VERDICT direct: reasoning_content {_verdict(ai.additional_kwargs.get('reasoning_content'))}")

    g_ai = _langgraph_invoke(llm, PROMPT)
    print("  [langgraph node]")
    _summarize_additional_kwargs(g_ai)
    print(f"  VERDICT graph : reasoning_content {_verdict(g_ai.additional_kwargs.get('reasoning_content'))}")
    g_rc = g_ai.additional_kwargs.get("reasoning_content")
    if isinstance(g_rc, str):
        print(f"  reasoning preview: {g_rc[:200]!r}")

    saw_rc, n_chunks, total = False, 0, 0
    for chunk in llm.stream([{"role": "user", "content": PROMPT}]):
        n_chunks += 1
        v = (chunk.additional_kwargs or {}).get("reasoning_content")
        if isinstance(v, str):
            saw_rc = True
            total += len(v)
    print(f"  [stream] chunks={n_chunks}, chunks with reasoning_content={saw_rc}, total reasoning chars={total}")


if __name__ == "__main__":
    a_raw_openai_client()
    b_langchain_direct()
    c_langgraph()
    d_langchain_stream()
    e_langchain_with_patch()
    _banner("SUMMARY")
    print("  A: vLLM server emits CoT in field 'reasoning' (never 'reasoning_content')")
    print("  B/C/D: vanilla langchain_openai DROPS it -> additional_kwargs = {refusal}")
    print("  E: install_vllm_reasoning_patch() -> additional_kwargs['reasoning_content'] is FILLED")
    print("     (src/code_graphrag/llm/vllm_reasoning_patch.py)")
