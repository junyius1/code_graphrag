#!/usr/bin/env python3
"""Probe whether a local vLLM-served Qwen3 model fills
``additional_kwargs["reasoning_content"]`` when called through LangChain /
LangGraph.

What this script checks, in three layers:

  A. RAW OpenAI client   -> ground truth: what does the vLLM server actually
                            return? (field name, present/absent)
  B. LangChain ChatOpenAI.invoke  -> does langchain_openai surface the
                            reasoning into ``additional_kwargs``?
  C. LangGraph StateGraph node    -> same, when the model is invoked inside a
                            compiled LangGraph graph (the repo's real path).

For each of B and C we report exactly which keys landed in
``additional_kwargs`` and whether ``reasoning_content`` (and, for vLLM/Qwen3,
the ``reasoning`` alias) is among them.

Run:
    VLLM_BASE_URL=http://localhost:8000/v1 \
    VLLM_MODEL=/mnt/D/models/Qwen3.8-27B \
    python test_reasoning_content.py
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "/mnt/D/models/Qwen3.8-27B")
PROMPT = "先简短思考，再用一句话回答：9+9等于几？"

# vLLM/Qwen3 chat template knob that turns on the thinking pass.
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


def _banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _summarize_additional_kwargs(ai) -> None:
    ak = ai.additional_kwargs or {}
    keys = list(ak.keys())
    print(f"  additional_kwargs keys   : {keys}")
    for key in ("reasoning_content", "reasoning"):
        present = key in ak
        val = ak.get(key)
        preview = (repr(val)[:120] + "...") if isinstance(val, str) and len(val) > 120 else repr(val)
        print(f"    - {key:18s} present={present!s:5s} value={preview}")


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
    print(f"  content               : {repr(msg.content)[:120]}")
    print(f"  raw message keys      : {list(raw.keys())}")
    for key in ("reasoning", "reasoning_content"):
        val = raw.get(key)
        preview = (repr(val)[:120] + "...") if isinstance(val, str) and len(val) > 120 else repr(val)
        print(f"  raw message {key:18s}: {preview}")
    if raw.get("reasoning") or raw.get("reasoning_content"):
        print("  -> server DOES emit reasoning (field name: "
              + ("reasoning_content" if raw.get("reasoning_content") else "reasoning") + ")")
    else:
        print("  -> server did NOT emit any reasoning field for this prompt")


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
    _banner("B. LANGCHAIN  (ChatOpenAI.invoke)")
    ai = _make_llm().invoke([{"role": "user", "content": PROMPT}])
    print(f"  content: {repr(ai.content)[:120]}")
    _summarize_additional_kwargs(ai)
    ok = "reasoning_content" in (ai.additional_kwargs or {})
    print(f"  VERDICT: additional_kwargs['reasoning_content'] "
          f"{'FILLED' if ok else 'NOT filled'}")


def c_langgraph() -> None:
    _banner("C. LANGGRAPH (StateGraph node invoking the model)")
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    llm = _make_llm()

    def chat(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage(content=PROMPT)]})
    ai = result["messages"][-1]
    print(f"  content: {repr(ai.content)[:120]}")
    _summarize_additional_kwargs(ai)
    ok = "reasoning_content" in (ai.additional_kwargs or {})
    print(f"  VERDICT: additional_kwargs['reasoning_content'] "
          f"{'FILLED' if ok else 'NOT filled'}")


def d_langchain_stream() -> None:
    """Streaming path: does reasoning_content appear on message chunks?"""
    _banner("D. LANGCHAIN STREAMING (for_chunks) -> chunk.additional_kwargs")
    llm = _make_llm()
    saw_rc = False
    saw_reasoning = False
    n_chunks = 0
    for chunk in llm.stream([{"role": "user", "content": PROMPT}]):
        n_chunks += 1
        ak = chunk.additional_kwargs or {}
        if "reasoning_content" in ak:
            saw_rc = True
        if "reasoning" in ak:
            saw_reasoning = True
    print(f"  chunks received           : {n_chunks}")
    print(f"  any chunk had 'reasoning_content' in additional_kwargs: {saw_rc}")
    print(f"  any chunk had 'reasoning' in additional_kwargs        : {saw_reasoning}")
    print(f"  VERDICT: reasoning surfaced on stream chunks: "
          f"{'YES' if (saw_rc or saw_reasoning) else 'NO (dropped by langchain_openai)'}")


if __name__ == "__main__":
    a_raw_openai_client()
    b_langchain_direct()
    c_langgraph()
    d_langchain_stream()
    _banner("SUMMARY")
    print("  vLLM server field name     : 'reasoning' (see A)")
    print("  langchain/langgraph        : does NOT populate additional_kwargs['reasoning_content']")
    print("  => the model's reasoning is dropped by langchain_openai's OpenAI-spec parsing.")
    print("     To capture it, intercept the raw delta (see src/code_graphrag/query/reasoning.py).")
