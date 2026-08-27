"""Interactive Query Agent REPL.

``query_repl`` runs one full agent turn (analyze -> tools -> synthesize) per
question and threads the previous answer back in, so follow-ups like
"what does its first caller do?" stay grounded in the conversation.
"""

from __future__ import annotations

import sys
from typing import Any

from code_graphrag.config.models import LLMConfig
from code_graphrag.graphrag_query.agent import run_query_agent
from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

_BANNER = (
    "Code GraphRAG Query Agent (interactive)\n"
    "Ask questions about the indexed repository, e.g.\n"
    "  build_index 是在哪里实现的？\n"
    "  谁调用了 build_index？\n"
    "  GraphRAG 的索引构建流程是什么？\n"
    "Commands: /exit or /quit to leave, /trace to print the last trace.\n"
)


def query_repl(
    index_dir: str,
    llm_config: LLMConfig | None = None,
    model: Any | None = None,
    max_iterations: int = 6,
    verbose: bool = False,
) -> None:
    """Run the interactive query loop (blocking; reads stdin)."""
    print(_BANNER, file=sys.stderr)
    print(f"index: {index_dir}", file=sys.stderr)

    history: list[str] = []
    last_trace: list[dict[str, Any]] = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if not question:
            continue
        low = question.lower()
        if low in {"/exit", "/quit", "exit", "quit"}:
            return
        if low == "/trace":
            for ev in last_trace[-20:]:
                print(f"  [{ev['step']}] {ev['payload']}", file=sys.stderr)
            continue

        # thread the immediately-previous exchange for follow-up questions
        turn_question = question
        if history:
            turn_question = (
                "Previous question: " + history[-2] + "\n"
                "Previous answer: " + history[-1] + "\n\n"
                "New question: " + question
            )
        res = run_query_agent(
            index_dir,
            turn_question,
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        last_trace = res.trace
        if res.error:
            print(f"(error: {res.error})", file=sys.stderr)
        else:
            print()
            print(res.answer)
            print()
        history.append(question)
        history.append(res.answer or "(no answer)")
        history = history[-2:]
