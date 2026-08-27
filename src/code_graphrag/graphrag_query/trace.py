"""Structured trace of the Query Agent.

Logs the *decisions* (question analysis, retrieval plan, chosen tool, args,
result summary) and the final answer. It deliberately does NOT capture or
print the model's hidden chain-of-thought — only the structured reasoning
summary produced by the question-analysis step and the tool trail
(requirement: debug/trace without leaking private reasoning).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


def _short(text: str, limit: int = 240) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " …"


@dataclass
class Tracer:
    """Collects a human-readable trace of one agent run (stderr-friendly)."""

    verbose: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    _print: bool = True

    def _emit(self, step: str, payload: str) -> None:
        self.events.append({"step": step, "payload": payload})
        logger.debug("AGENT %s: %s", step, payload)
        if self.verbose and self._print:
            import sys

            print(f"  [{step}] {payload}", file=sys.stderr)

    def question(self, q: str) -> None:
        self._emit("question", _short(q, 300))

    def analysis(self, analysis: dict[str, Any]) -> None:
        self._emit("analysis", json.dumps(analysis, ensure_ascii=False, default=str))

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._emit(
            "tool_call", f"{name}({_short(json.dumps(args, ensure_ascii=False, default=str))})"
        )

    def tool_result(self, name: str, n_items: int, preview: str) -> None:
        self._emit("tool_result", f"{name} -> {n_items} items; {_short(preview)}")

    def duplicate(self, name: str, args: dict[str, Any]) -> None:
        self._emit(
            "duplicate_skip", f"{name}({_short(json.dumps(args, ensure_ascii=False, default=str))})"
        )

    def synthesis(self, evidence: dict[str, Any]) -> None:
        self._emit(
            "synthesis",
            f"{evidence.get('items', 0)} evidence items "
            f"(dupes skipped: {evidence.get('duplicates', 0)})",
        )

    def answer(self, answer: str) -> None:
        self._emit("answer", _short(answer, 300))

    def to_dict(self) -> list[dict[str, Any]]:
        return self.events
