"""Evidence / context management for the Query Agent.

Retrieval results from the tools are ``EvidenceItem`` records. The
``ContextManager`` de-duplicates, ranks, and trims them to a character budget
so the final synthesis prompt cannot grow unbounded as the agent issues more
retrieval calls (requirement: no infinite context growth).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BUDGET_CHARS = 24000


@dataclass
class EvidenceItem:
    """One retrieval result the agent may cite in the final answer."""

    source: str  # e.g. "search_code_symbol", "search_documentation"
    kind: str  # "symbol" | "entity" | "relationship" | "source" | "doc" | "graph"
    title: str  # short label for dedup + logging
    content: str  # the payload to feed to the final LLM
    weight: float = 1.0  # relevance hint from the tool (higher = keep earlier)
    loc: str = ""  # file:line for display / citation

    @property
    def key(self) -> str:
        """Stable dedup key: kind + normalized title + content fingerprint."""
        fp = hashlib.md5(self.content.encode("utf-8", "replace")).hexdigest()[:10]
        return f"{self.kind}:{self.title.strip().lower()}:{fp}"


@dataclass
class ContextManager:
    """Collects evidence across tool calls; enforces dedup + a char budget."""

    budget_chars: int = DEFAULT_BUDGET_CHARS
    items: list[EvidenceItem] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)
    duplicates: int = 0

    def add(self, item: EvidenceItem) -> bool:
        """Add an item; returns True if it was new, False if duplicate."""
        if item.key in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(item.key)
        self.items.append(item)
        return True

    def add_many(self, items: list[EvidenceItem]) -> int:
        n = 0
        for it in items:
            if self.add(it):
                n += 1
        return n

    def rendered(self, budget_chars: int | None = None) -> str:
        """Render evidence to a budgeted string for the synthesis prompt.

        Higher-weight (earlier) items are kept first; once the budget is
        exceeded, the tail is truncated with a marker rather than silently
        dropped.
        """
        budget = budget_chars or self.budget_chars
        # sort by (weight desc, insertion order) — stable
        order = sorted(range(len(self.items)), key=lambda i: -self.items[i].weight)
        parts: list[str] = []
        used = 0
        for i in order:
            it = self.items[i]
            header = f"[{it.source} | {it.kind} | {it.title}]"
            if it.loc:
                header += f" ({it.loc})"
            block = f"{header}\n{it.content}"
            if used + len(block) > budget:
                room = budget - used - 60
                if room > 200:
                    parts.append(block[:room] + "\n... [truncated by context budget]")
                    used = budget
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    def summary(self) -> dict[str, Any]:
        """Counts for tracing / observability (not the raw CoT)."""
        by_source: dict[str, int] = {}
        for it in self.items:
            by_source[it.source] = by_source.get(it.source, 0) + 1
        return {
            "items": len(self.items),
            "duplicates": self.duplicates,
            "budget_chars": self.budget_chars,
            "by_source": by_source,
        }
