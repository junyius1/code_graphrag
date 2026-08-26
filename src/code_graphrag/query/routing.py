"""Map free-form questions to query routes (deterministic graph vs GraphRAG search)."""

from __future__ import annotations

import re

_DETERMINISTIC_HINTS = {
    "where": ("where", "implemented", "located", "defined"),
    "callers": ("callers", "used by", "invokes it", "calls it"),
    "callees": ("callees", "does it call", "what does .* call"),
    "imports": ("imports", "depends on", "dependencies"),
    "imported_by": ("imported by", "who imports", "what imports"),
    "inherits": ("inherit", "subclass", "extends", "derived from"),
    "config": ("affect", "impact", "which code"),
    "files": ("involve", "related files", "which files", "touch"),
}


def detect_deterministic_mode(question: str) -> str | None:
    """Return a deterministic query mode for the question, or None for LLM search."""
    q = question.lower()

    def has(*words: str) -> bool:
        return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)

    # call path / call chain (from A to B)
    if has("path", "chain") and has("call") or (has("from") and has("to")):
        return "path"
    # callers before callees: "what calls X" -> callers
    if has("caller", "callers") or has("used by", "invokes it", "calls it"):
        return "callers"
    # callees: "what does X call" / "what does X call" (singular or plural)
    if has("callee", "callees") or (has("call", "calls") and has("what", "which", "does it")):
        return "callees"
    # imported_by: "who imports X" / "what imports X"
    if has("imported by") or has("who imports", "what imports", "who import", "what import"):
        return "imported_by"
    # imports: "what does X import" / "dependencies of X"
    if has("import", "imports") or has("depends on", "dependencies", "depend on"):
        return "imports"
    # inherits: "what inherits from X" / "what does X inherit" / subclasses
    if has(
        "inherit",
        "inheritance",
        "inherits",
        "inherited",
        "subclass",
        "subclasses",
        "extends",
        "derived from",
    ):
        return "inherits"
    # where is X implemented / defined / located
    if has("where") or has("implemented", "implemented", "defined", "located", "declaration"):
        return "where"
    # config affects
    if has("config", "configuration") and has("affect", "affects", "impact", "impacted", "code"):
        return "config"
    # related files
    if has("file", "files") and has("involve", "involved", "related", "which", "touch", "touched"):
        return "files"
    return None


_STOPWORDS = {
    "where",
    "is",
    "are",
    "the",
    "a",
    "an",
    "implemented",
    "implement",
    "defined",
    "located",
    "callers",
    "callees",
    "caller",
    "callee",
    "of",
    "what",
    "does",
    "do",
    "it",
    "calls",
    "which",
    "import",
    "imports",
    "imported",
    "who",
    "by",
    "inherit",
    "inheritance",
    "inherits",
    "from",
    "to",
    "config",
    "configuration",
    "affects",
    "impact",
    "file",
    "files",
    "involved",
    "related",
    "touch",
    "this",
    "that",
    "code",
    "function",
    "class",
    "method",
    "module",
    "api",
    "cli",
    "command",
    "main",
    "path",
    "chain",
    "depends",
    "on",
    "for",
    "with",
    "when",
    "how",
    "why",
    "explain",
}


def extract_symbol(question: str, mode: str | None = None) -> str:
    """Pull the most likely symbol token out of a question (best effort)."""
    m = re.search(r"[\"'`]([A-Za-z_][A-Za-z0-9_.]*)[\"'`]", question)
    if m:
        return m.group(1)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", question)
    for t in tokens:
        if t.lower() in _STOPWORDS:
            continue
        if re.search(r"[A-Z]", t) or "_" in t:
            return t
    for t in tokens:
        if t.lower() not in _STOPWORDS:
            return t
    return question.strip().rstrip("?").strip()


def extract_path_symbols(question: str) -> tuple[str, str]:
    """Extract (from_symbol, to_symbol) for call-path questions."""
    m = re.search(
        r"from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+.*?\bto\s+([A-Za-z_][A-Za-z0-9_.]*)", question, re.I
    )
    if m:
        return m.group(1), m.group(2)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", question)
    code_tokens = [
        t
        for t in tokens
        if (re.search(r"[A-Z]", t) or "_" in t)
        and t.lower() not in {"from", "to", "call", "path", "chain", "the"}
    ]
    if len(code_tokens) >= 2:
        return code_tokens[0], code_tokens[-1]
    return "", ""


DETERMINISTIC_MODES = (
    "where",
    "callers",
    "callees",
    "imports",
    "imported_by",
    "inherits",
    "config",
    "files",
    "path",
)
GRAPHRAAG_MODES = ("local", "global", "drift", "basic")
