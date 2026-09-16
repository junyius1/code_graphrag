"""Map Query Agent evidence items to UI JSON.

The LangGraph Query Agent records ``EvidenceItem`` records as it retrieves.
This module resolves each item's title/location against the index so the
frontend can link evidence rows to real entities and source locations.
Nothing is invented: items that cannot be resolved keep ``entity_id=None``.
"""

from __future__ import annotations

import re
from typing import Any

from code_graphrag.web.adapter import IndexAdapter

# "<file>:<start>-<end>" inside an evidence loc, e.g. "agent/analyzer.py:92-99"
_LOC_RE = re.compile(r"([^\s():]+\.[A-Za-z0-9_./-]+):(\d+)-(\d+)")


def map_evidence(items: list[Any], adapter: IndexAdapter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        item: dict[str, Any] = {
            "source": it.source,
            "kind": it.kind,
            "title": it.title,
            "loc": it.loc,
            "content": (it.content or "")[:1200],
            "entity_id": None,
            "name": None,
            "file": None,
            "start_line": None,
            "end_line": None,
        }
        m = _LOC_RE.search(it.loc or "")
        if m:
            item["file"], item["start_line"], item["end_line"] = (
                m.group(1),
                int(m.group(2)),
                int(m.group(3)),
            )
        ent = adapter.by_title.get(it.title)
        if ent is None and it.kind in ("symbol", "entity"):
            hits = adapter.search(it.title, limit=1)
            if hits:
                ent = adapter.by_id.get(hits[0]["id"])
        if ent is not None:
            item["entity_id"] = ent["id"]
            item["name"] = ent.get("name")
            item["file"] = item["file"] or ent.get("file")
            if item["start_line"] is None and ent.get("start_line"):
                item["start_line"] = ent.get("start_line")
                item["end_line"] = ent.get("end_line")
        out.append(item)
    return out
