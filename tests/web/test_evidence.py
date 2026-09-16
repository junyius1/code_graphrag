"""Evidence mapping: real EvidenceItems -> UI JSON, no fabrication."""

from __future__ import annotations

from code_graphrag.web.evidence import map_evidence
from graphrag_query.context import EvidenceItem


def test_resolves_entity_from_title_and_loc(tiny_adapter):
    items = [
        EvidenceItem(
            source="search_code_symbol",
            kind="symbol",
            title="run",
            content="method body...",
            loc="main.py:5-12",
        ),
        EvidenceItem(
            source="search_documentation",
            kind="doc",
            title="nonexistent-doc-title",
            content="doc text",
            loc="",
        ),
    ]
    out = map_evidence(items, tiny_adapter)
    assert len(out) == 2
    e0, e1 = out
    assert e0["entity_id"] == "e-run"
    assert e0["name"] == "run"
    assert e0["file"] == "main.py"
    assert e0["start_line"] == 5 and e0["end_line"] == 12
    # unresolved item: no fabricated entity
    assert e1["entity_id"] is None
    assert e1["name"] is None


def test_loc_only_without_entity(tiny_adapter):
    items = [
        EvidenceItem(
            source="fetch_source_snippets",
            kind="source",
            title="weird title",
            content="code",
            loc="utils.py:1-3",
        )
    ]
    out = map_evidence(items, tiny_adapter)
    assert out[0]["file"] == "utils.py"
    assert out[0]["start_line"] == 1
    assert out[0]["entity_id"] is None


def test_empty_items(tiny_adapter):
    assert map_evidence([], tiny_adapter) == []
