"""Read-only access layer for the Query Agent over a built Code GraphRAG index.

Combines three already-produced data sources (no re-indexing, no GraphRAG
reimplementation):

1. the deterministic sidecar ``code_graphrag_index.json`` (entities with
   ``file`` / ``start_line`` / ``end_line`` + typed relations) -> precise,
   citation-bearing answers;
2. the GraphRAG parquet tables (``output/*.parquet``) -> semantic descriptions,
   relationships, community reports, raw text units (source snippets, docs);
3. ``Microsoft GraphRAG`` local / global search (``graphrag_search``) ->
   LLM-generated context for open-ended questions.

The sidecar and the parquet entity table are joined on
``sidecar.graphrag_title == parquet.entities.title`` (verified invariant), so
an agent can move from a precise symbol to its semantic description and its
source text units.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code_graphrag.logging_setup import get_logger
from code_graphrag.query.queries import CodeGraphQuery, QueryResult

logger = get_logger(__name__)

_CODE_SYMBOL_TYPES = ("function", "method", "class", "api")
_MODULE_TYPES = ("module", "file")
_MAX_RESULTS = 8
_SNIPPET_MAX_CHARS = 4000


def _to_list(value: Any) -> list[str]:
    """Normalize a parquet list-ish cell (ndarray / list / None) to list[str]."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(v) for v in np.atleast_1d(value).tolist()]


@dataclass
class SourceSnippet:
    file: str
    symbol: str
    start_line: int | None
    end_line: int | None
    text: str
    unit_id: str = ""


@dataclass
class QueryData:
    """All query-side data for one index, loaded once."""

    index_dir: Path
    graph: CodeGraphQuery
    entities_df: pd.DataFrame
    relationships_df: pd.DataFrame
    text_units_df: pd.DataFrame
    community_reports_df: pd.DataFrame
    sidecar_entities: list[dict] = field(default_factory=list)
    sidecar_relations: list[dict] = field(default_factory=list)
    _by_title: dict[str, dict] = field(default_factory=dict, repr=False)
    _sidecar_by_title: dict[str, dict] = field(default_factory=dict, repr=False)

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, index_dir: str | Path) -> QueryData:
        index_dir = Path(index_dir)
        graph = CodeGraphQuery(index_dir)
        out = index_dir / "output"
        entities_df = pd.read_parquet(out / "entities.parquet")
        relationships_df = pd.read_parquet(out / "relationships.parquet")
        text_units_df = pd.read_parquet(out / "text_units.parquet")
        community_reports_df = pd.read_parquet(out / "community_reports.parquet")

        sidecar = json.loads((index_dir / "code_graphrag_index.json").read_text())
        sidecar_entities = sidecar.get("entities", [])
        sidecar_relations = sidecar.get("relations", [])

        data = cls(
            index_dir=index_dir,
            graph=graph,
            entities_df=entities_df,
            relationships_df=relationships_df,
            text_units_df=text_units_df,
            community_reports_df=community_reports_df,
            sidecar_entities=sidecar_entities,
            sidecar_relations=sidecar_relations,
        )
        for row in entities_df.itertuples(index=False):
            data._by_title[str(row.title)] = data._entity_row(row)
        for e in sidecar_entities:
            title = e.get("graphrag_title") or e["qualified_name"]
            data._sidecar_by_title.setdefault(title, e)
        return data

    @staticmethod
    def _entity_row(row: Any) -> dict:
        return {
            "id": row.id,
            "title": str(row.title),
            "type": row.type,
            "description": row.description or "",
            "text_unit_ids": _to_list(row.text_unit_ids),
            "frequency": row.frequency,
            "degree": row.degree,
        }

    # -- sidecar (precise, citation-bearing) ----------------------------------

    def find_sidecar_entities(self, name: str, types: tuple[str, ...] | None = None) -> list[dict]:
        from code_graphrag.query.queries import _find_entity

        matches = _find_entity(self.sidecar_entities, name, types)
        return matches[:_MAX_RESULTS]

    def relations_between(
        self,
        source: str,
        target: str,
        rel_type: str | None = None,
    ) -> list[dict]:
        """Sidecar relations between the best-matching source/target entities.

        ``source``/``target`` are symbol names (not ids); the best match on each
        side is resolved via name scoring.
        """
        from code_graphrag.query.queries import _find_entity

        srcs = _find_entity(self.sidecar_entities, source) or self.sidecar_entities[:0]
        tgts = _find_entity(self.sidecar_entities, target)
        if not srcs or not tgts:
            return []
        out: list[dict] = []
        for s in srcs[:2]:
            for t in tgts[:2]:
                for r in self.sidecar_relations:
                    if {r["source"], r["target"]} != {s["id"], t["id"]}:
                        continue
                    if rel_type and r["type"] != rel_type:
                        continue
                    out.append({**r, "source_entity": s, "target_entity": t})
        out.sort(key=lambda r: -len(str(r.get("description") or "")))
        return out[:_MAX_RESULTS]

    # -- parquet entities ------------------------------------------------------

    def find_entities(
        self,
        query: str,
        types: tuple[str, ...] | None = None,
        limit: int = _MAX_RESULTS,
    ) -> list[dict]:
        """Find GraphRAG entities by title/name (exact then substring, ranked)."""
        q = query.strip().lower()
        scored: list[tuple[int, int, dict]] = []
        for title, ent in self._by_title.items():
            if types and ent["type"] not in types:
                continue
            tl = title.lower()
            base = tl.rsplit(".", 1)[-1]
            if tl == q or base == q:
                score = 5 if tl == q else 4
            elif tl.endswith("." + q) or q in tl:
                score = 2
            else:
                continue
            scored.append((score, -len(title), ent))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [ent for _, _, ent in scored[:limit]]

    def entity_full(self, title: str) -> dict | None:
        """Sidecar + parquet merged view of one entity (by title)."""
        ent = self._sidecar_by_title.get(title) or self._by_title.get(title)
        if ent is None:
            return None
        out = dict(ent)
        if "file" in ent:  # sidecar view: enrich with parquet description
            p = self._by_title.get(title)
            if p and p["description"] and p["description"] != ent.get("description"):
                out["semantic_description"] = p["description"]
                out["text_unit_ids"] = p["text_unit_ids"]
        elif "text_unit_ids" in ent:  # parquet view: enrich with sidecar location
            s = self._sidecar_by_title.get(title)
            if s:
                out.update(
                    {
                        "file": s["file"],
                        "start_line": s["start_line"],
                        "end_line": s["end_line"],
                        "location": s["location"],
                        "language": s.get("language"),
                    }
                )
        return out

    # -- relationships (parquet, semantic descriptions) -----------------------

    def find_relationships(
        self,
        query: str,
        rel_type: str | None = None,
        limit: int = _MAX_RESULTS,
    ) -> list[dict]:
        """Relationships whose endpoints or description mention ``query``.

        ``rel_type`` filters on the leading verb of the description
        (e.g. ``"CALLS: a calls b"``).
        """
        q = query.strip().lower()
        rows: list[dict] = []
        for row in self.relationships_df.itertuples(index=False):
            desc = str(row.description or "")
            if rel_type and not desc.upper().startswith(rel_type.upper()):
                continue
            if (
                q not in str(row.source).lower()
                and q not in str(row.target).lower()
                and q not in desc.lower()
            ):
                continue
            rows.append(
                {
                    "source": str(row.source),
                    "target": str(row.target),
                    "description": desc,
                    "weight": float(row.weight),
                }
            )
        rows.sort(key=lambda r: -r["weight"])
        return rows[:limit]

    # -- source snippets --------------------------------------------------------

    def _parse_unit_header(self, text: str) -> tuple[str | None, int | None, int | None]:
        """Extract (file, start_line, end_line) from a text-unit header line.

        Units start with ``[file: app.py] [language: python] [type: method]
        [lines: 18-26]`` (files omit ``[lines: ...]``).
        """
        header = text.split("\n", 1)[0]
        file_m = re.search(r"\[file:\s*([^\]]+)\]", header)
        lines_m = re.search(r"\[lines:\s*(\d+)\s*-\s*(\d+)\]", header)
        return (
            file_m.group(1) if file_m else None,
            int(lines_m.group(1)) if lines_m else None,
            int(lines_m.group(2)) if lines_m else None,
        )

    def source_snippets_for(
        self,
        symbol_title: str,
        max_snippets: int = 4,
    ) -> list[SourceSnippet]:
        """Raw source text for an entity: its own text units (+ the file unit).

        An entity's ``text_unit_ids`` point at method-level units that carry the
        actual code; the parent file's unit carries the whole file.
        """
        ent = self._by_title.get(symbol_title)
        unit_ids = _to_list(ent["text_unit_ids"]) if ent else []
        if not unit_ids and symbol_title in self._sidecar_by_title:
            # sidecar-only title: try matching by qualified name in parquet
            ent = self._by_title.get(symbol_title)
            unit_ids = _to_list(ent["text_unit_ids"]) if ent else []
        if not unit_ids:
            return []
        tu = self.text_units_df
        snippets: list[SourceSnippet] = []
        for uid in unit_ids[:max_snippets]:
            row = tu[tu["id"] == uid]
            if not len(row):
                continue
            r = row.iloc[0]
            file, sl, el = self._parse_unit_header(str(r["text"]))
            base = str(r["document_id"]).removeprefix("doc::")
            snippets.append(
                SourceSnippet(
                    file=file or base,
                    symbol=symbol_title,
                    start_line=sl,
                    end_line=el,
                    text=str(r["text"]),
                    unit_id=uid,
                )
            )
        return snippets

    def search_source(self, query: str, limit: int = 5) -> list[SourceSnippet]:
        """Full-text search over code text units (non-doc documents)."""
        q = query.strip().lower()
        hits: list[tuple[int, SourceSnippet]] = []
        for r in self.text_units_df.itertuples(index=False):
            text = str(r.text)
            doc = str(r.document_id)
            if doc.startswith("doc::doc:") or doc.endswith((".md", ".rst", ".txt")):
                continue  # documentation handled by search_documentation
            low = text.lower()
            if q not in low:
                continue
            file, sl, el = self._parse_unit_header(text)
            header = text.split("\n", 1)[0]
            sym_m = re.search(r"\[(?:symbol|type|title):\s*([^\]]+)\]", header)
            hits.append(
                (
                    low.count(q),
                    SourceSnippet(
                        file=file or doc.removeprefix("doc::"),
                        symbol=sym_m.group(1) if sym_m else "",
                        start_line=sl,
                        end_line=el,
                        text=text,
                        unit_id=str(r.id),
                    ),
                )
            )
        hits.sort(key=lambda x: -x[0])
        return [h for _, h in hits[:limit]]

    def search_documentation(self, query: str, limit: int = 5) -> list[dict]:
        """Search markdown/documentation text units and return their content."""
        q = query.strip().lower()
        hits: list[tuple[int, dict]] = []
        for r in self.text_units_df.itertuples(index=False):
            doc = str(r.document_id)
            if not doc.removeprefix("doc::").lower().endswith((".md", ".mdx", ".rst", ".txt")):
                continue
            text = str(r.text)
            low = text.lower()
            if q not in low:
                continue
            header = text.split("\n", 1)[0]
            file_m = re.search(r"\[file:\s*([^\]]+)\]", header)
            sec_m = re.search(r"\[section:\s*([^\]]+)\]", header)
            hits.append(
                (
                    low.count(q),
                    {
                        "document": file_m.group(1) if file_m else doc.removeprefix("doc::"),
                        "section": sec_m.group(1) if sec_m else "",
                        "content": text,
                        "unit_id": str(r.id),
                    },
                )
            )
        hits.sort(key=lambda x: -x[0])
        return [h for _, h in hits[:limit]]

    # -- GraphRAG LLM search (delegates, never reimplements) -------------------

    def local_search(self, question: str) -> QueryResult:
        from code_graphrag.query.graphrag_search import graphrag_search

        r = graphrag_search(self.index_dir, question, search_type="local")
        if r.error:
            return QueryResult(question, f"local search failed: {r.error}", source="graphrag:local")
        return QueryResult(question, r.answer, source="graphrag:local")

    def global_search(self, question: str, community_level: int = 2) -> QueryResult:
        from code_graphrag.query.graphrag_search import graphrag_search

        r = graphrag_search(
            self.index_dir, question, search_type="global", community_level=community_level
        )
        if r.error:
            return QueryResult(
                question, f"global search failed: {r.error}", source="graphrag:global"
            )
        return QueryResult(question, r.answer, source="graphrag:global")

    def overview(self) -> str:
        return self.graph.overview().answer
