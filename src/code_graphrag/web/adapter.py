"""Data adapter: built Code GraphRAG index -> explorer JSON.

The explorer backend never invents data: every node/edge/relation it serves
comes from the deterministic sidecar (``code_graphrag_index.json`` — entities
with ``file``/``start_line``/``end_line`` + typed relations) or from the
Microsoft GraphRAG parquet tables (``output/*.parquet`` — semantic
descriptions, text units, community reports). The two are joined on
``entity.graphrag_title == parquet.entities.title`` (verified invariant in
the index).

Responsibilities:
  * load & index the sidecar once per index dir (entities, relations,
    adjacency, title maps);
  * serve bounded subgraphs (seed + depth) for the Cytoscape frontend;
  * serve file trees, entity details, neighbors, call chains;
  * serve real source text from the parquet text units (the only place real
    code bytes exist in a built index) and, when the original repository is
    still on disk, from the files themselves.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

_REL_TYPE_RE = re.compile(r"^\s*\[?file:?\]?")
_UNIT_HEADER_RE = re.compile(
    r"^\[file:\s*(?P<file>[^\]]+)\]\s*(?:\[language:\s*[^\]]+\]\s*)?"
    r"(?:\[type:\s*(?P<type>[^\]]+)\]\s*)?(?:\[lines:\s*(?P<sl>\d+)\s*-\s*(?P<el>\d+)\]\s*)?"
)

# Entity types ordered by "code-ness" — used for default seeds/ranking.
_CODE_TYPES = ("function", "method", "class", "file", "module", "test")


class IndexAdapter:
    """One built index (sidecar + parquet tables) loaded and indexed once."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        out = self.index_dir / "output"
        sidecar_path = self.index_dir / "code_graphrag_index.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"sidecar index not found: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text())
        self.name: str = sidecar.get("name", self.index_dir.name)
        self.repo: str = sidecar.get("repo", "")

        self.entities: list[dict] = sidecar.get("entities", [])
        self.relations: list[dict] = sidecar.get("relations", [])
        self.by_id: dict[str, dict] = {e["id"]: e for e in self.entities}
        self.by_title: dict[str, dict] = {
            (e.get("graphrag_title") or e.get("qualified_name")): e for e in self.entities
        }

        self.outgoing: dict[str, list[dict]] = {}
        self.incoming: dict[str, list[dict]] = {}
        for r in self.relations:
            s, t = r.get("source"), r.get("target")
            if s not in self.by_id or t not in self.by_id:
                continue  # drop invalid relations defensively
            self.outgoing.setdefault(s, []).append(r)
            self.incoming.setdefault(t, []).append(r)
        for lst in self.outgoing.values():
            lst.sort(key=lambda r: _rel_sort_key(r))
        for lst in self.incoming.values():
            lst.sort(key=lambda r: _rel_sort_key(r))

        self._text_units: pd.DataFrame | None = None
        self._query_data: Any | None = None
        self._parquet_entities: pd.DataFrame | None = None
        if (out / "text_units.parquet").exists():
            self._text_units = pd.read_parquet(
                out / "text_units.parquet", columns=["id", "text", "document_id"]
            )
        if (out / "entities.parquet").exists():
            self._parquet_entities = pd.read_parquet(
                out / "entities.parquet",
                columns=["id", "title", "description", "text_unit_ids"],
            )

    # ------------------------------------------------------------------ #
    # stats / vocab
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for e in self.entities:
            by_type[e.get("type", "?")] = by_type.get(e.get("type", "?"), 0) + 1
        rel_types: dict[str, int] = {}
        for r in self.relations:
            rel_types[r.get("type", "?")] = rel_types.get(r.get("type", "?"), 0) + 1
        files = [e for e in self.entities if e.get("type") == "file"]
        return {
            "index_dir": str(self.index_dir),
            "name": self.name,
            "repo": self.repo,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "by_entity_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "relation_types": dict(sorted(rel_types.items(), key=lambda kv: -kv[1])),
            "file_count": len(files),
            "has_text_units": self._text_units is not None,
        }

    @property
    def query_data(self) -> Any:
        """The graphrag_query.QueryData view over this index (lazy, cached).

        Reused by the /api/query endpoint so the LangGraph Query Agent runs on
        exactly the same data access layer as the CLI.
        """
        if self._query_data is None:
            from graphrag_query.data_access import QueryData

            self._query_data = QueryData.load(self.index_dir)
        return self._query_data

    def entity_types(self) -> list[str]:
        return sorted({e.get("type", "?") for e in self.entities})

    def relation_types(self) -> list[str]:
        return sorted({r.get("type", "?") for r in self.relations})

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    def search(self, q: str, types: list[str] | None = None, limit: int = 20) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        exact: list[dict] = []
        prefix: list[dict] = []
        substr: list[dict] = []
        for e in self.entities:
            name = (e.get("name") or "").lower()
            qn = (e.get("qualified_name") or "").lower()
            if types and e.get("type") not in types:
                continue
            if name == q or qn == q:
                exact.append(e)
            elif name.startswith(q) or qn.startswith(q):
                prefix.append(e)
            elif q in name or q in qn:
                substr.append(e)
        out = (exact + prefix + substr)[:limit]
        return [self._node_view(e) for e in out]

    # ------------------------------------------------------------------ #
    # subgraph
    # ------------------------------------------------------------------ #

    def subgraph(
        self,
        seed_ids: list[str] | None = None,
        depth: int = 1,
        limit: int = 300,
        types: list[str] | None = None,
        rel: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Bounded BFS subgraph. No seed -> the repository entity's 1-hop view
        (top-level directories / files), which is a useful first frame."""
        depth = max(1, min(int(depth), 3))
        limit = max(10, min(int(limit), 800))
        rel = set(rel) if rel else None
        nodes: dict[str, dict] = {}
        edges: dict[str, dict] = {}

        def add_node(e: dict) -> None:
            if (
                types
                and e.get("type") not in types
                and e.get("type") not in ("directory", "repository")
            ):
                return
            nodes[e["id"]] = self._compact_node(e)

        def add_edge(r: dict) -> None:
            s, t = r.get("source"), r.get("target")
            if s not in self.by_id or t not in self.by_id:
                return
            if rel and r.get("type") not in rel:
                return
            eid = f"{s}::{r.get('type')}::{t}"
            if eid in edges:
                return
            nodes.setdefault(s, self._compact_node(self.by_id[s]))
            nodes.setdefault(t, self._compact_node(self.by_id[t]))
            edges[eid] = self._compact_edge(r)

        seeds = [s for s in (seed_ids or []) if s in self.by_id]
        if not seeds:
            repo = next((e for e in self.entities if e.get("type") == "repository"), None)
            if repo is not None:
                seeds = [repo["id"]]  # repo -> top-level dirs/files on depth 1
            else:
                seeds = [
                    e["id"]
                    for e in sorted(
                        self.entities,
                        key=lambda e: (
                            -len(self.outgoing.get(e["id"], []) + self.incoming.get(e["id"], []))
                        ),
                    )
                    if not types or e.get("type") in types
                ][: max(20, min(limit, 100))]
        for sid in seeds:
            add_node(self.by_id[sid])

        frontier = list(seeds)
        for _ in range(depth):
            if len(nodes) >= limit:
                break
            next_frontier: list[str] = []
            for nid in frontier:
                if len(nodes) >= limit:
                    break
                rows = self.outgoing.get(nid, []) + self.incoming.get(nid, [])
                # repository/directory: surface folders before loose files
                if self.by_id[nid].get("type") in ("repository", "directory"):
                    rows = sorted(
                        rows,
                        key=lambda r: (
                            0
                            if self.by_id.get(
                                r["target"] if r["source"] == nid else r["source"], {}
                            ).get("type")
                            in ("repository", "directory")
                            else 1
                        ),
                    )
                for r in rows:
                    other = r["target"] if r["source"] == nid else r["source"]
                    if other in nodes:
                        if rel is None or r.get("type") in rel:
                            add_edge(r)
                        continue
                    add_edge(r)
                    next_frontier.append(other)
                    if len(nodes) >= limit:
                        break
            frontier = next_frontier
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    # ------------------------------------------------------------------ #
    # entity detail / neighbors
    # ------------------------------------------------------------------ #

    def entity(self, entity_id: str) -> dict | None:
        e = self.by_id.get(entity_id)
        if e is None:
            return None
        out = self._node_view(e)
        out["description"] = e.get("description") or ""
        out["semantic_description"] = self._parquet_description(e.get("graphrag_title"))
        out["attributes"] = e.get("attributes") or {}
        out["language"] = e.get("language")
        out["start_line"] = e.get("start_line") or None
        out["end_line"] = e.get("end_line") or None
        out["degree"] = len(self.outgoing.get(entity_id, [])) + len(
            self.incoming.get(entity_id, [])
        )
        return out

    def _parquet_description(self, title: str | None) -> str:
        if not title or self._parquet_entities is None:
            return ""
        row = self._parquet_entities[self._parquet_entities["title"] == title]
        if row.empty:
            return ""
        return str(row.iloc[0]["description"] or "")

    def neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        limit: int = 60,
        rel: str | None = None,
        direction: str = "both",
    ) -> dict[str, list[dict]]:
        e = self.by_id.get(entity_id)
        if e is None:
            return {"nodes": [], "edges": []}
        depth = max(1, min(int(depth), 2))
        limit = max(5, min(int(limit), 400))
        ids: set[str] = {entity_id}
        out_rows = [
            r
            for r in self.outgoing.get(entity_id, [])
            if direction in ("out", "both") and (rel is None or r.get("type") == rel)
        ][:limit]
        in_rows = [
            r
            for r in self.incoming.get(entity_id, [])
            if direction in ("in", "both") and (rel is None or r.get("type") == rel)
        ][:limit]
        all_rows = list(out_rows) + list(in_rows)
        if depth > 1:
            next_ids: list[str] = []
            for r in all_rows:
                for nid in (r["source"], r["target"]):
                    if nid != entity_id and nid in self.by_id:
                        next_ids.append(nid)
            for _ in range(int(depth) - 1):
                if not next_ids or len(all_rows) >= limit * 3:
                    break
                more: list[str] = []
                for nid in next_ids:
                    for r in self.outgoing.get(nid, []) + self.incoming.get(nid, []):
                        if rel is not None and r.get("type") != rel:
                            continue
                        all_rows.append(r)
                        for other in (r["source"], r["target"]):
                            if other not in ids and other in self.by_id:
                                ids.add(other)
                                more.append(other)
                next_ids = more
        for r in all_rows:
            ids.add(r["source"])
            ids.add(r["target"])
        ids = list(ids)[: limit * 3]
        seen_rows: set[str] = set()
        edges = []
        for r in all_rows:
            eid = f"{r['source']}::{r.get('type')}::{r['target']}"
            if eid in seen_rows:
                continue
            seen_rows.add(eid)
            edges.append(self._compact_edge(r))
        return {
            "nodes": [self._compact_node(self.by_id[i]) for i in ids if i in self.by_id],
            "edges": edges,
        }

    # ------------------------------------------------------------------ #
    # call chains
    # ------------------------------------------------------------------ #

    def call_chain(
        self,
        entity_id: str,
        depth: int = 3,
        limit: int = 200,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Multi-level CALLS walk (depth-bounded, cycle-safe, node-capped)."""
        e = self.by_id.get(entity_id)
        if e is None:
            return {"root": None, "nodes": [], "edges": []}
        depth = max(1, min(int(depth), 6))
        limit = max(10, min(int(limit), 600))
        nodes: dict[str, dict] = {entity_id: self._compact_node(e)}
        edges: dict[str, dict] = {}
        cycles = 0

        def walk(nid: str, d: int, path: set[str]) -> None:
            if d > depth or len(nodes) >= limit:
                return
            for r in self.outgoing.get(nid, []) + self.incoming.get(nid, []):
                if r.get("type") != "CALLS":
                    continue
                key = (r["source"], r["target"])
                other = (
                    r["target"]
                    if direction in ("out", "both") and r["source"] == nid
                    else (r["source"] if direction in ("in", "both") else None)
                )
                if other is None:
                    continue
                if other in path:
                    nonlocal cycles
                    cycles += 1
                    if len(edges) < limit:
                        edges[f"{key[0]}::CALLS::{key[1]}"] = self._compact_edge(r)
                    continue
                if len(edges) < limit:
                    edges[f"{r['source']}::CALLS::{r['target']}"] = self._compact_edge(r)
                if other not in nodes and other in self.by_id:
                    nodes[other] = self._compact_node(self.by_id[other])
                if len(nodes) >= limit:
                    return
                walk(other, d + 1, path | {other})

        walk(entity_id, 1, {entity_id})
        return {
            "root": entity_id,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "truncated": len(nodes) >= limit,
            "cycles": cycles,
        }

    # ------------------------------------------------------------------ #
    # file tree / source
    # ------------------------------------------------------------------ #

    def file_tree(self) -> dict:
        files = [e for e in self.entities if e.get("type") == "file"]
        dirs = [e for e in self.entities if e.get("type") == "directory"]
        repo = next((e for e in self.entities if e.get("type") == "repository"), None)
        root: dict = {
            "name": repo["name"] if repo else self.index_dir.name,
            "path": "",
            "type": "directory",
            "children": {},
        }

        def ensure_dir(path: str) -> None:
            node = root
            parts = path.split("/")
            for i, part in enumerate(parts):
                node = node["children"].setdefault(
                    part,
                    {
                        "name": part,
                        "path": "/".join(parts[: i + 1]),
                        "type": "directory",
                        "children": {},
                    },
                )

        for e in dirs:
            ensure_dir(e.get("file", ""))
        for e in files:
            path = e.get("file", "")
            parts = path.split("/")
            if len(parts) > 1:
                ensure_dir("/".join(parts[:-1]))
            node = root
            for part in parts[:-1]:
                node = node["children"][part]
            node["children"].setdefault(
                parts[-1],
                {
                    "name": parts[-1],
                    "path": path,
                    "type": "file",
                    "entity_id": e["id"],
                    "module_entity_id": self._module_id_for(path),
                    "description": (e.get("description") or "")[:300],
                },
            )
        return root

    def _module_id_for(self, file_path: str) -> str | None:
        mod = file_path[:-3].replace("/", ".") if file_path.endswith(".py") else None
        e = self.by_title.get(mod) if mod else None
        return e["id"] if e else None

    def file_entities(self, file_path: str) -> list[dict]:
        """All entities defined in a file (via DEFINES/CONTAINS + direct file)."""
        out = []
        for e in self.entities:
            if e.get("file") == file_path and e.get("type") not in (
                "file",
                "module",
                "directory",
                "repository",
            ):
                out.append(self._node_view(e))
        return out

    def repo_root(self) -> Path | None:
        """The original repository on disk, if it can be located next to the
        index (``<...>/open-swe-code`` source was indexed from somewhere)."""
        env_root = (
            Path(os.environ.get("CODE_GRAPHRAG_SOURCE_ROOT", "")).expanduser()
            if os.environ.get("CODE_GRAPHRAG_SOURCE_ROOT")
            else None
        )
        if env_root and env_root.is_dir():
            return env_root
        return None

    def source_for_file(self, file_path: str) -> dict[str, Any]:
        """Full file source. Prefers the on-disk repo; falls back to the
        parquet text units (real indexed text, possibly per-symbol chunks)."""
        norm = file_path.lstrip("./").strip("/")
        root = self.repo_root()
        if root is not None:
            p = root / norm
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    return {
                        "path": norm,
                        "text": text,
                        "origin": "repo",
                        "line_count": text.count("\n") + 1,
                    }
                except OSError:
                    pass
        return self._source_from_units(norm)

    def _source_from_units(self, file_path: str) -> dict[str, Any]:
        if self._text_units is None:
            return {"path": file_path, "text": None, "origin": None, "line_count": None}
        doc = f"doc::{file_path}"
        d = self._text_units[self._text_units["document_id"] == doc]
        if d.empty:
            return {"path": file_path, "text": None, "origin": None, "line_count": None}
        chunks: list[tuple[int | None, str, int, int]] = []
        whole_file: str | None = None
        for r in d.itertuples(index=False):
            text = str(r.text)
            m = _UNIT_HEADER_RE.match(text)
            etype = m.group("type") if m else None
            if etype == "file":
                whole_file = text.split("\n", 1)[1] if "\n" in text else ""
                continue
            sl = int(m.group("sl")) if (m and m.group("sl")) else None
            el = int(m.group("el")) if (m and m.group("el")) else None
            body = text.split("\n", 1)[1] if "\n" in text else text
            chunks.append((sl, body, sl, el))
        if whole_file is not None:
            return {
                "path": file_path,
                "text": whole_file,
                "origin": "index:file-unit",
                "line_count": whole_file.count("\n") + 1,
            }
        if not chunks:
            return {"path": file_path, "text": None, "origin": None, "line_count": None}
        chunks.sort(key=lambda c: (c[0] is None, c[0] or 0))
        parts: list[str] = []
        prev_end: int | None = None
        for sl, body, _s, el in chunks:
            if prev_end is not None and sl is not None and sl > (prev_end + 1):
                parts.append(f"  # ... [{prev_end + 1}-{sl - 1} not stored in index] ...")
            parts.append(body.rstrip("\n"))
            if el is not None:
                prev_end = el
        return {
            "path": file_path,
            "text": "\n".join(parts),
            "origin": "index:units",
            "line_count": None,
        }

    def source_for_entity(self, entity: dict) -> dict[str, Any]:
        """Source for a code entity: exact slice if the repo is on disk,
        otherwise the entity's own text units (method-level code blocks)."""
        if entity.get("type") in ("file", "module"):
            return self.source_for_file(entity.get("file", ""))
        root = self.repo_root()
        file_path = entity.get("file", "")
        sl, el = entity.get("start_line"), entity.get("end_line")
        if root is not None and file_path:
            p = root / file_path
            if p.is_file() and sl and el and el > sl:
                try:
                    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                    if len(lines) >= el:
                        text = "\n".join(lines[sl - 1 : el])
                        return {
                            "path": file_path,
                            "text": text,
                            "origin": "repo",
                            "line_count": el - sl + 1,
                            "start_line": sl,
                            "end_line": el,
                        }
                except OSError:
                    pass
        if self._text_units is not None:
            title = entity.get("graphrag_title")
            unit_ids = self._entity_unit_ids(title)
            if unit_ids:
                d = self._text_units[self._text_units["id"].isin(unit_ids)]
                parts = []
                for r in d.itertuples(index=False):
                    text = str(r.text)
                    parts.append(text.split("\n", 1)[1] if "\n" in text else text)
                if parts:
                    return {
                        "path": file_path,
                        "text": "\n".join(parts),
                        "origin": "index:units",
                        "line_count": None,
                        "start_line": sl,
                        "end_line": el,
                    }
        return {
            "path": file_path or None,
            "text": None,
            "origin": None,
            "line_count": None,
            "start_line": sl,
            "end_line": el,
        }

    def _entity_unit_ids(self, title: str | None) -> list[str]:
        if not title or self._parquet_entities is None:
            return []
        row = self._parquet_entities[self._parquet_entities["title"] == title]
        if row.empty:
            return []
        v = row.iloc[0]["text_unit_ids"]
        if v is None:
            return []
        return [str(x) for x in v] if not isinstance(v, str) else [v]

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #

    def _node_view(self, e: dict) -> dict:
        return {
            "id": e["id"],
            "name": e.get("name"),
            "qualified_name": e.get("qualified_name"),
            "type": e.get("type"),
            "file": e.get("file"),
            "location": e.get("location"),
        }

    def _compact_node(self, e: dict) -> dict:
        return {
            "id": e["id"],
            "label": e.get("name"),
            "type": e.get("type"),
            "qualified_name": e.get("qualified_name"),
            "file": e.get("file"),
            "location": e.get("location"),
        }

    def _compact_edge(self, r: dict) -> dict:
        return {
            "id": f"{r['source']}::{r.get('type')}::{r['target']}",
            "source": r["source"],
            "target": r["target"],
            "type": r.get("type"),
            "description": (r.get("description") or "")[:200],
        }


def _rel_sort_key(r: dict) -> tuple:
    """Code relations first, then deterministic order."""
    prio = {
        "CALLS": 0,
        "IMPORTS": 1,
        "INHERITS": 2,
        "DEFINES": 3,
        "CONTAINS": 4,
        "TESTS": 5,
        "RAISES": 6,
        "DOCUMENTED_BY": 7,
        "ACCEPTS_PARAMETER": 8,
        "CONFIGURED_BY": 9,
    }
    return (prio.get(r.get("type"), 50), str(r.get("description", "")))


_STORES: dict[str, IndexAdapter] = {}
_LOCK = threading.Lock()


def get_adapter(index_dir: str | Path) -> IndexAdapter:
    """Process-wide cache: each index dir is parsed exactly once."""
    key = str(Path(index_dir).resolve())
    with _LOCK:
        a = _STORES.get(key)
        if a is None:
            a = IndexAdapter(index_dir)
            _STORES[key] = a
            logger.info("loaded index adapter for %s (%d entities)", key, len(a.entities))
        return a
