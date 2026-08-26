"""Query layer over a Code GraphRAG index.

Two complementary paths:
1. ``CodeGraphQuery`` — deterministic traversal of the code knowledge graph
   (sidecar) for precise questions: where is X, callers/callees, imports,
   inheritance, call chains, affected-by-config, related files.
2. ``graphrag_search`` — Microsoft GraphRAG local/global/drift/basic search for
   open-ended semantic questions (needs the index dir + a configured LLM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    question: str
    answer: str
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    source: str = "code_graph"


def _match_score(e: dict, term_l: str) -> int:
    """Specificity score for a term/entity match (higher = more specific)."""
    if e["qualified_name"].lower() == term_l:
        return 5
    if e["name"].lower() == term_l:
        return 4
    if e["qualified_name"].lower().endswith("." + term_l):
        return 3
    if e["name"].lower() in term_l:
        return 1
    if term_l in e["qualified_name"].lower():
        return 1
    return 0


def _find_entity(
    entities: list[dict], term: str, types: tuple[str, ...] | None = None
) -> list[dict]:
    term_l = term.lower()
    out = []
    for e in entities:
        if types and e["type"] not in types:
            continue
        if _match_score(e, term_l) > 0:
            out.append(e)
    # rank by specificity, then name length (shorter qualified name = more direct)
    out.sort(key=lambda e: (-_match_score(e, term_l), len(e["qualified_name"])))
    return out


class CodeGraphQuery:
    """Deterministic queries over the code knowledge graph sidecar."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        sidecar_path = self.index_dir / "code_graphrag_index.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"code_graphrag_index.json not found in {self.index_dir}")
        sidecar = json.loads(sidecar_path.read_text())
        self.repo = sidecar.get("repo", "")
        self.entities: list[dict] = sidecar["entities"]
        self.relations: list[dict] = sidecar["relations"]
        self._by_id = {e["id"]: e for e in self.entities}
        self._g = self._build_graph()
        self._out = nx.DiGraph()
        self._inn = nx.DiGraph()
        self._rel_type: dict[tuple[str, str], set[str]] = {}
        for r in self.relations:
            self._out.add_edge(r["source"], r["target"])
            self._inn.add_edge(r["target"], r["source"])
            self._rel_type.setdefault((r["source"], r["target"]), set()).add(r["type"])

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        for e in self.entities:
            g.add_node(e["id"], title=e["qualified_name"])
        for r in self.relations:
            g.add_edge(r["source"], r["target"], type=r["type"])
        return g

    # -- helpers -------------------------------------------------------------

    def _neighbors(self, entity_id: str, rel_type: str | None, direction: str) -> list[dict]:
        """Neighboring entities connected by ``rel_type`` in the given direction.

        ``_rel_type`` is keyed by the original (source, target) pair, so the key
        we look up depends on direction: for "out" we own the source, for "in"
        we own the target.
        """
        if direction == "out":
            graph = self._out
            lookup = lambda other: self._rel_type.get((entity_id, other), ())  # noqa: E731
        else:
            graph = self._inn
            lookup = lambda other: self._rel_type.get((other, entity_id), ())  # noqa: E731
        out = []
        for nb in graph.neighbors(entity_id):
            if rel_type is None or rel_type in lookup(nb):
                ent = self._by_id.get(nb)
                if ent is not None:
                    out.append(ent)
        return out

    def _format_entity(self, e: dict) -> str:
        loc = e["location"]
        return f"{e['qualified_name']} [{e['type']}] @ {loc}"

    def _format_relations(self, e: dict, rel_type: str | None, direction: str) -> str:
        nbs = self._neighbors(e["id"], rel_type, direction)
        if not nbs:
            return f"no {rel_type or ''} relations found".strip()
        lines = [f"{e['qualified_name']} {rel_type or 'relations'} ({direction}):"]
        for n in nbs:
            lines.append(f"  - {self._format_entity(n)}")
        return "\n".join(lines)

    # -- query APIs ----------------------------------------------------------

    def where_is(self, symbol: str) -> QueryResult:
        """Where is symbol implemented? (name matches, ranked by specificity)"""
        matches = _find_entity(self.entities, symbol)
        if not matches:
            return QueryResult(f"where is {symbol}", f"Symbol '{symbol}' not found in index.")

        # rank: exact qualified name > exact name > partial; prefer code symbols
        def score(e: dict) -> tuple[int, int]:
            s = 0
            if e["qualified_name"].lower() == symbol.lower():
                s += 4
            if e["name"].lower() == symbol.lower():
                s += 3
            if e["type"] in ("function", "method", "class"):
                s += 1
            return (-s, len(e["qualified_name"]))

        matches.sort(key=score)
        lines = []
        for e in matches[:8]:
            lines.append(self._format_entity(e))
            if e.get("description"):
                lines.append(f"      {e['description'][:200]}")
        ans = f"'{symbol}' is implemented in:\n" + "\n".join(lines)
        return QueryResult(f"where is {symbol}", ans, entities=matches[:8])

    def callers(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol, ("function", "method", "class", "api"))
        if not matches:
            return QueryResult(f"callers of {symbol}", f"Symbol '{symbol}' not found.")
        best = matches[0]
        return QueryResult(
            f"callers of {symbol}",
            self._format_relations(best, "CALLS", "in"),
            entities=[best],
        )

    def callees(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol, ("function", "method", "class"))
        if not matches:
            return QueryResult(f"callees of {symbol}", f"Symbol '{symbol}' not found.")
        best = matches[0]
        return QueryResult(
            f"callees of {symbol}",
            self._format_relations(best, "CALLS", "out"),
            entities=[best],
        )

    def imports_of(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol, ("module", "file"))
        if not matches:
            return QueryResult(f"imports of {symbol}", f"Module/file '{symbol}' not found.")
        best = sorted(matches, key=lambda e: (e["type"] != "module", e["qualified_name"]))[0]
        return QueryResult(
            f"imports of {symbol}",
            self._format_relations(best, "IMPORTS", "out"),
            entities=[best],
        )

    def imported_by(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol, ("module", "file"))
        if not matches:
            return QueryResult(f"imported by {symbol}", f"Module/file '{symbol}' not found.")
        best = sorted(matches, key=lambda e: (e["type"] != "module", e["qualified_name"]))[0]
        return QueryResult(
            f"imported by {symbol}",
            self._format_relations(best, "IMPORTS", "in"),
            entities=[best],
        )

    def inherits(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol, ("class",))
        if not matches:
            return QueryResult(f"inheritance of {symbol}", f"Class '{symbol}' not found.")
        best = matches[0]
        parts = [
            self._format_relations(best, "INHERITS", "out"),
            self._format_relations(best, "INHERITS", "in"),
        ]
        return QueryResult(f"inheritance of {symbol}", "\n".join(parts), entities=[best])

    def call_path(self, from_symbol: str, to_symbol: str) -> QueryResult:
        """Shortest call-chain path from A to B (CALLS edges, undirected fallback)."""
        src = _find_entity(self.entities, from_symbol, ("function", "method", "api"))
        dst = _find_entity(self.entities, to_symbol, ("function", "method", "class"))
        if not src or not dst:
            missing = from_symbol if not src else to_symbol
            return QueryResult(f"path {from_symbol}->{to_symbol}", f"Symbol '{missing}' not found.")
        s_id, t_id = src[0]["id"], dst[0]["id"]
        path = None
        try:
            path = nx.shortest_path(self._out, s_id, t_id)
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(self._g, s_id, t_id)
            except nx.NetworkXNoPath:
                path = None
        if not path:
            return QueryResult(
                f"path {from_symbol}->{to_symbol}",
                f"No call path found between {from_symbol} and {to_symbol}.",
                path=[],
            )
        lines = []
        for i, node_id in enumerate(path):
            e = self._by_id[node_id]
            lines.append(f"{i + 1}. {self._format_entity(e)}")
        return QueryResult(
            f"path {from_symbol}->{to_symbol}",
            "Call path:\n" + "\n".join(lines),
            entities=[self._by_id[n] for n in path],
            path=[self._by_id[n]["qualified_name"] for n in path],
        )

    def config_affects(self, config: str) -> QueryResult:
        """Code affected by a config key/file: configuration + constant entities."""
        key = config.strip().lower().replace("-", "_")
        matches = _find_entity(self.entities, config, ("configuration",))
        # config keys often surface as CONSTANTS (e.g. storage_path -> STORAGE_PATH)
        for e in self.entities:
            if e["type"] != "constant":
                continue
            norm = e["name"].lower().replace("_", "")
            if norm == key.replace("_", "") or e["name"].lower() == key:
                matches.append(e)
        if not matches:
            return QueryResult(f"config {config} affects", f"Configuration '{config}' not found.")
        lines = []
        for best in matches[:3]:
            affected = (
                self._neighbors(best["id"], "CONFIGURED_BY", "in")
                + self._neighbors(best["id"], "READS_CONFIG", "in")
                + self._neighbors(best["id"], None, "in")
            )
            seen: set[str] = set()
            deduped = [a for a in affected if not (a["id"] in seen or seen.add(a["id"]))]
            lines.append(
                f"Configuration/constant {best['qualified_name']} [{best['type']}] @ {best['location']} is referenced by:"
            )
            for a in deduped[:10]:
                lines.append(f"  - {self._format_entity(a)}")
            if not deduped:
                lines.append("  (no direct relations — nothing in the index reads it)")
        return QueryResult(f"config {config} affects", "\n".join(lines), entities=matches[:3])

    def files_for(self, symbol: str) -> QueryResult:
        matches = _find_entity(self.entities, symbol)
        if not matches:
            return QueryResult(f"files for {symbol}", f"Symbol '{symbol}' not found.")
        # the symbol's file + files of its direct relation neighbors
        files: dict[str, int] = {}
        for e in matches:
            files[e["file"]] = files.get(e["file"], 0) + 3
        for e in matches:
            for nb in self._neighbors(e["id"], None, "out") + self._neighbors(e["id"], None, "in"):
                if nb.get("file"):
                    files[nb["file"]] = files.get(nb["file"], 0) + 1
        lines = [f"Files involved in '{symbol}':"]
        for f, w in sorted(files.items(), key=lambda kv: -kv[1])[:15]:
            lines.append(f"  - {f} (weight {w})")
        return QueryResult(f"files for {symbol}", "\n".join(lines), entities=matches[:5])

    def overview(self) -> QueryResult:
        types: dict[str, int] = {}
        for e in self.entities:
            types[e["type"]] = types.get(e["type"], 0) + 1
        rels: dict[str, int] = {}
        for r in self.relations:
            rels[r["type"]] = rels.get(r["type"], 0) + 1
        lines = [
            f"Repository: {self.repo}",
            f"Entities: {len(self.entities)}",
            "  by type: "
            + ", ".join(f"{k}={v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])),
            f"Relations: {len(self.relations)}",
            "  by type: "
            + ", ".join(f"{k}={v}" for k, v in sorted(rels.items(), key=lambda kv: -kv[1])),
        ]
        return QueryResult("overview", "\n".join(lines))
