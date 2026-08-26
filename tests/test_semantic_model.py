"""Semantic model: stable entity ids, deduplication, graph construction."""

from __future__ import annotations

from code_graphrag.semantic.builder import CodeGraphBuilder, FileContext
from code_graphrag.semantic.model import CodeKnowledgeGraph, stable_entity_id


def test_stable_entity_id_deterministic_and_unique():
    a = stable_entity_id("function", "repo", "m.py", "m.main")
    b = stable_entity_id("function", "repo", "m.py", "m.main")
    c = stable_entity_id("function", "repo", "m.py", "m.other")
    d = stable_entity_id("class", "repo", "m.py", "m.main")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 20


def _ctx(rel: str, src: str) -> FileContext:
    return FileContext(
        rel_path=rel, kind="source", structure=None, text=src, line_count=src.count("\n")
    )


def test_entity_deduplication_same_symbol():
    g = CodeKnowledgeGraph(repo="r")
    e1 = g.entity("function", "main", "m.main", "m.py")
    e2 = g.entity("function", "main", "m.main", "m.py")
    assert e1.id == e2.id
    assert len(g.entities) == 1


def test_entity_no_dedup_across_types():
    g = CodeKnowledgeGraph(repo="r")
    f = g.entity("function", "handle", "m.handle", "m.py")
    c = g.entity("class", "handle", "m.handle", "m.py")
    assert f.id != c.id
    assert len(g.entities) == 2


def test_relate_deduplicates_and_builds_graph():
    g = CodeKnowledgeGraph(repo="r")
    a = g.entity("module", "app", "app", "app.py")
    b = g.entity("module", "svc", "svc", "svc.py")
    r1 = g.relate(a.id, b.id, "IMPORTS", "app imports svc")
    r2 = g.relate(a.id, b.id, "IMPORTS", "app imports svc")  # duplicate
    assert r1.key() == r2.key()
    assert r2 is r1  # deduped to the same object (weight summed)
    assert len(g.relations) == 1
    by_type = g.stats()["by_relation_type"]
    assert by_type.get("IMPORTS") == 1


def test_build_full_graph_on_sample(sample_project):
    from code_graphrag.config.models import IndexConfig
    from code_graphrag.discovery.discovery import discover_files
    from code_graphrag.parsing.parse import parse_files

    cfg = IndexConfig()
    cfg.discovery.source = str(sample_project)
    disc = discover_files(sample_project, cfg)
    parsed = parse_files(disc, cfg)
    ctxs = {}
    for f in disc.sources:
        ctxs[f.rel_path] = FileContext(
            rel_path=f.rel_path,
            kind="source",
            structure=parsed.structures.get(f.rel_path),
            text=parsed.text_cache.get(f.rel_path, ""),
            line_count=len((parsed.text_cache.get(f.rel_path) or "").splitlines()),
        )
    builder = CodeGraphBuilder(sample_project.name, ctxs)
    g = builder.build()

    # core entities present
    names = {e.qualified_name for e in g.entities.values()}
    assert "UserService" in names
    assert "UserService.register" in names
    assert "main" in names
    assert "App" in names

    # relations present
    rel_types = {r.type for r in g.relations.values()}
    assert {"CONTAINS", "DEFINES", "IMPORTS", "CALLS", "INHERITS", "ACCEPTS_PARAMETER"} <= rel_types

    # every relation endpoint references a real entity (graph integrity)
    ids = set(g.entities)
    for r in g.relations.values():
        assert r.source_id in ids, f"dangling source {r.source_id}"
        assert r.target_id in ids, f"dangling target {r.target_id}"

    # inheritance chain BaseUser <- User <- AdminUser
    inh = {
        (g.entities[r.source_id].qualified_name, g.entities[r.target_id].qualified_name)
        for r in g.relations.values()
        if r.type == "INHERITS"
    }
    assert ("User", "BaseUser") in inh
    assert ("AdminUser", "User") in inh
