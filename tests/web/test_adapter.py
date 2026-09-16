"""Unit tests for the explorer data adapter over a synthetic index."""

from __future__ import annotations


def test_stats_and_vocab(tiny_adapter):
    s = tiny_adapter.stats()
    assert s["entity_count"] == 9
    assert s["relation_count"] == 11
    assert s["by_entity_type"]["function"] == 2
    assert "CALLS" in tiny_adapter.relation_types()
    assert "class" in tiny_adapter.entity_types()


def test_subgraph_default_is_repo_first_frame(tiny_adapter):
    g = tiny_adapter.subgraph(depth=1, limit=100)
    labels = {n["label"] for n in g["nodes"]}
    assert "demo" in labels
    assert "main.py" in labels  # repo CONTAINS file
    assert all(n["id"] and n["type"] for n in g["nodes"])
    assert all(e["source"] and e["target"] and e["type"] for e in g["edges"])


def test_subgraph_seed_depth_and_limit(tiny_adapter):
    g = tiny_adapter.subgraph(seed_ids=["e-run"], depth=2, limit=100)
    ids = {n["id"] for n in g["nodes"]}
    assert {"e-run", "e-helper", "e-fmt"} <= ids  # 2-hop CALLS chain
    g1 = tiny_adapter.subgraph(seed_ids=["e-run"], depth=1, limit=100)
    ids1 = {n["id"] for n in g1["nodes"]}
    assert "e-fmt" not in ids1  # only 1 hop
    g_small = tiny_adapter.subgraph(depth=1, limit=3)
    assert len(g_small["nodes"]) <= 3


def test_subgraph_relation_filter(tiny_adapter):
    g = tiny_adapter.subgraph(seed_ids=["e-run"], depth=1, rel=["CALLS"])
    types = {e["type"] for e in g["edges"]}
    assert types == {"CALLS"}


def test_search_ranking(tiny_adapter):
    res = tiny_adapter.search("helper", limit=5)
    assert res[0]["name"] == "helper"
    assert res[0]["type"] == "function"
    assert tiny_adapter.search("zzz-nope") == []
    res_cls = tiny_adapter.search("App", types=["class"])
    assert [r["name"] for r in res_cls] == ["App"]


def test_entity_detail(tiny_adapter):
    e = tiny_adapter.entity("e-run")
    assert e["name"] == "run"
    assert e["type"] == "method"
    assert e["file"] == "main.py"
    assert e["start_line"] == 5 and e["end_line"] == 12
    assert "method App.run" in e["description"]
    assert "semantic:" in e["semantic_description"]
    assert tiny_adapter.entity("missing") is None


def test_neighbors_direction_and_filter(tiny_adapter):
    n = tiny_adapter.neighbors("e-run", depth=1, rel="CALLS", direction="out")
    edges = n["edges"]
    assert len(edges) == 1 and edges[0]["target"] == "e-helper"
    n_in = tiny_adapter.neighbors("e-helper", depth=1, rel="CALLS", direction="in")
    assert any(e["source"] == "e-run" for e in n_in["edges"])
    n_both = tiny_adapter.neighbors("e-run", depth=1)
    assert len(n_both["nodes"]) >= 3


def test_call_chain_cycle_safe_and_bounded(tiny_adapter):
    # helper <-> fmt would cycle if mutual; here chain is run->helper->fmt
    cg = tiny_adapter.call_chain("e-run", depth=6, limit=50, direction="out")
    ids = [n["id"] for n in cg["nodes"]]
    assert ids[0] == "e-run"
    assert "e-fmt" in ids
    assert cg["truncated"] is False
    # in direction: run has no callers
    cg_in = tiny_adapter.call_chain("e-run", depth=2, direction="in")
    assert (
        cg_in["nodes"] == [tiny_adapter._compact_node(tiny_adapter.by_id["e-run"])]
        or len(cg_in["nodes"]) == 1
    )
    # unknown entity
    assert tiny_adapter.call_chain("nope", depth=1) == {"root": None, "nodes": [], "edges": []}


def test_file_tree(tiny_adapter):
    tree = tiny_adapter.file_tree()
    assert tree["name"] == "demo"
    assert set(tree["children"]) == {"main.py", "utils.py"}
    assert tree["children"]["main.py"]["entity_id"] == "e-main"
    assert tree["children"]["main.py"]["module_entity_id"] == "e-mainm"


def test_file_entities(tiny_adapter):
    ents = tiny_adapter.file_entities("main.py")
    names = {e["name"] for e in ents}
    assert names == {"App", "run", "helper"}
    assert tiny_adapter.file_entities("missing.py") == []


def test_source_from_units_file(tiny_adapter):
    s = tiny_adapter.source_for_file("utils.py")
    assert s["origin"] == "index:units"
    assert "def fmt(x):" in s["text"]
    assert "[4-7 not stored in index]" in s["text"]  # gap between the two stored chunks
    s2 = tiny_adapter.source_for_file("missing.py")
    assert s2["text"] is None and s2["origin"] is None


def test_source_for_entity_method(tiny_adapter):
    s = tiny_adapter.source_for_entity(tiny_adapter.by_id["e-run"])
    assert s["origin"] == "index:units"
    assert "def run(self):" in s["text"]
    assert s["start_line"] == 5 and s["end_line"] == 12


def test_invalid_relations_are_dropped(tmp_path_factory):
    """Relations pointing at unknown entities must not break loading."""
    import json

    from code_graphrag.web.adapter import IndexAdapter

    root = tmp_path_factory.mktemp("bad_idx")
    (root / "output").mkdir(parents=True)
    sidecar = {
        "name": "x",
        "repo": "x",
        "entities": [
            {
                "id": "a",
                "name": "a",
                "qualified_name": "a",
                "type": "file",
                "file": "a.py",
                "start_line": 0,
                "end_line": 0,
                "location": "a.py",
                "description": "",
                "language": "python",
                "attributes": {},
                "graphrag_title": "a",
            }
        ],
        "relations": [
            {
                "source": "a",
                "target": "ghost",
                "type": "CALLS",
                "description": "a calls ghost",
                "weight": 1.0,
            },
        ],
    }
    (root / "code_graphrag_index.json").write_text(json.dumps(sidecar))
    a = IndexAdapter(root)
    assert len(a.outgoing) == 0
    assert a.subgraph(seed_ids=["a"])["edges"] == []
