"""API tests for the explorer backend (synthetic index, no LLM)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def client(tiny_index: Path):
    os.environ.pop("CODE_GRAPHRAG_SOURCE_ROOT", None)
    from fastapi.testclient import TestClient

    from code_graphrag.web.app import create_app

    app = create_app(str(tiny_index))
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["stats"]["entity_count"] == 9
    assert body["stats"]["name"] == "demo-index"


def test_graph_default_and_seed(client):
    r = client.get("/api/graph")
    assert r.status_code == 200
    g = r.json()
    assert g["nodes"] and g["edges"]
    assert any(n["label"] == "demo" for n in g["nodes"])

    r = client.get("/api/graph", params={"seed": "helper", "depth": 2})
    assert r.status_code == 200
    ids = {n["id"] for n in r.json()["nodes"]}
    assert "e-run" in ids  # caller found via incoming edge
    assert "e-fmt" in ids  # callee found via outgoing edge

    r = client.get("/api/graph", params={"seed": "zzz-nope"})
    assert r.status_code == 404


def test_graph_limits_respected(client):
    r = client.get("/api/graph", params={"limit": 10, "depth": 3})
    assert r.status_code == 200
    assert len(r.json()["nodes"]) <= 10


def test_file_tree_and_files(client):
    r = client.get("/api/file-tree")
    assert r.status_code == 200
    tree = r.json()["tree"]
    assert tree["name"] == "demo"
    assert "main.py" in tree["children"]

    r = client.get("/api/files")
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()["files"]]
    assert paths == ["main.py", "utils.py"]

    r = client.get("/api/files", params={"q": "util"})
    assert [f["path"] for f in r.json()["files"]] == ["utils.py"]


def test_file_entities_endpoint(client):
    r = client.get("/api/files/e-main/entities")
    assert r.status_code == 200
    names = {e["name"] for e in r.json()["entities"]}
    assert names == {"App", "run", "helper"}
    assert client.get("/api/files/nope/entities").status_code == 404


def test_entity_and_neighbors(client):
    r = client.get("/api/entities/e-run")
    assert r.status_code == 200
    e = r.json()
    assert e["name"] == "run" and e["file"] == "main.py"
    assert e["start_line"] == 5

    r = client.get("/api/entities/e-run/neighbors", params={"rel": "CALLS", "direction": "out"})
    assert r.status_code == 200
    assert [ed["target"] for ed in r.json()["edges"]] == ["e-helper"]

    assert client.get("/api/entities/nope").status_code == 404
    assert client.get("/api/entities/nope/neighbors").status_code == 404


def test_call_graph_endpoint(client):
    r = client.get("/api/call-graph/e-run", params={"depth": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["root"] == "e-run"
    ids = {n["id"] for n in body["nodes"]}
    assert {"e-helper", "e-fmt"} <= ids
    assert all(ed["type"] == "CALLS" for ed in body["edges"])
    assert client.get("/api/call-graph/nope").status_code == 404


def test_search_endpoint(client):
    r = client.get("/api/search", params={"q": "App"})
    assert r.status_code == 200
    assert r.json()["results"][0]["name"] == "App"
    assert client.get("/api/search").status_code == 422


def test_source_endpoints(client):
    # entity source from index units (no source root configured)
    r = client.get("/api/source", params={"entity_id": "e-run"})
    assert r.status_code == 200
    s = r.json()
    assert s["origin"] == "index:units"
    assert "def run(self):" in s["text"]

    # file source from units
    r = client.get("/api/source", params={"path": "utils.py"})
    assert r.status_code == 200
    assert "def fmt(x):" in r.json()["text"]

    # missing
    r = client.get("/api/source", params={"path": "missing.py"})
    assert r.status_code == 200
    assert r.json()["text"] is None

    # no args
    assert client.get("/api/source").status_code == 400


def test_source_from_repo_when_root_set(tmp_path):
    """With CODE_GRAPHRAG_SOURCE_ROOT set, the on-disk repo wins (file + slice)."""
    import os

    from code_graphrag.web import adapter as adapter_mod
    from tests.web.conftest import _mk_index

    repo = tmp_path / "repo"
    repo.mkdir()
    # 12 lines (no trailing newline on last) so entity slice 5-12 is valid
    (repo / "main.py").write_text("".join(f"l{i}\n" for i in range(1, 12)) + "l12")
    old = os.environ.get("CODE_GRAPHRAG_SOURCE_ROOT")
    os.environ["CODE_GRAPHRAG_SOURCE_ROOT"] = str(repo)
    adapter_mod._STORES.clear()
    try:
        idx = _mk_index(tmp_path / "idx2")
        a = adapter_mod.IndexAdapter(idx)
        s = a.source_for_file("main.py")
        assert s["origin"] == "repo"
        assert s["line_count"] == 12
        e = a.by_id["e-run"]
        s2 = a.source_for_entity(e)
        assert s2["origin"] == "repo"
        assert s2["text"] == "l5\nl6\nl7\nl8\nl9\nl10\nl11\nl12"
        # entity with no lines -> falls back to units
        s3 = a.source_for_entity(a.by_id["e-cls"])
        assert s3["origin"] in (None, "index:units")
    finally:
        if old is None:
            os.environ.pop("CODE_GRAPHRAG_SOURCE_ROOT", None)
        else:
            os.environ["CODE_GRAPHRAG_SOURCE_ROOT"] = old
        adapter_mod._STORES.clear()
