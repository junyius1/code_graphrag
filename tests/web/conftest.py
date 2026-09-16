"""Fixtures: a small synthetic Code GraphRAG index (sidecar + parquet)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


def _mk_index(root: Path) -> Path:
    index_dir = root / "idx"
    (index_dir / "output").mkdir(parents=True)

    def ent(
        i: str,
        name: str,
        type_: str,
        file: str,
        sl: int,
        el: int,
        desc: str,
        title: str | None = None,
    ):
        return {
            "id": i,
            "name": name,
            "qualified_name": name,
            "type": type_,
            "file": file,
            "start_line": sl,
            "end_line": el,
            "location": (f"{file}:{sl}-{el}" if sl and el else file),
            "description": desc,
            "language": "python",
            "attributes": {},
            "graphrag_title": title or name,
        }

    entities = [
        ent("e-repo", "demo", "repository", "__repo__", 0, 0, "repo"),
        ent("e-main", "main.py", "file", "main.py", 0, 0, "file main.py"),
        ent("e-mainm", "main", "module", "main.py", 0, 0, "module main"),
        ent("e-cls", "App", "class", "main.py", 1, 20, "class App"),
        ent("e-run", "run", "method", "main.py", 5, 12, "method App.run"),
        ent("e-helper", "helper", "function", "main.py", 15, 18, "function helper"),
        ent("e-util", "utils.py", "file", "utils.py", 0, 0, "file utils.py"),
        ent("e-utilm", "utils", "module", "utils.py", 0, 0, "module utils"),
        ent("e-fmt", "fmt", "function", "utils.py", 1, 3, "function fmt"),
    ]

    def rel(s: str, t: str, type_: str, desc: str):
        return {"source": s, "target": t, "type": type_, "description": desc, "weight": 1.0}

    relations = [
        rel("e-repo", "e-main", "CONTAINS", "demo contains main.py"),
        rel("e-repo", "e-util", "CONTAINS", "demo contains utils.py"),
        rel("e-main", "e-mainm", "DEFINES", "main.py defines main"),
        rel("e-main", "e-cls", "DEFINES", "main.py defines App"),
        rel("e-main", "e-run", "DEFINES", "main.py defines App.run"),
        rel("e-main", "e-helper", "DEFINES", "main.py defines helper"),
        rel("e-util", "e-fmt", "DEFINES", "utils.py defines fmt"),
        rel("e-mainm", "e-utilm", "IMPORTS", "main imports utils"),
        rel("e-run", "e-helper", "CALLS", "App.run calls helper"),
        rel("e-helper", "e-fmt", "CALLS", "helper calls fmt"),
        rel("e-cls", "e-mainm", "CONTAINS", "App contains main"),
    ]
    sidecar = {
        "name": "demo-index",
        "repo": "demo",
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entities": entities,
        "relations": relations,
    }
    (index_dir / "code_graphrag_index.json").write_text(json.dumps(sidecar))

    tu_rows = [
        {
            "id": "tu::doc::main.py::0",
            "text": "[file: main.py] [language: python] [type: file]\nclass App:\n    pass",
            "document_id": "doc::main.py",
        },
        {
            "id": "tu::doc::main.py::1",
            "text": "[file: main.py] [language: python] [type: method] [lines: 5-12]\n    def run(self):\n        return helper()",
            "document_id": "doc::main.py",
        },
        {
            "id": "tu::doc::utils.py::0",
            "text": "[file: utils.py] [language: python] [type: function] [lines: 1-3]\ndef fmt(x):\n    return str(x)",
            "document_id": "doc::utils.py",
        },
        {
            "id": "tu::doc::utils.py::1",
            "text": "[file: utils.py] [language: python] [type: function] [lines: 8-10]\ndef other(x):\n    return x + 1",
            "document_id": "doc::utils.py",
        },
    ]
    pd.DataFrame(tu_rows).to_parquet(index_dir / "output" / "text_units.parquet")
    ent_rows = [
        {
            "id": f"parq-{e['id']}",
            "title": e["graphrag_title"],
            "description": f"semantic: {e['description']}",
            "text_unit_ids": ["tu::doc::main.py::1"] if e["id"] == "e-run" else [],
        }
        for e in entities
    ]
    pd.DataFrame(ent_rows).to_parquet(index_dir / "output" / "entities.parquet")
    return index_dir


@pytest.fixture(scope="session")
def tiny_index(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny_idx")
    return _mk_index(root)


@pytest.fixture(scope="session")
def tiny_adapter(tiny_index):
    # keep the synthetic index on the text-unit fallback path
    os.environ.pop("CODE_GRAPHRAG_SOURCE_ROOT", None)
    from code_graphrag.web.adapter import IndexAdapter

    return IndexAdapter(tiny_index)
