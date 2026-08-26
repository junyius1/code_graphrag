"""File discovery: recursion, classification, and filtering."""

from __future__ import annotations

from pathlib import Path

from code_graphrag.config.models import IndexConfig
from code_graphrag.discovery.discovery import discover_files


def _cfg(root: Path) -> IndexConfig:
    cfg = IndexConfig()
    cfg.discovery.source = str(root)
    return cfg


def test_discover_sample_project(sample_project: Path):
    cfg = _cfg(sample_project)
    res = discover_files(sample_project, cfg)
    rels = {f.rel_path for f in res.files}
    # 5 python sources, 1 markdown doc, 1 json config
    assert len(res.sources) == 5
    assert len(res.docs) == 1
    assert len(res.configs) == 1
    assert "app.py" in rels
    assert "cli.py" in rels
    assert "models.py" in rels
    assert "services.py" in rels
    assert "test_services.py" in rels
    assert "README.md" in rels
    assert "config.json" in rels


def test_discovery_kind_classification(sample_project: Path):
    cfg = _cfg(sample_project)
    res = discover_files(sample_project, cfg)
    kinds = {f.rel_path: f.kind for f in res.files}
    assert kinds["app.py"] == "source"
    assert kinds["README.md"] == "doc"
    assert kinds["config.json"] == "config"


def test_discovery_skips_excluded_dir(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("X = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("Y = 2\n")
    cfg = _cfg(tmp_path)
    res = discover_files(tmp_path, cfg)
    rels = {f.rel_path for f in res.files}
    assert "src/a.py" in rels
    assert not any("node_modules" in r for r in rels)


def test_discovery_skips_binary(tmp_path: Path):
    (tmp_path / "ok.py").write_text("A = 1\n")
    (tmp_path / "blob.py").write_bytes(b"\x00\x01\x02\x03")  # NUL -> binary
    cfg = _cfg(tmp_path)
    res = discover_files(tmp_path, cfg)
    rels = {f.rel_path for f in res.files}
    assert "ok.py" in rels
    assert "blob.py" not in rels
    assert "blob.py" in res.skipped_binary


def test_discovery_skips_unmatched_extension(tmp_path: Path):
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "notes.xyz").write_text("hello\n")
    cfg = _cfg(tmp_path)
    res = discover_files(tmp_path, cfg)
    rels = {f.rel_path for f in res.files}
    assert "a.py" in rels
    assert "notes.xyz" not in rels
