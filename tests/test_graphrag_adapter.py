"""GraphRAG integration: code graph -> GraphRAG tables -> offline pipeline."""

from __future__ import annotations

import json

import pandas as pd

from code_graphrag.config.models import IndexConfig, LLMProvider
from code_graphrag.discovery.discovery import discover_files
from code_graphrag.graphrag.adapter import build_graphrag_config, save_graphrag_settings
from code_graphrag.graphrag.tables import build_graphrag_tables
from code_graphrag.parsing.parse import parse_files
from code_graphrag.semantic.builder import CodeGraphBuilder, FileContext


def _build_graph(sample_project):
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
    graph = CodeGraphBuilder(sample_project.name, ctxs).build()
    return graph, ctxs


def test_build_graphrag_tables_structure(sample_project, tmp_path):
    graph, ctxs = _build_graph(sample_project)
    cfg = IndexConfig()
    cfg.llm.provider = LLMProvider.MOCK
    tables = build_graphrag_tables(graph, ctxs, max_text_unit_chars=2400)

    assert isinstance(tables.documents, pd.DataFrame)
    assert isinstance(tables.text_units, pd.DataFrame)
    assert isinstance(tables.entities, pd.DataFrame)
    assert isinstance(tables.relationships, pd.DataFrame)

    # entities: pre-finalize columns
    assert {"title", "type", "description", "text_unit_ids", "frequency"} <= set(
        tables.entities.columns
    )
    assert len(tables.entities) > 0

    # relationships: pre-finalize columns; EVERY row must carry >=1 text unit id
    assert {"source", "target", "weight", "description", "text_unit_ids"} <= set(
        tables.relationships.columns
    )
    for ids in tables.relationships["text_unit_ids"]:
        assert ids is not None and len(list(ids)) >= 1, "relationship with empty text_unit_ids"

    # titles must be globally unique
    titles = list(tables.entities["title"])
    assert len(titles) == len(set(titles)), "duplicate entity titles"


def test_graphrag_config_uses_mock_and_custom_pipeline(sample_project, tmp_path):
    graph, ctxs = _build_graph(sample_project)
    cfg = IndexConfig()
    cfg.llm.provider = LLMProvider.MOCK
    tables = build_graphrag_tables(graph, ctxs)
    gcfg = build_graphrag_config(cfg, tmp_path, tables)

    # mock completion + embedding models
    comp = gcfg.completion_models["default_completion_model"]
    emb = gcfg.embedding_models["default_embedding_model"]
    assert comp.type == "mock"
    assert emb.type == "mock"
    assert comp.mock_responses, "mock responses must be populated"

    # custom pipeline overrides the stock one (no extract_graph / load_input_documents)
    wf = list(gcfg.workflows)
    assert "load_code_graph" in wf
    assert "create_communities" in wf
    assert "extract_graph" not in wf
    assert "load_input_documents" not in wf

    # use_lcc forced False (case-sensitive code titles)
    assert gcfg.cluster_graph.use_lcc is False


def test_settings_redact_api_key_but_keep_mock(sample_project, tmp_path):
    graph, ctxs = _build_graph(sample_project)
    cfg = IndexConfig()
    cfg.llm.provider = LLMProvider.MOCK
    tables = build_graphrag_tables(graph, ctxs)
    gcfg = build_graphrag_config(cfg, tmp_path, tables)
    save_graphrag_settings(gcfg, tmp_path, cfg)

    data = json.loads((tmp_path / "graphrag_settings.json").read_text())
    comp = data["completion_models"]["default_completion_model"]
    assert comp["api_key"] is None  # redacted
    assert comp["mock_responses"], "mock responses must survive redaction"


def test_full_mock_index_end_to_end(built_index):
    """The session fixture built a full index; verify the artifacts exist."""
    import pandas as pd

    out = built_index / "output"
    for f in (
        "documents.parquet",
        "text_units.parquet",
        "entities.parquet",
        "relationships.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "graph.graphml",
        "stats.json",
    ):
        assert (out / f).exists(), f"missing {f}"

    ents = pd.read_parquet(out / "entities.parquet")
    rels = pd.read_parquet(out / "relationships.parquet")
    assert len(ents) > 0
    assert len(rels) > 0
    # finalized entities carry degree / id
    assert {"title", "id", "degree"} <= set(ents.columns)

    # sidecar present for deterministic queries
    assert (built_index / "code_graphrag_index.json").exists()
    assert (built_index / "graphrag_settings.json").exists()
