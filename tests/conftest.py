"""Shared fixtures for the Code GraphRAG test suite.

Everything runs offline (mock LLM + mock embeddings) against the checked-in
``tests/sample_project`` so the suite needs no network or API keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_graphrag.config.models import IndexConfig, LLMProvider

SAMPLE_PROJECT = Path(__file__).parent / "sample_project"


def make_mock_config(source: str, output: str) -> IndexConfig:
    cfg = IndexConfig()
    cfg.discovery.source = source
    cfg.output_dir = output
    cfg.llm.provider = LLMProvider.MOCK
    cfg.verbose = False
    return cfg


@pytest.fixture(scope="session")
def sample_project() -> Path:
    return SAMPLE_PROJECT


@pytest.fixture(scope="session")
def built_index(tmp_path_factory) -> Path:
    """Build a full mock-mode index once per session and return its dir."""
    import asyncio

    from code_graphrag.workflow.index_workflow import run_index_workflow

    out = tmp_path_factory.mktemp("cg_index")
    cfg = make_mock_config(str(SAMPLE_PROJECT), str(out))
    asyncio.run(run_index_workflow(str(SAMPLE_PROJECT), str(out), cfg))
    return out
