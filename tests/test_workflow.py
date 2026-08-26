"""LangGraph workflow orchestration: index pipeline + query pipeline."""

from __future__ import annotations


async def test_index_workflow_end_to_end(tmp_path, sample_project):
    from code_graphrag.config.models import IndexConfig, LLMProvider
    from code_graphrag.workflow.index_workflow import run_index_workflow

    cfg = IndexConfig()
    cfg.llm.provider = LLMProvider.MOCK
    out = tmp_path / "idx"
    state = await run_index_workflow(str(sample_project), str(out), cfg)
    assert state.get("error") is None
    assert state.get("status") == "done"
    # deterministic summary fields surfaced in final state
    assert state.get("entities", 0) > 0
    assert state.get("relations", 0) > 0
    # artifacts written
    assert (out / "code_graphrag_index.json").exists()
    assert (out / "graphrag_settings.json").exists()
    assert (out / "output" / "community_reports.parquet").exists()


async def test_query_workflow_deterministic(built_index):
    from code_graphrag.workflow.query_workflow import run_query_workflow

    state = await run_query_workflow(built_index, "Where is the handle method implemented?")
    assert state.get("route") == "where"
    assert "App.handle" in state.get("answer", "")
    assert state.get("source", "").startswith("code_graph:")


async def test_query_workflow_forced_semantic(built_index):
    from code_graphrag.workflow.query_workflow import run_query_workflow

    state = await run_query_workflow(
        built_index, "How does user registration work?", forced_mode="local"
    )
    assert state.get("route") == "local"
    assert state.get("error") is None
    assert state.get("answer")


async def test_query_workflow_call_path(built_index):
    from code_graphrag.workflow.query_workflow import run_query_workflow

    state = await run_query_workflow(built_index, "Show the call path from main to send_email")
    assert state.get("route") == "path"
    assert "main" in state.get("answer", "")
