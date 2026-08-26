"""Deep Agents integration: code-exploration agent over the index (offline)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from code_graphrag.config.models import LLMConfig, LLMProvider
from code_graphrag.llm.models import _MockChatModel


def test_build_code_tools_names(built_index):
    from code_graphrag.agents import build_code_tools

    tools = build_code_tools(built_index)
    names = {t.name for t in tools}
    assert {
        "code_overview",
        "where_is",
        "callers",
        "callees",
        "imports_of",
        "imported_by",
        "inheritance",
        "call_path",
        "related_files",
        "config_impact",
        "graphrag_local_search",
        "graphrag_global_search",
    } <= names


def test_code_tool_returns_grounded_answer(built_index):
    from code_graphrag.agents import build_code_tools

    tools = {t.name: t for t in build_code_tools(built_index)}
    out = tools["where_is"].invoke({"symbol": "UserService.register"})
    assert "UserService.register" in out
    assert "services.py" in out


def test_mock_model_emits_tool_call_then_answer():
    from langchain_core.messages import ToolMessage

    m = _MockChatModel()
    first = m.invoke([HumanMessage(content="What does send_email do?")])
    assert first.tool_calls and first.tool_calls[0]["name"] == "where_is"
    assert first.tool_calls[0]["args"]["symbol"] == "send_email"
    second = m.invoke(
        [
            HumanMessage(content="What does send_email do?"),
            ToolMessage(
                content="NotificationService.send_email @ services.py:47", tool_call_id="x"
            ),
        ]
    )
    assert second.tool_calls == []
    assert "Mock answer" in second.content


def test_run_code_agent_offline(built_index, sample_project):
    from code_graphrag.agents import run_code_agent

    cfg = LLMConfig()
    cfg.provider = LLMProvider.MOCK
    answer = run_code_agent(
        built_index,
        "What does the handle method do?",
        source_dir=str(sample_project),
        llm_config=cfg,
    )
    assert isinstance(answer, str) and answer
