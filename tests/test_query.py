"""Query layer: deterministic code-graph queries + routing + GraphRAG search."""

from __future__ import annotations

import pytest

from code_graphrag.query.queries import CodeGraphQuery
from code_graphrag.query.routing import (
    detect_deterministic_mode,
    extract_path_symbols,
    extract_symbol,
)


@pytest.fixture(scope="module")
def q(built_index):
    return CodeGraphQuery(built_index)


# ---- routing ---------------------------------------------------------------


def test_routing_deterministic():
    cases = {
        "Where is the handle method implemented?": "where",
        "What are the callers of send_email?": "callers",
        "What does handle_register call?": "callees",
        "What modules does app import?": "imports",
        "Who imports the services module?": "imported_by",
        "What class inherits from User?": "inherits",
        "What code is affected by the storage_path config?": "config",
        "Which files are involved in UserService?": "files",
        "Show the call path from main to send_email": "path",
    }
    for question, expected in cases.items():
        assert detect_deterministic_mode(question) == expected, question


def test_routing_llm_fallback():
    for question in (
        "How does user registration work end to end?",
        "Explain the overall architecture and how modules relate",
    ):
        assert detect_deterministic_mode(question) is None, question


def test_extract_symbol_prefers_code_tokens():
    assert extract_symbol("What does UserService.register do?") in (
        "UserService.register",
        "UserService",
    )
    assert extract_symbol("Explain send_email usage") == "send_email"
    assert extract_symbol("Where is `handle` defined?") == "handle"


def test_extract_path_symbols():
    a, b = extract_path_symbols("call path from main to send_email")
    assert a == "main"
    assert b == "send_email"


# ---- deterministic queries -------------------------------------------------


def test_where_is(q):
    r = q.where_is("UserService.register")
    assert "UserService.register" in r.answer
    assert r.entities and r.entities[0]["qualified_name"] == "UserService.register"
    assert "services.py" in r.entities[0]["location"]


def test_callers(q):
    r = q.callers("send_email")
    assert "App.handle_notify" in r.answer
    assert "main" in r.answer


def test_callees(q):
    r = q.callees("handle_register")
    assert "UserService.register" in r.answer


def test_imports_of(q):
    r = q.imports_of("app")
    assert "services" in r.answer


def test_imported_by(q):
    r = q.imported_by("services")
    for mod in ("app", "cli", "test_services"):
        assert mod in r.answer


def test_inherits(q):
    r = q.inherits("User")
    assert "BaseUser" in r.answer  # superclass
    assert "AdminUser" in r.answer  # subclass


def test_call_path(q):
    r = q.call_path("main", "send_email")
    assert r.path and r.path[0] == "main" and r.path[-1] == "NotificationService.send_email"


def test_config_affects(q):
    r = q.config_affects("storage_path")
    assert "STORAGE_PATH" in r.answer or "not found" not in r.answer


def test_files_for(q):
    r = q.files_for("UserService")
    assert "services.py" in r.answer


def test_overview(q):
    r = q.overview()
    assert "Entities:" in r.answer
    assert "Relations:" in r.answer


def test_unknown_symbol_is_graceful(q):
    r = q.where_is("does_not_exist_xyz")
    assert "not found" in r.answer.lower()


# ---- graphrag semantic search (mock, offline) -------------------------------


def test_graphrag_local_search_mock(built_index):
    from code_graphrag.query.graphrag_search import graphrag_search

    r = graphrag_search(built_index, "How does user registration work?", search_type="local")
    assert r.error is None
    assert r.answer


def test_graphrag_global_search_mock(built_index):
    from code_graphrag.query.graphrag_search import graphrag_search

    r = graphrag_search(built_index, "What is this repository about?", search_type="global")
    assert r.error is None
    assert r.answer


def test_graphrag_missing_settings_errors(tmp_path):
    from code_graphrag.query.graphrag_search import graphrag_search

    r = graphrag_search(tmp_path, "anything")
    assert r.error is not None
