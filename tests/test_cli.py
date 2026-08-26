"""CLI end-to-end: index / query / inspect / explore via Typer's runner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from code_graphrag.cli.app import app

runner = CliRunner()


def test_cli_index_mock(tmp_path, sample_project):
    out = tmp_path / "idx"
    res = runner.invoke(
        app,
        ["index", "--source", str(sample_project), "--output", str(out), "--mock"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert "Index build complete." in res.output
    assert (out / "code_graphrag_index.json").exists()
    # summary line appears
    assert "entities" in res.output


def test_cli_query_deterministic(built_index):
    res = runner.invoke(
        app,
        [
            "query",
            "--index",
            str(built_index),
            "--question",
            "Where is the handle method implemented?",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "App.handle" in res.output
    assert "code_graph:" in res.output


def test_cli_query_json(built_index):
    res = runner.invoke(
        app,
        [
            "query",
            "--index",
            str(built_index),
            "--json",
            "--question",
            "Where is UserService.register implemented?",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["source"] == "code_graph:where"
    assert payload["entities"]


def test_cli_query_semantic_mock(built_index):
    res = runner.invoke(
        app,
        [
            "query",
            "--index",
            str(built_index),
            "--mode",
            "local",
            "--question",
            "How does the notification flow work?",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "graphrag:local" in res.output


def test_cli_inspect(built_index):
    res = runner.invoke(
        app,
        ["inspect", "--index", str(built_index), "--entity", "UserService"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "UserService" in res.output
