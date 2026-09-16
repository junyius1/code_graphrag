"""Code Graph Explorer backend.

FastAPI service exposing a built Code GraphRAG index (sidecar + parquet
tables) to the explorer frontend:

  GET  /api/health              liveness + index stats
  GET  /api/graph               bounded subgraph (cytoscape nodes/edges)
  GET  /api/file-tree           project tree built from file/directory entities
  GET  /api/entities/{id}       entity detail (precise + semantic description)
  GET  /api/entities/{id}/neighbors   1-2 hop neighborhood, type-filterable
  GET  /api/call-graph/{id}     multi-level CALLS walk (cycle-safe)
  GET  /api/files               indexed file list
  GET  /api/search?q=           entity search
  GET  /api/source              real source text (repo on disk, else index units)
  POST /api/query               NL question -> the existing LangGraph Query Agent

No data is invented: everything is read from the built index. The NL query
reuses ``graphrag_query.agent.build_query_graph`` (the same workflow the CLI
``query --agent`` runs) — this service only adds a thin async wrapper and
maps the agent's evidence items to JSON for the UI.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi import Query as FQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from code_graphrag.logging_setup import get_logger
from code_graphrag.web.adapter import get_adapter
from code_graphrag.web.evidence import map_evidence

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    entity_id: str | None = None
    max_iterations: int = Field(default=6, ge=1, le=12)


def _index_dir() -> str:
    d = os.environ.get("CODE_GRAPHRAG_INDEX_DIR", "")
    if not d:
        raise HTTPException(status_code=500, detail="CODE_GRAPHRAG_INDEX_DIR is not set")
    return d


def create_app(index_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Code Graph Explorer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.query_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cg-query")

    def adapter():
        d = index_dir or _index_dir()
        try:
            return get_adapter(d)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------ #
    # basics
    # ------------------------------------------------------------------ #

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        a = adapter()
        return {
            "ok": True,
            "stats": a.stats(),
            "config": {
                "llm_base_url": os.environ.get("LOCAL_LLM_BASE_URL") or None,
                "llm_model": os.environ.get("LLM_MODEL_ID") or None,
                "source_root": os.environ.get("CODE_GRAPHRAG_SOURCE_ROOT") or None,
            },
        }

    @app.get("/api/vocab")
    def vocab() -> dict[str, Any]:
        a = adapter()
        return {"entity_types": a.entity_types(), "relation_types": a.relation_types()}

    # ------------------------------------------------------------------ #
    # graph
    # ------------------------------------------------------------------ #

    @app.get("/api/graph")
    def graph(
        seed: str | None = None,
        seed_ids: str | None = None,
        depth: int = FQuery(default=1, ge=1, le=3),
        limit: int = FQuery(default=250, ge=10, le=800),
        types: str | None = None,
        rel: str | None = None,
    ) -> dict[str, Any]:
        a = adapter()
        ids: list[str] = []
        if seed_ids:
            ids = [s for s in seed_ids.split(",") if s in a.by_id]
        elif seed:
            matches = a.search(seed, limit=1)
            if not matches:
                raise HTTPException(status_code=404, detail=f"no entity matches '{seed}'")
            ids = [matches[0]["id"]]
        type_list = [t for t in (types or "").split(",") if t] or None
        rel_list = [t for t in (rel or "").split(",") if t] or None
        return a.subgraph(
            seed_ids=ids or None, depth=depth, limit=limit, types=type_list, rel=rel_list
        )

    @app.get("/api/file-tree")
    def file_tree() -> dict[str, Any]:
        return {"tree": adapter().file_tree()}

    @app.get("/api/files")
    def files(
        q: str | None = None, limit: int = FQuery(default=500, ge=1, le=2000)
    ) -> dict[str, Any]:
        a = adapter()
        out = [
            {
                "id": e["id"],
                "path": e.get("file"),
                "module_entity_id": a._module_id_for(e.get("file", "")),
            }
            for e in a.entities
            if e.get("type") == "file"
        ]
        if q:
            ql = q.lower()
            out = [f for f in out if ql in f["path"]]
        out.sort(key=lambda f: f["path"])
        return {"files": out[:limit], "total": len(out)}

    @app.get("/api/files/{file_id}/entities")
    def file_entities(file_id: str) -> dict[str, Any]:
        a = adapter()
        e = a.by_id.get(file_id)
        if e is None:
            raise HTTPException(status_code=404, detail="file entity not found")
        return {"file": e.get("file"), "entities": a.file_entities(e.get("file", ""))}

    # ------------------------------------------------------------------ #
    # entities
    # ------------------------------------------------------------------ #

    @app.get("/api/entities/{entity_id}")
    def entity(entity_id: str) -> dict[str, Any]:
        a = adapter()
        out = a.entity(entity_id)
        if out is None:
            raise HTTPException(status_code=404, detail="entity not found")
        return out

    @app.get("/api/entities/{entity_id}/neighbors")
    def neighbors(
        entity_id: str,
        depth: int = FQuery(default=1, ge=1, le=2),
        limit: int = FQuery(default=60, ge=5, le=400),
        rel: str | None = None,
        direction: str = FQuery(default="both", pattern="^(out|in|both)$"),
    ) -> dict[str, Any]:
        a = adapter()
        if entity_id not in a.by_id:
            raise HTTPException(status_code=404, detail="entity not found")
        return a.neighbors(entity_id, depth=depth, limit=limit, rel=rel, direction=direction)

    @app.get("/api/call-graph/{entity_id}")
    def call_graph(
        entity_id: str,
        depth: int = FQuery(default=3, ge=1, le=6),
        limit: int = FQuery(default=200, ge=10, le=600),
        direction: str = FQuery(default="both", pattern="^(out|in|both)$"),
    ) -> dict[str, Any]:
        a = adapter()
        if entity_id not in a.by_id:
            raise HTTPException(status_code=404, detail="entity not found")
        return a.call_chain(entity_id, depth=depth, limit=limit, direction=direction)

    @app.get("/api/search")
    def search(
        q: str = FQuery(min_length=1),
        types: str | None = None,
        limit: int = FQuery(default=20, ge=1, le=50),
    ) -> dict[str, Any]:
        a = adapter()
        type_list = [t for t in (types or "").split(",") if t] or None
        return {"results": a.search(q, types=type_list, limit=limit)}

    # ------------------------------------------------------------------ #
    # source
    # ------------------------------------------------------------------ #

    @app.get("/api/source")
    def source(
        path: str | None = None,
        entity_id: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        a = adapter()
        if entity_id:
            e = a.by_id.get(entity_id)
            if e is None:
                raise HTTPException(status_code=404, detail="entity not found")
            out = a.source_for_entity(e)
        elif path:
            out = a.source_for_file(path)
            if out["text"] is not None and start_line and end_line and end_line >= start_line:
                lines = out["text"].splitlines()
                if len(lines) >= start_line:
                    out["text"] = "\n".join(lines[start_line - 1 : end_line])
        else:
            raise HTTPException(status_code=400, detail="path or entity_id required")
        return out

    # ------------------------------------------------------------------ #
    # LLM query (reuses the existing LangGraph Query Agent)
    # ------------------------------------------------------------------ #

    @app.post("/api/query")
    async def query(req: QueryRequest) -> dict[str, Any]:
        a = adapter()
        question = req.question
        if req.entity_id:
            e = a.by_id.get(req.entity_id)
            if e is not None:
                question = (
                    f"About the entity '{e.get('qualified_name')}' [{e.get('type')}] "
                    f"(file: {e.get('file')}, location: {e.get('location')}): {question}"
                )
        from langchain_core.messages import HumanMessage

        from code_graphrag.config.models import LLMConfig, LLMProvider
        from code_graphrag.llm.models import create_chat_model
        from graphrag_query.agent import _resolve_local, build_query_graph
        from graphrag_query.context import ContextManager
        from graphrag_query.trace import Tracer

        def _cfg() -> LLMConfig:
            cfg = LLMConfig()
            base = os.environ.get("LOCAL_LLM_BASE_URL")
            model = os.environ.get("LLM_MODEL_ID")
            if base:
                cfg.provider = LLMProvider.OPENAI_COMPATIBLE
                cfg.base_url = base
                if model:
                    cfg.model = model if model.startswith("local:") else f"local:{model}"
                cfg.api_key = os.environ.get("LOCAL_LLM_API_KEY") or "sk-no-key-offline"
            return cfg

        cfg = _resolve_local(_cfg())
        try:
            model = create_chat_model(cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"cannot create LLM model: {exc}") from exc

        data = a.query_data  # graphrag_query.QueryData (lazy, cached)
        ctx = ContextManager()
        tracer = Tracer()
        g = build_query_graph(data, model, ctx, tracer, max_iterations=req.max_iterations)

        def _run() -> dict[str, Any]:
            return g.invoke({"messages": [HumanMessage(content=question)]})

        try:
            result = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            logger.exception("query agent failed")
            raise HTTPException(status_code=502, detail=f"query agent failed: {exc}") from exc

        answer = result.get("final_answer") or ""
        if not answer:
            from langchain_core.messages import AIMessage

            ai = [m for m in result.get("messages", []) if isinstance(m, AIMessage) and m.content]
            answer = ai[-1].content if ai else "(agent produced no answer)"
        return {
            "question": question,
            "answer": answer,
            "analysis": result.get("analysis", {}),
            "tool_calls": result.get("tool_calls", []),
            "evidence": map_evidence(ctx.items, a),
            "stats": ctx.summary(),
        }

    # ------------------------------------------------------------------ #
    # frontend static files (if built)
    # ------------------------------------------------------------------ #

    frontend = _find_frontend_dist()
    if frontend is not None:
        app.mount(
            "/assets",
            StaticFiles(directory=frontend / "assets"),
            name="assets",
        )

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            f = frontend / path
            if path and f.is_file():
                return FileResponse(f)
            return FileResponse(frontend / "index.html")

    return app


def _find_frontend_dist() -> Path | None:
    """Locate the built explorer frontend: env override, then common layouts."""
    env = os.environ.get("CODE_GRAPHRAG_WEB_DIST")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    root = Path(__file__).resolve().parents[3]
    for cand in (
        root / "frontend" / "dist",
        root / "dist",
        root.parent / "code_browser" / "dist",  # standalone repo layout
    ):
        if cand.is_dir() and (cand / "index.html").exists():
            return cand
    return None


app = create_app(os.environ.get("CODE_GRAPHRAG_INDEX_DIR"))
