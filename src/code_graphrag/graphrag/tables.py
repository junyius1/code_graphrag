"""Convert the Code Knowledge Graph into GraphRAG v3 pipeline tables.

GraphRAG v3 consumes plain DataFrames written to its output table provider.
The supported seam is a custom workflow (``config.workflows``) that writes the
deterministic tables directly, so we skip GraphRAG's file-loading + chunking +
LLM-extraction steps and feed it our pre-built, structurally-grounded graph.

Critical invariant (from graphrag ``finalize_entities``): **entity ``title`` is
the unique key** — duplicates are dropped and community detection maps
title -> entity id. So titles must be globally unique. We base titles on the
symbol ``qualified_name`` and disambiguate any collision with the file path.
Relationship ``source``/``target`` must equal entity titles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from code_graphrag.semantic.builder import FileContext
from code_graphrag.semantic.model import CodeEntity, CodeKnowledgeGraph


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _metadata_header(entity: CodeEntity) -> str:
    parts = [f"[file: {entity.file}]"]
    if entity.language:
        parts.append(f"[language: {entity.language}]")
    parts.append(f"[type: {entity.type}]")
    if entity.start_line:
        rng = (
            f"{entity.start_line}-{entity.end_line}"
            if entity.end_line > entity.start_line
            else f"{entity.start_line}"
        )
        parts.append(f"[lines: {rng}]")
    return " ".join(parts) + "\n"


def _symbol_source(ctx: FileContext, entity: CodeEntity) -> str:
    if not ctx.text or not entity.start_line:
        return ""
    lines = ctx.text.splitlines()
    start = max(0, entity.start_line - 1)
    end = min(len(lines), entity.end_line)
    if end <= start:
        end = min(len(lines), start + 1)
    return "\n".join(lines[start:end]).strip()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf = ""
    for line in lines:
        if len(buf) + len(line) > max_chars and buf:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    return chunks


@dataclass
class GraphRagTables:
    documents: pd.DataFrame
    text_units: pd.DataFrame
    entities: pd.DataFrame
    relationships: pd.DataFrame
    # reverse mapping used to write the index sidecar (entity id -> title)
    entity_title_by_id: dict[str, str] = field(default_factory=dict)
    text_unit_for_entity: dict[str, list[str]] = field(default_factory=dict)


def _unique_titles(entities: list[CodeEntity]) -> dict[str, str]:
    """Assign a globally unique title per entity id (deterministic).

    GraphRAG deduplicates entities by title, so same-named symbols in different
    files must get distinct titles. We disambiguate with the file path and a
    counter as a last resort.
    """
    used: set[str] = set()
    titles: dict[str, str] = {}
    for e in entities:
        base = e.qualified_name or e.name or e.id
        title = base
        n = 1
        while title in used:
            title = f"{base} ({e.file}) #{n}"
            n += 1
        used.add(title)
        titles[e.id] = title
    return titles


def build_graphrag_tables(
    graph: CodeKnowledgeGraph,
    files: dict[str, FileContext],
    max_text_unit_chars: int = 2400,
    include_config: bool = True,
    include_docs: bool = True,
) -> GraphRagTables:
    """Materialize GraphRAG input tables from the Code Knowledge Graph."""
    entities = list(graph.entities.values())
    titles = _unique_titles(entities)
    entity_title = {e.id: titles[e.id] for e in entities}

    # ---- documents (one per source/doc/config file) ----
    doc_rows: list[dict[str, Any]] = []
    file_doc_id: dict[str, str] = {}
    for rel in sorted(files):
        if rel == "__repo__":
            continue
        doc_id = f"doc::{rel}"
        file_doc_id[rel] = doc_id
        doc_rows.append(
            {"id": doc_id, "title": rel, "text": "", "creation_date": "", "raw_data": ""}
        )

    # ---- text units (semantic boundaries) ----
    unit_rows: list[dict[str, Any]] = []
    entity_unit_ids: dict[str, list[str]] = {}
    unit_counter = 0
    repo_unit_id: str | None = None

    def new_unit(document_id: str, header: str, body: str) -> str:
        nonlocal unit_counter
        unit_counter += 1
        text = header + body if body else header
        unit_id = f"tu::{document_id}::{unit_counter}"
        unit_rows.append(
            {
                "id": unit_id,
                "document_id": document_id,
                "text": text,
                "n_tokens": _est_tokens(text),
            }
        )
        return unit_id

    def link(entity_id: str, unit_id: str) -> None:
        entity_unit_ids.setdefault(entity_id, [])
        if unit_id not in entity_unit_ids[entity_id]:
            entity_unit_ids[entity_id].append(unit_id)

    # repository-level unit (so repo-scoped relations always have a text unit)
    repo_entities = [e for e in entities if e.file == "__repo__"]
    if repo_entities:
        doc_id = "doc::__repo__"
        file_doc_id["__repo__"] = doc_id
        doc_rows.append(
            {"id": doc_id, "title": graph.repo, "text": "", "creation_date": "", "raw_data": ""}
        )
        repo_ent = repo_entities[0]
        repo_body = (
            f"Repository: {graph.repo}. "
            f"{len([e for e in entities if e.type == 'file'])} files, "
            f"{len([e for e in entities if e.type in ('class',)])} classes, "
            f"{len([e for e in entities if e.type in ('function', 'method')])} functions/methods. "
            + repo_ent.description
        )
        header = f"[repository: {graph.repo}]\n"
        repo_unit_id = f"tu::{doc_id}::0"
        unit_rows.append(
            {
                "id": repo_unit_id,
                "document_id": doc_id,
                "text": header + repo_body,
                "n_tokens": _est_tokens(repo_body),
            }
        )
        for e in repo_entities:
            link(e.id, repo_unit_id)

    for e in entities:
        if e.file == "__repo__":
            continue
        ctx = files.get(e.file)
        if ctx is None:
            continue
        doc_id = file_doc_id.get(e.file)
        if doc_id is None:
            continue

        # Code symbols with a source span -> their own unit (sub-chunked if big).
        if e.type in (
            "function",
            "method",
            "class",
            "constant",
            "variable",
            "exception",
            "test",
            "cli",
            "command",
            "api",
        ):
            body = _symbol_source(ctx, e)
            if not body:
                body = e.description
            chunks = _chunk_text(body, max_text_unit_chars) or [""]
            for i, chunk in enumerate(chunks):
                header = _metadata_header(e) + (
                    f"[part {i + 1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
                )
                unit_id = new_unit(doc_id, header, chunk)
                link(e.id, unit_id)
            continue

        if e.type in ("file", "module"):
            if e.type == "module":
                # reuse the file entity's unit (the file's first unit)
                file_ent = graph.entity_by_file(e.file, "file")
                if file_ent is not None:
                    for uid in entity_unit_ids.get(file_ent.id, []):
                        link(e.id, uid)
                    if entity_unit_ids.get(e.id):
                        continue
            body = ctx.text[:max_text_unit_chars] if ctx.kind == "source" else ""
            unit_id = new_unit(doc_id, _metadata_header(e), body)
            link(e.id, unit_id)
            continue

        if e.type in ("parameter",):
            # attach to the file unit (parent symbol unit is linked via relations)
            continue

        if e.type == "documentation" and include_docs:
            body = (ctx.text or e.description)[:max_text_unit_chars]
            unit_id = new_unit(doc_id, _metadata_header(e), body)
            link(e.id, unit_id)
            continue

        if e.type == "configuration" and include_config:
            body = (ctx.text or e.description)[:max_text_unit_chars]
            unit_id = new_unit(doc_id, _metadata_header(e), body)
            link(e.id, unit_id)
            continue

        if e.type in ("directory", "package"):
            continue

        # fallback: give it a small unit so it is retrievable
        unit_id = new_unit(doc_id, _metadata_header(e), e.description)
        link(e.id, unit_id)

    # ---- entities table ----
    ent_rows: list[dict[str, Any]] = []
    for e in entities:
        tu_ids = entity_unit_ids.get(e.id, [])
        # ensure at least the file's first unit for linkage
        if not tu_ids:
            # find any unit in this document
            doc_id = file_doc_id.get(e.file)
            if doc_id:
                tu_ids = [u["id"] for u in unit_rows if u["document_id"] == doc_id][:1]
        ent_rows.append(
            {
                "title": entity_title[e.id],
                "type": e.type,
                "description": e.description,
                "text_unit_ids": tu_ids,
                "frequency": 1,
            }
        )

    # ---- relationships table (source/target = entity titles) ----
    rel_rows: list[dict[str, Any]] = []
    for rel in graph.relations.values():
        src_title = entity_title.get(rel.source_id)
        tgt_title = entity_title.get(rel.target_id)
        if src_title is None or tgt_title is None:
            continue
        tu_ids = list(entity_unit_ids.get(rel.source_id, []))
        if not tu_ids:
            src_ent = graph.entities.get(rel.source_id)
            if src_ent is not None:
                doc_id = file_doc_id.get(src_ent.file)
                if doc_id:
                    tu_ids = [u["id"] for u in unit_rows if u["document_id"] == doc_id][:1]
        if not tu_ids:
            tgt_ent = graph.entities.get(rel.target_id)
            if tgt_ent is not None:
                doc_id = file_doc_id.get(tgt_ent.file)
                if doc_id:
                    tu_ids = [u["id"] for u in unit_rows if u["document_id"] == doc_id][:1]
        if not tu_ids and repo_unit_id is not None:
            tu_ids = [repo_unit_id]
        if not tu_ids:
            continue  # skip relations with no resolvable context (should not happen)
        desc = rel.description or f"{src_title} {rel.type.lower()} {tgt_title}"
        # prefix the relation type so GraphRAG's LLM-less graph still encodes it
        rel_rows.append(
            {
                "source": src_title,
                "target": tgt_title,
                "weight": rel.weight,
                "description": f"{rel.type}: {desc}",
                "text_unit_ids": tu_ids,
            }
        )

    docs_df = pd.DataFrame(doc_rows, columns=["id", "title", "text", "creation_date", "raw_data"])
    units_df = pd.DataFrame(unit_rows, columns=["id", "document_id", "text", "n_tokens"])
    ent_df = pd.DataFrame(
        ent_rows, columns=["title", "type", "description", "text_unit_ids", "frequency"]
    )
    rel_df = pd.DataFrame(
        rel_rows, columns=["source", "target", "weight", "description", "text_unit_ids"]
    )

    if docs_df.empty:
        docs_df = pd.DataFrame(columns=["id", "title", "text", "creation_date", "raw_data"])
    if units_df.empty:
        units_df = pd.DataFrame(columns=["id", "document_id", "text", "n_tokens"])
    if ent_df.empty:
        ent_df = pd.DataFrame(
            columns=["title", "type", "description", "text_unit_ids", "frequency"]
        )
    if rel_df.empty:
        rel_df = pd.DataFrame(
            columns=["source", "target", "weight", "description", "text_unit_ids"]
        )

    return GraphRagTables(
        documents=docs_df,
        text_units=units_df,
        entities=ent_df,
        relationships=rel_df,
        entity_title_by_id=entity_title,
        text_unit_for_entity=entity_unit_ids,
    )
