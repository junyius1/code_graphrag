"""Optional LLM enrichment of the Code Semantic Model (LangChain structured output).

The deterministic layer never asks the LLM for facts it can compute itself
(spans, imports, calls, inheritance). This step only adds *semantic* text:
short purpose summaries for key symbols, used to improve GraphRAG embeddings
and retrieval. It is off by default and a no-op in mock mode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from code_graphrag.config.models import IndexConfig
from code_graphrag.logging_setup import get_logger
from code_graphrag.semantic.model import CodeEntity, CodeKnowledgeGraph

logger = get_logger(__name__)


class SymbolSummary(BaseModel):
    """Semantic summary of a single code symbol."""

    purpose: str = Field(description="One or two sentences describing what this symbol does.")
    concept: str = Field(description="One short noun phrase for the primary concept it implements.")


async def enrich_graph(
    graph: CodeKnowledgeGraph,
    files: dict,
    config: IndexConfig,
    max_symbols: int | None = None,
) -> int:
    """Enrich up to ``max_symbols`` key entities with LLM summaries.

    Only class/function/method entities are enriched, in degree order. Failures
    on individual symbols are logged and skipped (never fatal).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from code_graphrag.llm.models import create_chat_model

    model = create_chat_model(config.llm)
    structured = model.with_structured_output(SymbolSummary)

    ranked = [
        e
        for e in graph.entities.values()
        if e.type in ("class", "function", "method") and e.file != "__repo__"
    ]
    ranked.sort(key=lambda e: (e.type != "class", e.qualified_name))
    ranked = ranked[: (max_symbols or config.enrichment.max_symbols_per_call * 2)]

    ctx = files
    enriched = 0
    for e in ranked:
        code = _symbol_source(ctx, e, config.enrichment.max_code_chars)
        if not code:
            continue
        try:
            result: SymbolSummary = await structured.ainvoke(
                [
                    SystemMessage(
                        content="You summarize code symbols for a code knowledge graph. "
                        "Be concrete and technical. Never invent behavior not present in the code."
                    ),
                    HumanMessage(
                        content=(
                            f"Symbol: {e.qualified_name}\nType: {e.type}\nFile: {e.file}\n\n"
                            f"```{e.language or ''}\n{code}\n```"
                        )
                    ),
                ]
            )
            e.description = f"{e.description}\nPurpose: {result.purpose} Concept: {result.concept}."
            e.attributes["concept"] = result.concept
            enriched += 1
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
            logger.warning("enrichment failed for %s: %s", e.qualified_name, exc)
    logger.info("Enriched %d/%d symbols with LLM summaries", enriched, len(ranked))
    return enriched


def _symbol_source(ctx: dict, entity: CodeEntity, max_chars: int) -> str:
    fctx = ctx.get(entity.file)
    if fctx is None or not getattr(fctx, "text", ""):
        return ""
    lines = fctx.text.splitlines()
    start = max(0, entity.start_line - 1)
    end = min(len(lines), entity.end_line)
    if end <= start:
        end = min(len(lines), start + 1)
    return "\n".join(lines[start:end])[:max_chars]
