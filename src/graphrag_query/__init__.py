"""Code GraphRAG Query Agent: LangGraph agent over the built index.

`run_query_agent(index_dir, question, ...)` runs a tool-calling agent
(question analysis -> controlled GraphRAG/code-graph tools -> multi-round
retrieval -> evidence budgeting -> grounded final answer). `query_repl`
provides the interactive CLI loop.
"""

from graphrag_query.agent import AgentResult, run_query_agent
from graphrag_query.context import ContextManager, EvidenceItem
from graphrag_query.data_access import QueryData
from graphrag_query.repl import query_repl
from graphrag_query.trace import Tracer

__all__ = [
    "AgentResult",
    "ContextManager",
    "EvidenceItem",
    "QueryData",
    "Tracer",
    "query_repl",
    "run_query_agent",
]
