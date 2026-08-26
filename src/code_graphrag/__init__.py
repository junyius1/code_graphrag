"""Code GraphRAG: deterministic code knowledge graphs + Microsoft GraphRAG."""

import os

# Local-first: Code GraphRAG works fully offline (mock LLM or local models), so
# remote LangSmith tracing is off by default. Set CODE_GRAPHRAG_TRACING=1 to
# enable it (requires LANGSMITH_API_KEY).
if os.environ.get("CODE_GRAPHRAG_TRACING", "").lower() in ("1", "true", "yes"):
    pass
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)
