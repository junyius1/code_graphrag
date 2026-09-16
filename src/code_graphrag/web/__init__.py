"""Code Graph Explorer: FastAPI backend over a built Code GraphRAG index.

Run:
    CODE_GRAPHRAG_INDEX_DIR=./output/open-swe-code-local \
        uvicorn code_graphrag.web.app:app --port 8100
"""

from code_graphrag.web.adapter import IndexAdapter, get_adapter
from code_graphrag.web.app import create_app

__all__ = ["IndexAdapter", "create_app", "get_adapter"]
