"""Optional Deep Agents integration: an interactive code-exploration agent.

Builds a Deep Agents agent (``create_deep_agent``) whose tools query the
deterministic code knowledge graph *and* the GraphRAG index, plus the built-in
filesystem tools rooted at the source repository. Used for open-ended
"explore / reason over this codebase" sessions.
"""

from code_graphrag.agents.code_agent import (
    build_code_tools,
    create_code_agent,
    run_code_agent,
)

__all__ = ["build_code_tools", "create_code_agent", "run_code_agent"]
