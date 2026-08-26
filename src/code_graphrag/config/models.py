"""Configuration models for Code GraphRAG.

Everything is explicit and overridable from CLI / YAML. No hardcoded paths or keys.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Language(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    MOCK = "mock"


class LLMConfig(BaseModel):
    """LLM configuration for the semantic-enrichment and report stages.

    The provider-agnostic knobs map onto ``langchain`` partner classes at use time.
    ``provider=mock`` runs fully offline with canned responses (used by tests).
    """

    provider: LLMProvider = LLMProvider.OPENAI_COMPATIBLE
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None

    @field_validator("provider")
    @classmethod
    def _norm(cls, v: str | LLMProvider) -> LLMProvider:
        return LLMProvider(v)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration (maps onto GraphRAG ``ModelConfig``)."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None


class DiscoveryConfig(BaseModel):
    """File discovery / filtering configuration."""

    source: str = "."
    include_globs: list[str] = Field(
        default_factory=lambda: [
            "*.py",
            "*.js",
            "*.jsx",
            "*.ts",
            "*.tsx",
            "*.mjs",
            "*.cjs",
            "*.java",
            "*.go",
            "*.rs",
            "*.c",
            "*.cc",
            "*.cpp",
            "*.h",
            "*.hpp",
            "*.hxx",
            "*.cs",
            "*.scala",
            "*.kt",
            "*.rb",
            "*.php",
        ]
    )
    doc_globs: list[str] = Field(default_factory=lambda: ["*.md", "*.mdx", "*.rst", "*.txt"])
    config_globs: list[str] = Field(
        default_factory=lambda: ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"]
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "coverage",
            "egg-info",
            ".tox",
            ".eggs",
            ".cache",
        ]
    )
    max_file_bytes: int = 1_000_000
    extra_exclude: list[str] = Field(default_factory=list)


class EnrichmentConfig(BaseModel):
    """LLM semantic-enrichment toggles and batching."""

    enabled: bool = False
    max_symbols_per_call: int = 24
    batch_size: int = 16
    max_code_chars: int = 4000
    max_symbol_chars: int = 400


class GraphRAGAdapterConfig(BaseModel):
    """How the Code Knowledge Graph maps onto the GraphRAG pipeline."""

    max_text_unit_chars: int = 2400
    # Controlled vocabulary of entity types for the deterministic graph.
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "file",
            "module",
            "package",
            "class",
            "function",
            "method",
            "parameter",
            "variable",
            "constant",
            "api",
            "cli",
            "command",
            "exception",
            "test",
            "documentation",
            "configuration",
            "concept",
        ]
    )
    include_config_entities: bool = True
    include_doc_entities: bool = True
    relationship_types: list[str] = Field(
        default_factory=lambda: [
            "CONTAINS",
            "DEFINES",
            "IMPORTS",
            "CALLS",
            "INHERITS",
            "IMPLEMENTS",
            "USES",
            "RETURNS",
            "ACCEPTS_PARAMETER",
            "RAISES",
            "TESTS",
            "DOCUMENTED_BY",
            "CONFIGURED_BY",
            "REFERENCES",
            "RELATED_TO",
        ]
    )


class ClusterConfig(BaseModel):
    """GraphRAG community-detection knobs (mirrors graphrag ClusterGraphConfig)."""

    max_cluster_size: int = 25
    # NOTE: use_lcc must stay False for code graphs. GraphRAG's stable_lcc
    # uppercases node names, which breaks the case-sensitive title -> entity
    # merge in create_communities. See graphrag/graphs/stable_lcc.py.
    use_lcc: bool = False
    seed: int = 0xDEADBEEF


class IndexConfig(BaseModel):
    """Top-level configuration for a Code GraphRAG index build."""

    name: str = "code-graphrag-index"
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    adapter: GraphRAGAdapterConfig = Field(default_factory=GraphRAGAdapterConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    # Vector store location relative to the output dir (lancedb).
    vector_db_subdir: str = "lancedb"
    output_dir: str | None = None
    verbose: bool = False

    @property
    def languages(self) -> list[Language]:
        """Languages enabled, derived from which source globs are present."""
        enabled: set[Language] = set()
        g = set(self.discovery.include_globs)
        mapping = {
            Language.PYTHON: ("*.py",),
            Language.JAVASCRIPT: ("*.js", "*.jsx", "*.mjs", "*.cjs"),
            Language.TYPESCRIPT: ("*.ts", "*.tsx"),
            Language.JAVA: ("*.java",),
            Language.GO: ("*.go",),
            Language.RUST: ("*.rs",),
            Language.C: ("*.c", "*.h"),
            Language.CPP: ("*.cc", "*.cpp", "*.hpp", "*.hxx"),
            Language.CSHARP: ("*.cs",),
        }
        for lang, exts in mapping.items():
            if any(e in g for e in exts):
                enabled.add(lang)
        return sorted(enabled, key=lambda x: x.value)


def default_index_config() -> IndexConfig:
    return IndexConfig()
