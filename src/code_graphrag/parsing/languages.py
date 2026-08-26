"""Language registry: maps file extensions to tree-sitter language ids and kinds."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_language_pack as tslp
from tree_sitter import Language as TSLanguage

from code_graphrag.config.models import Language
from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LangSpec:
    language: Language
    ts_id: str  # tree-sitter-language-pack language id
    source_extensions: tuple[str, ...]
    header_extensions: tuple[str, ...] = ()


LANG_SPECS: tuple[LangSpec, ...] = (
    LangSpec(Language.PYTHON, "python", (".py", ".pyi")),
    LangSpec(Language.JAVASCRIPT, "javascript", (".js", ".jsx", ".mjs", ".cjs")),
    LangSpec(Language.TYPESCRIPT, "typescript", (".ts", ".tsx")),
    LangSpec(Language.JAVA, "java", (".java",)),
    LangSpec(Language.GO, "go", (".go",)),
    LangSpec(Language.RUST, "rust", (".rs",)),
    LangSpec(Language.C, "c", (".c",), (".h",)),
    LangSpec(Language.CPP, "cpp", (".cc", ".cpp", ".cxx"), (".hpp", ".hxx", ".hh")),
    LangSpec(Language.CSHARP, "c_sharp", (".cs",)),
)

_EXT_TO_SPEC: dict[str, LangSpec] = {}
for _spec in LANG_SPECS:
    for _ext in _spec.source_extensions + _spec.header_extensions:
        _EXT_TO_SPEC[_ext] = _spec


def language_for_path(path: str | Path) -> LangSpec | None:
    """Return the language spec for a file path, or None if unsupported."""
    ext = Path(path).suffix.lower()
    return _EXT_TO_SPEC.get(ext)


def supported_extensions() -> set[str]:
    return set(_EXT_TO_SPEC)


_parser_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def get_language(ts_id: str) -> TSLanguage:
    """Get (and cache) a compiled tree-sitter Language by pack id."""
    with _cache_lock:
        lang = _parser_cache.get(ts_id)
        if lang is None:
            lang = tslp.get_language(ts_id)
            _parser_cache[ts_id] = lang
        return lang  # type: ignore[no-any-return]


class ParserPool:
    """Small thread-local cache of tree-sitter Parser objects."""

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self, ts_id: str):
        cache: dict[str, object] = getattr(self._local, "parsers", None) or {}
        if ts_id not in cache:
            from tree_sitter import Parser

            cache[ts_id] = Parser(get_language(ts_id))
        self._local.parsers = cache
        return cache[ts_id]


def parse_tree(parser_pool: ParserPool, ts_id: str, source: bytes) -> object:
    """Parse source bytes into a tree-sitter Tree."""
    return parser_pool.get(ts_id).parse(source)
