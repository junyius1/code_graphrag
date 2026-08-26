"""Orchestrate parsing of discovered files into per-file RawStructure results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from code_graphrag.config.models import IndexConfig
from code_graphrag.discovery.discovery import DiscoveryResult
from code_graphrag.logging_setup import get_logger
from code_graphrag.parsing import languages
from code_graphrag.parsing.common import ParseFailure, RawStructure
from code_graphrag.parsing.extractors.generic import extract_generic
from code_graphrag.parsing.extractors.python import extract_python

logger = get_logger(__name__)


@dataclass
class ParseResult:
    structures: dict[str, RawStructure] = field(default_factory=dict)
    failures: list[ParseFailure] = field(default_factory=list)
    skipped_unsupported: list[str] = field(default_factory=list)
    text_cache: dict[str, str] = field(default_factory=dict)

    @property
    def ok_count(self) -> int:
        return len(self.structures)


def _read_text(path: Path, max_bytes: int) -> str:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", "replace")


def parse_files(discovery: DiscoveryResult, config: IndexConfig) -> ParseResult:
    """Parse every discovered source file into a RawStructure.

    Files without a registered language are recorded and skipped (not failed).
    A per-file parse error is captured (failure) but never aborts the run.
    """
    result = ParseResult()
    d = config.discovery

    for f in discovery.sources:
        text = _read_text(f.path, d.max_file_bytes)
        result.text_cache[f.rel_path] = text
        spec = languages.language_for_path(f.rel_path)
        if spec is None:
            result.skipped_unsupported.append(f.rel_path)
            continue
        try:
            if spec.language is languages.Language.PYTHON:
                structure = extract_python(text, f.rel_path)
            else:
                structure = extract_generic(text, f.rel_path, spec.language)
        except Exception as exc:  # noqa: BLE001 - isolate per-file errors
            logger.warning("parse failed for %s: %s", f.rel_path, exc)
            result.failures.append(ParseFailure(file=f.rel_path, error=str(exc)))
            continue
        result.structures[f.rel_path] = structure
        if structure.parse_error:
            logger.debug("parse warnings in %s: %s", f.rel_path, structure.raw_error)

    logger.info(
        "Parsed %d source files (%d failures, %d unsupported)",
        len(result.structures),
        len(result.failures),
        len(result.skipped_unsupported),
    )
    return result


def read_doc_text(discovery: DiscoveryResult, rel: str, max_bytes: int) -> str:
    for f in discovery.docs:
        if f.rel_path == rel:
            return _read_text(f.path, max_bytes)
    return ""


def read_config_text(discovery: DiscoveryResult, rel: str, max_bytes: int) -> str:
    for f in discovery.configs:
        if f.rel_path == rel:
            return _read_text(f.path, max_bytes)
    return ""
