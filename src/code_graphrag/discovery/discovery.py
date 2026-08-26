"""Recursive, configurable file discovery for source repositories."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from code_graphrag.config.models import DiscoveryConfig, IndexConfig
from code_graphrag.logging_setup import get_logger

logger = get_logger(__name__)

FileKind = Literal["source", "doc", "config", "other"]


@dataclass
class DiscoveredFile:
    """A single file selected by discovery, with its relative path and kind."""

    path: Path
    rel_path: str
    kind: FileKind
    size: int = 0


@dataclass
class DiscoveryResult:
    root: Path
    files: list[DiscoveredFile] = field(default_factory=list)
    skipped_binary: list[str] = field(default_factory=list)
    skipped_too_large: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[DiscoveredFile]:
        return [f for f in self.files if f.kind == "source"]

    @property
    def docs(self) -> list[DiscoveredFile]:
        return [f for f in self.files if f.kind == "doc"]

    @property
    def configs(self) -> list[DiscoveredFile]:
        return [f for f in self.files if f.kind == "config"]

    def by_kind(self, kind: FileKind) -> list[DiscoveredFile]:
        return [f for f in self.files if f.kind == kind]


def _norm(patterns: list[str]) -> list[str]:
    return [p for p in patterns if p]


def _is_excluded(rel: Path, exclude_parts: list[str]) -> bool:
    """Return True if any path component matches an exclude pattern."""
    for part in rel.parts:
        for pat in exclude_parts:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _kind_for(rel: Path, d: DiscoveryConfig) -> FileKind | None:
    name = rel.name
    if any(fnmatch.fnmatch(name, p) for p in _norm(d.doc_globs)):
        return "doc"
    if any(fnmatch.fnmatch(name, p) for p in _norm(d.config_globs)):
        return "config"
    if any(fnmatch.fnmatch(name, p) for p in _norm(d.include_globs)):
        return "source"
    return None


def discover_files(root: str | Path, config: IndexConfig) -> DiscoveryResult:
    """Recursively discover files under ``root`` per the discovery config.

    - Skips excluded directories (by name match on any path component).
    - Skips unreadable / binary / oversized files (recorded, not fatal).
    - Classifies each kept file as source / doc / config.
    """
    root = Path(root).resolve()
    d = config.discovery
    exclude_parts = _norm(d.exclude_globs + d.extra_exclude)
    result = DiscoveryResult(root=root)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_excluded(rel, exclude_parts):
            continue
        kind = _kind_for(rel, d)
        if kind is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            result.skipped_unreadable.append(str(rel))
            continue
        if size > d.max_file_bytes:
            result.skipped_too_large.append(str(rel))
            continue
        try:
            head = path.open("rb").read(8192)
        except OSError:
            result.skipped_unreadable.append(str(rel))
            continue
        if head and _is_binary(head):
            result.skipped_binary.append(str(rel))
            continue
        result.files.append(DiscoveredFile(path=path, rel_path=str(rel), kind=kind, size=size))

    logger.info(
        "Discovery: %d files (%d source, %d doc, %d config); %d binary, %d large, %d unreadable skipped",
        len(result.files),
        len(result.sources),
        len(result.docs),
        len(result.configs),
        len(result.skipped_binary),
        len(result.skipped_too_large),
        len(result.skipped_unreadable),
    )
    return result
