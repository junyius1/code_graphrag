"""Language-agnostic extraction results produced by per-language extractors.

These are *raw* structural facts from the CST/AST. The semantic layer converts
them into a Code Semantic Model (entities + relationships).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    start_line: int  # 0-based (tree-sitter convention)
    start_col: int
    end_line: int
    end_col: int

    @property
    def start_line_1based(self) -> int:
        return self.start_line + 1

    @property
    def end_line_1based(self) -> int:
        return self.end_line + 1


@dataclass
class RawDefinition:
    """A top-level or nested symbol definition (function/class/method...)."""

    kind: str  # "function" | "class" | "method" | "interface" | "struct" | ...
    name: str
    qualified_name: str
    file: str
    span: Span
    params: list[RawParam] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    docstring: str | None = None
    base: str | None = None  # primary base/extends target (raw text)
    is_test: bool = False
    is_cli_entry: bool = False
    is_api_endpoint: bool = False
    is_exported: bool = False
    signature: str = ""
    body_preview: str = ""


@dataclass
class RawParam:
    name: str
    annotation: str | None = None
    has_default: bool = False


@dataclass
class RawImport:
    module: str  # dotted module path (or file path for relative)
    level: int  # 0 = absolute, 1+ = relative
    symbols: list[str] = field(default_factory=list)  # imported names
    alias: dict[str, str] = field(default_factory=dict)  # local -> original
    line: int = 0  # 1-based


@dataclass
class RawCall:
    callee: str  # raw callee text (e.g. "obj.method", "foo", "len")
    is_method_call: bool = False
    receiver: str | None = None  # text of the receiver expression
    args: list[str] = field(default_factory=list)
    line: int = 0  # 1-based


@dataclass
class RawInheritance:
    base: str  # raw base text


@dataclass
class RawAssignment:
    name: str
    value_preview: str
    is_class_attr: bool = False
    is_constant: bool = False
    line: int = 0


@dataclass
class RawRaise:
    exception: str
    line: int = 0


@dataclass
class RawStructure:
    """High-level structural summaries used for entity/relationship building."""

    file: str
    module_name: str
    language: str
    package: str | None = None
    definitions: list[RawDefinition] = field(default_factory=list)
    imports: list[RawImport] = field(default_factory=list)
    calls: list[RawCall] = field(default_factory=list)
    raises: list[RawRaise] = field(default_factory=list)
    assignments: list[RawAssignment] = field(default_factory=list)
    decorators_on: dict[str, list[str]] = field(default_factory=dict)  # qualified -> decorators
    parse_error: bool = False
    raw_error: str | None = None


@dataclass
class ParseFailure:
    file: str
    error: str
