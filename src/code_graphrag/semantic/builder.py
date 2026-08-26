"""Build the Code Knowledge Graph from per-file parse results.

Deterministic only: every entity and relation here comes from CST/AST facts
(spans, imports, calls, inheritance, decorators, docstrings) plus simple,
documented heuristics (CLI/API/test/config/doc linking). No LLM involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from code_graphrag.logging_setup import get_logger
from code_graphrag.parsing.common import RawStructure
from code_graphrag.semantic.model import CodeEntity, CodeKnowledgeGraph

logger = get_logger(__name__)

_ROUTE_RE = re.compile(r"[\"']([^\"']{1,120})[\"']")
_METHOD_NAMES = ("get", "post", "put", "delete", "patch", "head", "options")


@dataclass
class FileContext:
    """Everything the builder needs about one file (parsed structure + text)."""

    rel_path: str
    kind: str  # source | doc | config
    structure: RawStructure | None = None
    text: str = ""
    line_count: int = 0


@dataclass
class SymbolIndex:
    """Cross-file symbol resolution tables built before entity emission."""

    module_to_file: dict[str, str] = field(default_factory=dict)
    # qualified_name -> (file, span start, span end)
    definitions: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    # file -> list of (name, qualified_name) local definitions (innermost last)
    local_names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # class qualified name -> {method name: method qualified name}
    class_methods: dict[str, dict[str, str]] = field(default_factory=dict)
    # method name -> [class qualified names that define it] (for receiver-free resolution)
    method_names: dict[str, list[str]] = field(default_factory=dict)
    # file -> {local import name: (module, original name)}
    imports: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    # file -> set of local definition names (for scope)
    file_def_names: set[str] = field(default_factory=set)
    # module name -> file
    module_files: dict[str, str] = field(default_factory=dict)
    # qualified name -> entity type (function/method/class)
    definition_kinds: dict[str, str] = field(default_factory=dict)


def _module_name_from_rel(rel_path: str, language: str) -> str:
    name = rel_path
    base = name.rsplit("/", 1)[-1]
    for lang, sufs in (
        ("python", (".pyi", ".py")),
        ("javascript", (".cjs", ".mjs", ".jsx", ".js")),
        ("typescript", (".tsx", ".ts")),
        ("java", (".java",)),
        ("go", (".go",)),
        ("rust", (".rs",)),
        ("c", (".c", ".h")),
        ("cpp", (".hpp", ".hxx", ".cpp", ".cc")),
        ("csharp", (".cs",)),
    ):
        if language == lang:
            for suf in sufs:
                if base.endswith(suf):
                    base = base[: -len(suf)]
                    break
            break
    else:
        base = base.rsplit(".", 1)[0]
    parts = name.replace("\\", "/").split("/")
    parts = parts[:-1] + [base] if len(parts) > 1 else [base]
    parts = [p for p in parts if p and p != "__init__"]
    return ".".join(parts) if parts else base


def _resolve_module_file(
    module: str, index: SymbolIndex, from_file: str | None = None
) -> str | None:
    if module in index.module_files:
        return index.module_files[module]
    # try progressively shorter suffixes (a.b.c -> b.c -> c)
    parts = module.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in index.module_files:
            return index.module_files[suffix]
    return None


def _resolve_relative_import(module: str, from_rel_path: str, language: str) -> str | None:
    """Resolve a relative import string to a repo-relative module name."""
    if not module.startswith("."):
        return module.lstrip(".")
    level = len(module) - len(module.lstrip("."))
    rest = module.lstrip(".")
    src_dir = from_rel_path.rsplit("/", 1)[0] if "/" in from_rel_path else ""
    parts = src_dir.split("/") if src_dir else []
    for _ in range(level - 1):
        if parts:
            parts = parts[:-1]
    if rest:
        parts = parts + rest.split(".")
    elif language == "python":
        # from . import x handled elsewhere; "from . import" -> the package itself
        pass
    return ".".join(p for p in parts if p) if parts else None


class CodeGraphBuilder:
    def __init__(
        self, repo: str, files: dict[str, FileContext], repo_label: str | None = None
    ) -> None:
        self.repo = repo_label or repo
        self.files = files
        self.graph = CodeKnowledgeGraph(repo=self.repo)
        self.index = SymbolIndex()
        self._file_entities: dict[str, str] = {}
        self._module_entities: dict[str, str] = {}
        self._doc_entity_map: dict[str, list[str]] = {}
        self._config_entity_map: dict[str, list[str]] = {}
        self._docs_text: dict[str, str] = {}
        self._config_text: dict[str, str] = {}

    # -- pass 1: resolution index -------------------------------------------

    def _build_index(self) -> None:
        for rel, ctx in self.files.items():
            if ctx.kind != "source" or ctx.structure is None:
                continue
            st = ctx.structure
            lang = st.language
            module = _module_name_from_rel(rel, lang) if not st.module_name else st.module_name
            self.index.module_files[module] = rel
            # register alternate module spellings
            for alt in {rel, module, module.replace(".", "/")}:
                self.index.module_files.setdefault(alt, rel)
            base = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if "/" in rel:
                prefix = rel.rsplit("/", 1)[0]
                if prefix.endswith("__init__"):
                    self.index.module_files.setdefault(prefix.replace("\\", "/"), rel)
            if base and "/" not in rel:
                self.index.module_files.setdefault(base, rel)

            for imp in st.imports:
                mod = imp.module
                if mod.startswith("."):
                    resolved = _resolve_relative_import(mod, rel, lang)
                    if resolved:
                        mod = resolved
                else:
                    mod = mod.split(" ")[0]
                for sym in imp.symbols:
                    orig = imp.alias.get(sym, sym)
                    self.index.imports.setdefault(rel, {})[sym] = (mod, orig)
                if mod and "*" not in imp.symbols:
                    self.index.imports.setdefault(rel, {})[mod.rsplit(".", 1)[-1]] = (
                        mod,
                        mod.rsplit(".", 1)[-1],
                    )

            for d in st.definitions:
                self.index.definitions[d.qualified_name] = (rel, d.span.start_line, d.span.end_line)
                self.index.definition_kinds[d.qualified_name] = (
                    "class"
                    if d.kind == "class"
                    else ("method" if d.kind == "method" else "function")
                )
                self.index.local_names.setdefault(rel, []).append((d.name, d.qualified_name))
                if d.kind == "class":
                    self.index.class_methods.setdefault(d.qualified_name, {})
                self.index.method_names.setdefault(d.name, [])
                if d.kind == "method":
                    cls = d.qualified_name.rsplit(".", 1)[0] if "." in d.qualified_name else None
                    if cls:
                        self.index.class_methods.setdefault(cls, {})[d.name] = d.qualified_name
                        self.index.method_names[d.name].append(cls)

    # -- pass 2: entity + relation emission ---------------------------------

    def build(self) -> CodeKnowledgeGraph:
        self._build_index()
        self._add_repo_and_dirs()
        # Pre-create file + module entities for every source file so import
        # relations can consistently target module entities (not file entities)
        # regardless of which file is processed first.
        self._pre_create_containers()
        for _, ctx in sorted(self.files.items()):
            if ctx.kind == "source":
                self._build_source_file(ctx)
            elif ctx.kind == "doc" and self.graph.repo:
                self._build_doc_file(ctx)
            elif ctx.kind == "config":
                self._build_config_file(ctx)
        self._add_doc_links()
        self._add_config_links()
        logger.info(
            "Code graph built: %d entities, %d relations",
            len(self.graph.entities),
            len(self.graph.relations),
        )
        return self.graph

    def _pre_create_containers(self) -> None:
        """Create file + module entities for all source files up front."""
        for _, ctx in sorted(self.files.items()):
            if ctx.kind != "source" or ctx.structure is None:
                continue
            self._file_entity(ctx)
            module = ctx.structure.module_name or _module_name_from_rel(
                ctx.rel_path, ctx.structure.language
            )
            self._module_entity(ctx, module)

    def _add_repo_and_dirs(self) -> None:
        repo_entity = self.graph.entity(
            "repository",
            self.repo,
            self.repo,
            "__repo__",
            description=f"Source repository '{self.repo}' with {len(self.files)} indexed files.",
        )
        dirs: set[str] = set()
        for rel in self.files:
            parts = rel.replace("\\", "/").split("/")
            if len(parts) > 1:
                acc = ""
                for p in parts[:-1]:
                    acc = f"{acc}/{p}" if acc else p
                    dirs.add(acc)
        for d in sorted(dirs):
            entity = self.graph.entity(
                "package" if (d.endswith("__init__.py") or d == "") else "directory",
                d.rsplit("/", 1)[-1],
                d,
                d,
                description=f"Directory '{d}' in {self.repo}.",
            )
            self.graph.relate(repo_entity.id, entity.id, "CONTAINS", f"{self.repo} contains {d}")

    def _file_entity(self, ctx: FileContext) -> str:
        fid = self._file_entities.get(ctx.rel_path)
        if fid:
            return fid
        st = ctx.structure
        defs = [d.name for d in st.definitions[:12]] if st else []
        imports = (
            sorted({i.module for i in st.imports if i.module and not i.module.startswith(".")})[:12]
            if st
            else []
        )
        desc = f"File '{ctx.rel_path}' ({st.language if st else ctx.kind}, {ctx.line_count} lines)."
        if defs:
            desc += f" Defines: {', '.join(defs)}."
        if imports:
            desc += f" Imports: {', '.join(imports)}."
        entity = self.graph.entity(
            "file",
            ctx.rel_path,
            ctx.rel_path,
            ctx.rel_path,
            description=desc,
            language=st.language if st else None,
        )
        self._file_entities[ctx.rel_path] = entity.id
        return entity.id

    def _module_entity(self, ctx: FileContext, module: str) -> str:
        mid = self._module_entities.get(module)
        if mid:
            return mid
        st = ctx.structure
        n_defs = len(st.definitions) if st else 0
        entity = self.graph.entity(
            "module",
            module,
            module,
            ctx.rel_path,
            description=f"Module '{module}' (file {ctx.rel_path}), {n_defs} top-level definitions.",
            language=st.language if st else None,
        )
        self._module_entities[module] = entity.id
        return entity.id

    def _build_source_file(self, ctx: FileContext) -> None:
        st = ctx.structure
        if st is None:
            self._file_entity(ctx)
            return
        repo_ent = self.graph.entity("repository", self.repo, self.repo, "__repo__")
        file_id = self._file_entity(ctx)
        module = (
            _module_name_from_rel(ctx.rel_path, st.language)
            if not st.module_name
            else st.module_name
        )
        module_id = self._module_entity(ctx, module)
        self.graph.relate(repo_ent.id, file_id, "CONTAINS", f"{self.repo} contains {ctx.rel_path}")
        self.graph.relate(
            file_id, module_id, "DEFINES", f"{ctx.rel_path} implements module {module}"
        )
        self.graph.relate(
            module_id, file_id, "CONTAINS", f"module {module} lives in {ctx.rel_path}"
        )

        # definition entities
        self._emit_definitions(ctx, st, file_id, module_id)

        # imports
        for imp in st.imports:
            mod = imp.module
            if mod.startswith("."):
                resolved = _resolve_relative_import(mod, ctx.rel_path, st.language)
                mod = resolved or mod
            if not mod:
                continue
            target_file = _resolve_module_file(mod, self.index, ctx.rel_path)
            if target_file and target_file != ctx.rel_path:
                target_module = _module_name_from_rel(
                    target_file, self.index.module_files.get(target_file, "")
                )
                tmid = self._module_entities.get(target_module) or self._file_entity_entity_id(
                    target_file
                )
                self.graph.relate(module_id, tmid, "IMPORTS", f"module {module} imports {mod}")
                self.graph.relate(
                    file_id,
                    self._file_entity_id(target_file),
                    "IMPORTS",
                    f"{ctx.rel_path} imports {mod}",
                )

        # calls / raises / inheritance / params
        self._emit_calls(ctx, st, file_id)
        self._emit_inheritance(ctx, st)
        self._emit_raises(ctx, st)
        self._emit_constants(ctx, st, file_id)
        self._emit_cli(ctx, st, file_id, module_id)
        self._emit_apis(ctx, st, file_id)
        self._emit_tests(ctx, st, file_id)

    def _file_entity_id(self, rel: str) -> str:
        if rel in self._file_entities:
            return self._file_entities[rel]
        return self._file_entity(FileContext(rel_path=rel, kind="source"))

    # alias kept for readability above
    def _file_entity_entity_id(self, rel: str) -> str:
        return self._file_entity_id(rel)

    def _etype_for(self, qualified_name: str) -> str:
        return self.index.definition_kinds.get(qualified_name, "function")

    def _emit_definitions(
        self, ctx: FileContext, st: RawStructure, file_id: str, module_id: str
    ) -> None:
        for d in st.definitions:
            etype = (
                "class" if d.kind == "class" else ("method" if d.kind == "method" else "function")
            )
            desc = f"{d.kind.capitalize()} '{d.qualified_name}' in {ctx.rel_path} lines {d.span.start_line_1based}-{d.span.end_line_1based}."
            if d.docstring:
                desc += f" Docstring: {d.docstring[:240]}"
            else:
                desc += f" Signature: {d.signature[:200]}"
            entity = self.graph.entity(
                etype,
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
                description=desc,
                language=st.language,
                bases=d.annotations.get("bases", ""),
                returns=d.annotations.get("returns", ""),
                decorators="; ".join(d.decorators),
            )
            self.graph.relate(
                file_id, entity.id, "DEFINES", f"{ctx.rel_path} defines {d.qualified_name}"
            )
            self.graph.relate(
                module_id,
                entity.id,
                "DEFINES",
                f"module {st.module_name or _module_name_from_rel(ctx.rel_path, st.language)} defines {d.qualified_name}",
            )

            if d.kind == "class":
                self.graph.relate(
                    entity.id, module_id, "CONTAINS", f"{d.qualified_name} is part of module"
                )
            # parameters
            for p in d.params:
                pname = p.name.lstrip("*")
                pid = self.graph.entity(
                    "parameter",
                    pname,
                    f"{d.qualified_name}.{pname}",
                    ctx.rel_path,
                    start_line=d.span.start_line_1based,
                    description=f"Parameter '{pname}' of {d.qualified_name}"
                    + (f" (annotation: {p.annotation})" if p.annotation else "")
                    + (" with default" if p.has_default else "")
                    + ".",
                ).id
                self.graph.relate(
                    entity.id,
                    pid,
                    "ACCEPTS_PARAMETER",
                    f"{d.qualified_name} accepts parameter {pname}",
                )

    def _resolve_call_target(self, ctx: FileContext, st: RawStructure, call) -> CodeEntity | None:
        """Resolve a raw call to a known in-repo entity, or None (external)."""
        # 1) local definition by name (function/method/class)
        for name, qual in reversed(self.index.local_names.get(ctx.rel_path, [])):
            if name == call.callee and qual in self.index.definitions:
                f, s, e = self.index.definitions[qual]
                etype = self._etype_for(qual)
                return self.graph.entity(etype, name, qual, f, start_line=s, end_line=e)
        # 2) method call: look up by method name across known classes
        if call.is_method_call:
            candidates = self.index.method_names.get(call.callee, [])
            if len(candidates) == 1:
                cls = candidates[0]
                mqual = self.index.class_methods.get(cls, {}).get(call.callee)
                if mqual:
                    f, s, e = self.index.definitions[mqual]
                    return self.graph.entity(
                        "method", call.callee, mqual, f, start_line=s, end_line=e
                    )
            # receiver is a local class name -> call a method of it (first match)
            if call.receiver:
                recv = call.receiver.split(".")[-1]
                for name, qual in self.index.local_names.get(ctx.rel_path, []):
                    if name == recv and qual in self.index.definitions:
                        mqual = self.index.class_methods.get(qual, {}).get(call.callee)
                        if mqual:
                            f, s, e = self.index.definitions[mqual]
                            return self.graph.entity(
                                "method", call.callee, mqual, f, start_line=s, end_line=e
                            )
        # 3) via imports: symbol name -> module -> definition in that module file
        imp = self.index.imports.get(ctx.rel_path, {}).get(call.callee)
        if imp:
            mod, orig = imp
            target_file = _resolve_module_file(mod, self.index, ctx.rel_path)
            if target_file:
                for name, qual in self.index.local_names.get(target_file, []):
                    if name == orig or name == call.callee:
                        f, s, e = self.index.definitions[qual]
                        etype = self._etype_for(qual)
                        return self.graph.entity(etype, name, qual, f, start_line=s, end_line=e)
        # 4) module-level call (callee is an imported module name)
        if imp and call.callee == imp[1].rsplit(".", 1)[-1] and not call.is_method_call:
            target_file = _resolve_module_file(imp[0], self.index, ctx.rel_path)
            if target_file:
                return self.graph.entity("module", imp[0], imp[0], target_file)
        return None

    def _emit_calls(self, ctx: FileContext, st: RawStructure, file_id: str) -> None:
        # attribute each call to the innermost enclosing definition
        defs = sorted(st.definitions, key=lambda d: d.span.start_line)
        for call in st.calls:
            caller = self._enclosing_def(defs, call.line)
            caller_id = None
            caller_desc = ctx.rel_path
            if caller is not None:
                etype = self._etype_for(caller.qualified_name)
                caller_id = self.graph.entity(
                    etype,
                    caller.name,
                    caller.qualified_name,
                    ctx.rel_path,
                    start_line=caller.span.start_line_1based,
                    end_line=caller.span.end_line_1based,
                ).id
                caller_desc = caller.qualified_name
            target = self._resolve_call_target(ctx, st, call)
            if target is None:
                continue
            if caller_id is None:
                caller_id = file_id
            self.graph.relate(
                caller_id,
                target.id,
                "CALLS",
                f"{caller_desc} calls {target.qualified_name}",
                file=ctx.rel_path,
            )

    @staticmethod
    def _enclosing_def(defs, line_1based: int):
        best = None
        for d in defs:
            if d.span.start_line_1based <= line_1based <= d.span.end_line_1based and (
                best is None or d.span.start_line >= best.span.start_line
            ):
                best = d
        return best

    def _emit_inheritance(self, ctx: FileContext, st: RawStructure) -> None:
        for d in st.definitions:
            if d.kind != "class" or not d.base:
                continue
            child = self.graph.entity(
                "class",
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
            )
            base_text = d.base.strip()
            base_qual = None
            # local class
            for name, qual in self.index.local_names.get(ctx.rel_path, []):
                if name == base_text.rsplit(".", 1)[-1] and qual in self.index.definitions:
                    base_qual = qual
                    break
            if base_qual is None:
                # via imports
                imp = self.index.imports.get(ctx.rel_path, {}).get(base_text.rsplit(".", 1)[-1])
                if imp:
                    target_file = _resolve_module_file(imp[0], self.index, ctx.rel_path)
                    if target_file:
                        for name, qual in self.index.local_names.get(target_file, []):
                            if name == base_text.rsplit(".", 1)[-1]:
                                base_qual = qual
                                break
            if base_qual is None:
                # unresolved -> still record base text as an external class entity (type class, file __external__)
                base_qual = f"__external__::{base_text}"
            f, s, e = self.index.definitions.get(base_qual, ("__external__", 0, 0))
            base = self.graph.entity(
                "class", base_text.rsplit(".", 1)[-1], base_qual, f, start_line=s, end_line=e
            )
            self.graph.relate(
                child.id, base.id, "INHERITS", f"{d.qualified_name} inherits from {base_text}"
            )

    def _emit_raises(self, ctx: FileContext, st: RawStructure) -> None:
        defs = sorted(st.definitions, key=lambda d: d.span.start_line)
        for r in st.raises:
            d = self._enclosing_def(defs, r.line)
            if d is None:
                continue
            etype = self._etype_for(d.qualified_name)
            fn = self.graph.entity(
                etype,
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
            )
            exc = self.graph.entity(
                "exception", r.exception, f"{ctx.rel_path}::{r.exception}", "__external__"
            )
            self.graph.relate(
                fn.id, exc.id, "RAISES", f"{d.qualified_name} may raise {r.exception}"
            )

    def _emit_constants(self, ctx: FileContext, st: RawStructure, file_id: str) -> None:
        for a in st.assignments:
            etype = "constant" if a.is_constant else "variable"
            ent = self.graph.entity(
                etype,
                a.name,
                f"{st.module_name or ctx.rel_path}.{a.name}",
                ctx.rel_path,
                start_line=a.line,
                description=f"{etype.capitalize()} '{a.name}' in {ctx.rel_path}. Value: {a.value_preview[:120]}.",
            )
            self.graph.relate(file_id, ent.id, "DEFINES", f"{ctx.rel_path} defines {a.name}")

    def _emit_cli(self, ctx: FileContext, st: RawStructure, file_id: str, module_id: str) -> None:
        for d in st.definitions:
            if not d.is_cli_entry:
                continue
            etype = self._etype_for(d.qualified_name)
            fn = self.graph.entity(
                etype,
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
            )
            cli = self.graph.entity(
                "cli",
                d.name,
                f"{st.module_name}.{d.name}",
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                description=f"CLI entrypoint '{d.name}' in {ctx.rel_path}.",
            )
            self.graph.relate(module_id, cli.id, "DEFINES", f"module exposes CLI command {d.name}")
            self.graph.relate(
                cli.id, fn.id, "CALLS", f"CLI command {d.name} invokes {d.qualified_name}"
            )
            # sub-commands: other functions in same file decorated with *.command
            for d2 in st.definitions:
                if d2 is d:
                    continue
                if any("command" in dec for dec in d2.decorators):
                    fn2 = self.graph.entity(
                        "function" if d2.kind != "method" else "method",
                        d2.name,
                        d2.qualified_name,
                        ctx.rel_path,
                        start_line=d2.span.start_line_1based,
                        end_line=d2.span.end_line_1based,
                    )
                    cmd = self.graph.entity(
                        "command",
                        d2.name,
                        f"{d.name} {d2.name}",
                        ctx.rel_path,
                        start_line=d2.span.start_line_1based,
                        description=f"Subcommand '{d2.name}' of CLI '{d.name}'.",
                    )
                    self.graph.relate(
                        cli.id, cmd.id, "CONTAINS", f"CLI {d.name} has subcommand {d2.name}"
                    )
                    self.graph.relate(
                        cmd.id, fn2.id, "CALLS", f"subcommand {d2.name} invokes {d2.qualified_name}"
                    )

    def _emit_apis(self, ctx: FileContext, st: RawStructure, file_id: str) -> None:
        for d in st.definitions:
            if not d.is_api_endpoint:
                continue
            etype = self._etype_for(d.qualified_name)
            fn = self.graph.entity(
                etype,
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
            )
            method = "GET"
            route = "/"
            for dec in d.decorators:
                m = re.match(r"(\.\w+)?\.(\w+)\s*\((.*)$", dec.strip())
                if m:
                    verb = m.group(2).lower()
                    if verb in _METHOD_NAMES:
                        method = verb.upper()
                    rm = _ROUTE_RE.search(m.group(3) or "")
                    if rm:
                        route = rm.group(1)
            api_name = f"{method} {route}"
            api = self.graph.entity(
                "api",
                api_name,
                f"{api_name} ({d.qualified_name})",
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                description=f"HTTP API endpoint {api_name} implemented by {d.qualified_name} in {ctx.rel_path}.",
            )
            self.graph.relate(
                api.id,
                fn.id,
                "REFERENCES",
                f"endpoint {api_name} is implemented by {d.qualified_name}",
            )

    def _emit_tests(self, ctx: FileContext, st: RawStructure, file_id: str) -> None:
        is_test_file = "test" in ctx.rel_path.lower()
        if not is_test_file and not any(d.is_test for d in st.definitions):
            return
        base = ctx.rel_path.rsplit("/", 1)[-1]
        target_base = re.sub(r"^test[_-]?", "", base).rsplit(".", 1)[0]
        target_file = None
        for _, f in self.index.module_files.items():
            fb = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if fb == target_base and f != ctx.rel_path:
                target_file = f
                break
        if is_test_file:
            test_ent = self.graph.entity(
                "test",
                base,
                f"test:{ctx.rel_path}",
                ctx.rel_path,
                description=f"Test file {ctx.rel_path}.",
            )
            self.graph.relate(file_id, test_ent.id, "DEFINES", f"{ctx.rel_path} is a test file")
            if target_file:
                self.graph.relate(
                    test_ent.id,
                    self._file_entity_id(target_file),
                    "TESTS",
                    f"test {ctx.rel_path} tests {target_file}",
                )
        for d in st.definitions:
            if not d.is_test:
                continue
            etype = self._etype_for(d.qualified_name)
            fn = self.graph.entity(
                etype,
                d.name,
                d.qualified_name,
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                end_line=d.span.end_line_1based,
            )
            test_ent = self.graph.entity(
                "test",
                d.name,
                f"test:{d.qualified_name}",
                ctx.rel_path,
                start_line=d.span.start_line_1based,
                description=f"Test case {d.qualified_name} in {ctx.rel_path}.",
            )
            self.graph.relate(fn.id, test_ent.id, "CONTAINS", f"{d.qualified_name} is a test case")
            if target_file:
                self.graph.relate(
                    test_ent.id,
                    self._file_entity_id(target_file),
                    "TESTS",
                    f"test {d.qualified_name} tests {target_file}",
                )

    def _build_doc_file(self, ctx: FileContext) -> None:
        base = ctx.rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self._docs_text[base] = ctx.text
        doc = self.graph.entity(
            "documentation",
            base,
            base,
            ctx.rel_path,
            description=f"Documentation file {ctx.rel_path}. {ctx.text[:200]!r}",
        )
        repo_ent = self.graph.entity("repository", self.repo, self.repo, "__repo__")
        self.graph.relate(repo_ent.id, doc.id, "CONTAINS", f"{self.repo} contains {ctx.rel_path}")
        self._doc_entity_map.setdefault(base, []).append(doc.id)
        # headings -> sections
        for line_no, line in enumerate(ctx.text.splitlines(), start=1):
            m = re.match(r"^(#{1,4})\s+(.+)$", line)
            if m and len(m.group(2)) < 120:
                section = self.graph.entity(
                    "documentation",
                    m.group(2).strip(),
                    f"{base}#{m.group(2).strip()}",
                    ctx.rel_path,
                    start_line=line_no,
                    description=f"Section '{m.group(2).strip()}' of {ctx.rel_path}.",
                )
                self.graph.relate(
                    doc.id, section.id, "CONTAINS", f"{base} has section {m.group(2).strip()}"
                )

    def _build_config_file(self, ctx: FileContext) -> None:
        base = ctx.rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self._config_text[base] = ctx.text
        cfg = self.graph.entity(
            "configuration",
            ctx.rel_path,
            ctx.rel_path,
            ctx.rel_path,
            description=f"Configuration file {ctx.rel_path}. Keys: {self._config_keys(ctx.text)}",
        )
        repo_ent = self.graph.entity("repository", self.repo, self.repo, "__repo__")
        self.graph.relate(repo_ent.id, cfg.id, "CONTAINS", f"{self.repo} contains {ctx.rel_path}")
        self._config_entity_map.setdefault(base, []).append(cfg.id)

    @staticmethod
    def _config_keys(text: str) -> str:
        return ", ".join(CodeGraphBuilder._config_keys_list(text))

    @staticmethod
    def _config_keys_list(text: str) -> list[str]:
        keys: list[str] = []
        for line in text.splitlines()[:200]:
            m = re.match(r"^\s*[\"']?([A-Za-z_][A-Za-z0-9_.-]{0,60})[\"']?\s*[:=]", line)
            if m:
                keys.append(m.group(1))
        seen: list[str] = []
        for k in keys:
            if k not in seen:
                seen.append(k)
        return seen

    def _add_doc_links(self) -> None:
        for rel, ctx in self.files.items():
            if ctx.kind != "source":
                continue
            st = ctx.structure
            names = [d.name for d in (st.definitions if st else []) if len(d.name) > 2][:24]
            for base, doc_ids in self._doc_entity_map.items():
                text = self._docs_text.get(base, "")
                if any(n in text for n in ([rel] + names)):
                    fid = self._file_entity_id(rel)
                    for did in doc_ids:
                        self.graph.relate(
                            fid, did, "DOCUMENTED_BY", f"{rel} is documented in {base}"
                        )

    def _add_config_links(self) -> None:
        for rel, ctx in self.files.items():
            if ctx.kind != "source":
                continue
            for base, cfg_ids in self._config_entity_map.items():
                cfg_text = self._config_text.get(base, "")
                cfg_keys = [k for k in self._config_keys_list(cfg_text)]
                # link if the source references the config file name or one of its keys
                if base in ctx.text or any(k in ctx.text for k in cfg_keys):
                    fid = self._file_entity_id(rel)
                    for cid in cfg_ids:
                        self.graph.relate(
                            fid, cid, "CONFIGURED_BY", f"{rel} reads configuration {base}"
                        )
