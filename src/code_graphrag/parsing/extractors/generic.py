"""Generic tree-sitter extractor for JS/TS/Java/Go/Rust/C/C++/C#.

One walker, per-language node-type tables. Produces the same RawStructure
shape as the Python extractor so the semantic layer is language-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from code_graphrag.config.models import Language
from code_graphrag.logging_setup import get_logger
from code_graphrag.parsing import languages
from code_graphrag.parsing.common import (
    RawCall,
    RawDefinition,
    RawImport,
    RawParam,
    RawRaise,
    RawStructure,
    Span,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class LangNodes:
    class_types: tuple[str, ...]
    fn_types: tuple[str, ...]
    import_node_types: tuple[str, ...]
    call_node_types: tuple[str, ...]
    raise_node_types: tuple[str, ...]
    annotation_types: tuple[str, ...]  # decorators/attributes/annotations
    module_node_types: tuple[str, ...]  # package/module declaration


TABLES: dict[Language, LangNodes] = {
    Language.JAVASCRIPT: LangNodes(
        class_types=("class_declaration", "interface_declaration", "abstract_class"),
        fn_types=("function_declaration", "function_expression", "generator_function"),
        import_node_types=("import_statement", "import_declaration"),
        call_node_types=("call_expression", "new_expression"),
        raise_node_types=("throw_statement",),
        annotation_types=(),
        module_node_types=(),
    ),
    Language.TYPESCRIPT: LangNodes(
        class_types=("class_declaration", "abstract_class"),
        fn_types=(
            "function_declaration",
            "function_expression",
            "method_definition",
            "generator_function",
            "generator_method",
        ),
        import_node_types=("import_statement", "import_declaration"),
        call_node_types=("call_expression", "new_expression"),
        raise_node_types=("throw_statement",),
        annotation_types=(),
        module_node_types=("declaration",),
    ),
    Language.JAVA: LangNodes(
        class_types=(
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        ),
        fn_types=("method_declaration", "constructor_declaration", "field_declaration"),
        import_node_types=("import_declaration",),
        call_node_types=("method_invocation", "object_creation_expression"),
        raise_node_types=("throw_statement",),
        annotation_types=("marker_annotation", "annotation"),
        module_node_types=("package_declaration",),
    ),
    Language.GO: LangNodes(
        class_types=(),
        fn_types=("function_declaration", "method_declaration"),
        import_node_types=("import_spec",),
        call_node_types=("call_expression",),
        raise_node_types=(),
        annotation_types=(),
        module_node_types=("package_clause",),
    ),
    Language.RUST: LangNodes(
        class_types=(),
        fn_types=("function_item",),
        import_node_types=("use_declaration",),
        call_node_types=("call_expression", "method_call_expression"),
        raise_node_types=("macro_invocation",),
        annotation_types=("attribute_item",),
        module_node_types=(),
    ),
    Language.C: LangNodes(
        class_types=("struct_specifier",),
        fn_types=("function_definition",),
        import_node_types=("preproc_def",),
        call_node_types=("call_expression",),
        raise_node_types=(),
        annotation_types=(),
        module_node_types=(),
    ),
    Language.CPP: LangNodes(
        class_types=("class_specifier", "struct_specifier"),
        fn_types=("function_definition",),
        import_node_types=("preproc_def",),
        call_node_types=("call_expression",),
        raise_node_types=("try_statement", "throw_expression"),
        annotation_types=(),
        module_node_types=(),
    ),
    Language.CSHARP: LangNodes(
        class_types=(
            "class_declaration",
            "interface_declaration",
            "record_declaration",
            "struct_declaration",
        ),
        fn_types=("function_declaration", "method_declaration", "constructor_declaration"),
        import_node_types=("using_directive",),
        call_node_types=("invocation_expression", "object_creation_expression"),
        raise_node_types=("throw_statement",),
        annotation_types=("attribute_list",),
        module_node_types=("namespace_declaration",),
    ),
}

_JS_ROUTE_RE = re.compile(r"^\s*(get|post|put|delete|patch|all)\s*\(")
_API_HINT_RE = re.compile(
    r"(route|api_view|get|post|put|delete|requestmapping|controller|router|endpoint|handler|view|action)",
    re.I,
)
_CLI_HINT_RE = re.compile(
    r"(command|argparse|click|typer|subparser|arg_?parse|console|cli|main)", re.I
)
_TEST_HINT_RE = re.compile(r"(\btest\b|_test|Test|spec|fixture|mock)", re.I)
_EXPORT_RE = re.compile(r"^\s*export\b")


def _text(node) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", "replace")


def _span_of(node) -> Span:
    return Span(node.start_point[0], node.start_point[1], node.end_point[0], node.end_point[1])


def _field(node, name: str):
    try:
        return node.child_by_field_name(name)
    except Exception:  # noqa: BLE001 - older bindings may raise
        return None


def _name_node(node) -> str:
    n = _field(node, "name")
    if n is None:
        for child in node.named_children:
            if child.type in (
                "identifier",
                "type_identifier",
                "name",
                "simple_identifier",
                "name_identifier",
            ):
                return _text(child)
    return _text(n)


def _params_of(node, lang: Language) -> list[RawParam]:
    params: list[RawParam] = []
    pend = _field(node, "parameters") or _field(node, "formal_parameters")
    if pend is None:
        for child in node.named_children:
            if child.type in ("formal_parameters", "parameters", "parameter_list", "argument_list"):
                pend = child
                break
    if pend is None:
        return params
    for child in pend.named_children:
        if child.type in ("identifier", "simple_identifier", "identifier_pattern", "pattern"):
            params.append(RawParam(name=_text(child)))
        elif child.type in (
            "required_parameter",
            "optional_parameter",
            "rest_parameter",
            "assignment_pattern",
            "typed_parameter",
        ):
            n = _field(child, "name") or _field(child, "pattern")
            t = _field(child, "type")
            params.append(
                RawParam(
                    name=_text(n) if n else _text(child).split("=")[0].strip() or "?",
                    annotation=_text(t) if t else None,
                    has_default=child.type in ("assignment_pattern", "optional_parameter"),
                )
            )
    return params


def _bases_of(node, lang: Language) -> list[str]:
    bases: list[str] = []
    for field in ("superclasses", "superclass", "super_type", "bases", "super_interfaces"):
        b = _field(node, field)
        if b is not None:
            for child in b.named_children:
                if _text(child):
                    bases.append(_text(child))
            if not bases and _text(b):
                bases.append(_text(b).strip())
    # java extends/implements via keywords
    for child in node.children:
        if child.type == "extends" or _text(child).startswith("extends "):
            for sib in node.named_children:
                if sib.start_byte > child.end_byte:
                    bases.append(_text(sib))
                    break
        if child.type == "implements" or _text(child).startswith("implements "):
            for sib in node.named_children:
                if sib.start_byte > child.end_byte:
                    bases.append(_text(sib).split(",")[0])
                    break
    return bases


def _annotations_of(node, types: tuple[str, ...]) -> list[str]:
    anns: list[str] = []
    for child in node.children:
        if child.type in types:
            anns.append(_text(child).strip())
    return anns


def _line(source: str, line0: int) -> str:
    try:
        return source.splitlines()[line0].strip()
    except IndexError:
        return ""


def _is_test_symbol(name: str, rel_path: str, annotations: list[str]) -> bool:
    return (
        _TEST_HINT_RE.search(name) is not None
        or "test" in rel_path.lower()
        or any(_TEST_HINT_RE.search(a) for a in annotations)
    )


def _module_name_for(lang: Language, rel_path: str, tree_root) -> str:
    if lang is Language.JAVA:
        for node in tree_root.named_children:
            if node.type == "package_declaration":
                scoped = node.child_by_field_name("name")
                return (
                    _text(scoped)
                    if scoped
                    else _text(node).removeprefix("package").removesuffix(";").strip()
                )
    if lang is Language.GO:
        for node in tree_root.named_children:
            if node.type == "package_clause":
                ident = node.child_by_field_name("name")
                return _text(ident) if ident else _text(node).removeprefix("package").strip()
    if lang is Language.CSHARP:
        for node in tree_root.named_children:
            if node.type == "namespace_declaration":
                name = _field(node, "name")
                return _text(name) if name else ""
    name = rel_path
    base = name.rsplit("/", 1)[-1]
    for suf in (
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".mjs",
        ".cjs",
        ".hpp",
        ".hxx",
    ):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = name.replace("\\", "/").split("/")
    parts[-1] = base
    return ".".join(p for p in parts if p)


# ---- imports per language --------------------------------------------------


def _js_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in root.named_children:
        if node.type not in ("import_statement", "import_declaration"):
            continue
        src = _field(node, "source")
        if src is not None:
            module = _text(src).strip("'\"")
            if module.startswith("."):
                level = len(module) - len(module.lstrip("."))
                imports.append(
                    RawImport(module=module, level=level, symbols=[], line=node.start_point[0] + 1)
                )
                continue
            symbols: list[str] = []
            alias: dict[str, str] = []
            clause = _field(node, "clause")
            if clause is not None:
                for child in clause.named_children:
                    if child.type == "identifier":
                        symbols.append(_text(child))
                    elif child.type == "namespace_import":
                        local = _field(child, "local_name") or _field(child, "name")
                        if local is not None:
                            symbols.append(_text(local))
                    elif child.type == "named_imports":
                        for spec in child.named_children:
                            local = _field(spec, "local_name")
                            orig = _field(spec, "name")
                            local_text = _text(local) if local else "?"
                            o = _text(orig) if orig else local_text
                            symbols.append(local_text)
                            if local_text != o:
                                alias.append(f"{local_text}={o}")
            imports.append(
                RawImport(
                    module=module,
                    level=0,
                    symbols=symbols,
                    alias=dict(zip(symbols, [a.split("=")[1] for a in alias], strict=False))
                    if alias
                    else {},
                    line=node.start_point[0] + 1,
                )
            )
    return imports


def _java_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in root.named_children:
        if node.type != "import_declaration":
            continue
        name = _field(node, "name")
        module = (
            _text(name) if name else _text(node).removeprefix("import").removesuffix(";").strip()
        )
        star = module.endswith(".*")
        base = module.removesuffix(".*")
        imports.append(
            RawImport(
                module=base,
                level=0,
                symbols=["*"] if star else [base.rsplit(".", 1)[-1]],
                line=node.start_point[0] + 1,
            )
        )
    return imports


def _go_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        for child in node.named_children:
            if child.type == "import_spec" and id(child) not in seen:
                seen.add(id(child))
                path = _field(child, "path")
                module = _text(path).strip('"') if path is not None else _text(child).strip('"')
                name = _field(child, "name")
                alias = {}
                if (
                    name is not None
                    and not _text(name).startswith("_")
                    and not _text(name).startswith(".")
                ):
                    alias = {_text(name): module.rsplit("/", 1)[-1]}
                imports.append(
                    RawImport(
                        module=module,
                        level=0,
                        symbols=[],
                        alias=alias,
                        line=child.start_point[0] + 1,
                    )
                )
            stack.append(child)
    return imports


def _rust_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "use_declaration":
            imports.append(
                RawImport(
                    module=_text(node).removeprefix("use").removesuffix(";").strip(),
                    level=0,
                    symbols=[],
                    line=node.start_point[0] + 1,
                )
            )
        stack.extend(node.named_children)
    return imports


def _c_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "preproc_def":
            name = _field(node, "name")
            arg = _field(node, "argument")
            if name is not None and _text(name) == "include" and arg is not None:
                module = _text(arg).strip().strip("<>").strip('"')
                imports.append(
                    RawImport(module=module, level=0, symbols=[], line=node.start_point[0] + 1)
                )
        stack.extend(node.named_children)
    return imports


def _cs_imports(root, rel_path: str) -> list[RawImport]:
    imports: list[RawImport] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "using_directive":
            ns = _field(node, "name") or _field(node, "namespace")
            imports.append(
                RawImport(
                    module=_text(ns) if ns else _text(node),
                    level=0,
                    symbols=[],
                    line=node.start_point[0] + 1,
                )
            )
        stack.extend(node.named_children)
    return imports


IMPORT_HANDLERS = {
    Language.JAVASCRIPT: _js_imports,
    Language.TYPESCRIPT: _js_imports,
    Language.JAVA: _java_imports,
    Language.GO: _go_imports,
    Language.RUST: _rust_imports,
    Language.C: _c_imports,
    Language.CPP: _c_imports,
    Language.CSHARP: _cs_imports,
}


# ---- calls per language ----------------------------------------------------


def _calls_of_node(node, lang: Language, out: list[RawCall]) -> None:
    if lang in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        if node.type in ("call_expression", "new_expression"):
            f = _field(node, "function") or _field(node, "constructor")
            if f is not None:
                if f.type == "member_expression":
                    obj = _field(f, "object")
                    prop = _field(f, "property")
                    out.append(
                        RawCall(
                            callee=_text(prop),
                            is_method_call=True,
                            receiver=_text(obj) if obj else None,
                            line=node.start_point[0] + 1,
                        )
                    )
                elif f.type == "identifier":
                    out.append(
                        RawCall(callee=_text(f), is_method_call=False, line=node.start_point[0] + 1)
                    )
    elif lang is Language.JAVA:
        if node.type == "method_invocation":
            obj = _field(node, "object")
            nm = _field(node, "name")
            out.append(
                RawCall(
                    callee=_text(nm),
                    is_method_call=obj is not None,
                    receiver=_text(obj) if obj else None,
                    line=node.start_point[0] + 1,
                )
            )
        elif node.type == "object_creation_expression":
            t = _field(node, "type")
            if t is not None:
                out.append(
                    RawCall(callee=_text(t), is_method_call=False, line=node.start_point[0] + 1)
                )
    elif lang is Language.GO:
        if node.type == "call_expression":
            f = _field(node, "function")
            if f is not None:
                if f.type == "selector_expression":
                    out.append(
                        RawCall(
                            callee=_text(_field(f, "field")),
                            is_method_call=True,
                            receiver=_text(_field(f, "operand")),
                            line=node.start_point[0] + 1,
                        )
                    )
                else:
                    out.append(
                        RawCall(callee=_text(f), is_method_call=False, line=node.start_point[0] + 1)
                    )
    elif lang is Language.RUST:
        if node.type == "call_expression":
            f = _field(node, "function")
            if f is not None and f.type in ("identifier", "scoped_identifier"):
                out.append(
                    RawCall(
                        callee=_text(f).split("::")[-1],
                        is_method_call=False,
                        line=node.start_point[0] + 1,
                    )
                )
        elif node.type == "method_call_expression":
            out.append(
                RawCall(
                    callee=_text(_field(node, "method")),
                    is_method_call=True,
                    receiver=_text(_field(node, "object")),
                    line=node.start_point[0] + 1,
                )
            )
    elif lang in (Language.C, Language.CPP):
        f = _field(node, "function") if node.type == "call_expression" else None
        if f is not None and f.type == "field_expression":
            out.append(
                RawCall(
                    callee=_text(_field(f, "field")),
                    is_method_call=True,
                    receiver=_text(_field(f, "argument")),
                    line=node.start_point[0] + 1,
                )
            )
        elif f is not None:
            out.append(
                RawCall(
                    callee=_text(f).split("(")[0].strip(),
                    is_method_call=False,
                    line=node.start_point[0] + 1,
                )
            )
    elif lang is Language.CSHARP:
        f = (
            (_field(node, "object") or _field(node, "function"))
            if node.type == "invocation_expression"
            else None
        )
        if f is not None and f.type in ("member_access_expression", "generic_name"):
            out.append(
                RawCall(
                    callee=_text(_field(f, "name") or _field(f, "member")),
                    is_method_call=True,
                    receiver=_text(_field(f, "expression")),
                    line=node.start_point[0] + 1,
                )
            )
        elif f is not None:
            out.append(
                RawCall(
                    callee=_text(f).split("(")[0].strip(),
                    is_method_call=False,
                    line=node.start_point[0] + 1,
                )
            )


def _raise_of_node(node, lang: Language) -> RawRaise | None:
    if node.type not in (
        "throw_statement",
        "try_statement",
        "throw_expression",
        "macro_invocation",
    ):
        return None
    exc = ""
    if node.type == "throw_statement":
        arg = _field(node, "argument") or (node.named_children[0] if node.named_children else None)
        exc = _text(arg) if arg is not None else ""
    elif node.type == "throw_expression":
        v = _field(node, "value") or (node.named_children[0] if node.named_children else None)
        exc = _text(v) if v is not None else ""
    elif node.type == "macro_invocation":
        exc = _text(node)
        if not exc.startswith("panic"):
            return None
        exc = "panic"
    elif node.type == "try_statement":
        return None
    name = exc.split("(")[0].strip().split(".")[-1]
    if not name:
        return None
    return RawRaise(exception=name, line=node.start_point[0] + 1)


def _extract(
    node,
    lang: Language,
    spec,
    source: str,
    rel_path: str,
    qual_prefix: str,
    in_class: bool | None,
    out: list[RawDefinition],
) -> None:
    t = TABLES[lang]
    if node.type in t.class_types:
        name = _name_node(node)
        if not name:
            return
        qual = f"{qual_prefix}{name}"
        bases = _bases_of(node, lang)
        anns = _annotations_of(node, t.annotation_types)
        out.append(
            RawDefinition(
                kind="class",
                name=name,
                qualified_name=qual,
                file=rel_path,
                span=_span_of(node),
                decorators=anns,
                base=bases[0] if bases else None,
                annotations={"bases": ", ".join(bases)} if bases else {},
                is_test=_is_test_symbol(name, rel_path, anns),
                is_api_endpoint=any(_API_HINT_RE.search(a) for a in anns),
                signature=_line(source, node.start_point[0]),
            )
        )
        # recurse into body for methods
        body = _field(node, "body") or _field(node, "members")
        if body is not None:
            _walk_children(body, lang, spec, source, rel_path, f"{qual}.", True, out)
    elif node.type in t.fn_types:
        name = _name_node(node)
        if not name or name in ("catch", "default", "return"):
            return
        qual = f"{qual_prefix}{name}"
        anns = _annotations_of(node, t.annotation_types)
        is_method = in_class is True
        if lang in (Language.JAVASCRIPT, Language.TYPESCRIPT) and node.type in (
            "function_expression",
        ):
            is_method = False
        if lang is Language.TYPESCRIPT and node.type in ("method_definition", "generator_method"):
            is_method = in_class is True or True
        sig = _line(source, node.start_point[0])
        params = _params_of(node, lang)
        out.append(
            RawDefinition(
                kind="method" if is_method else "function",
                name=name,
                qualified_name=qual,
                file=rel_path,
                span=_span_of(node),
                params=params,
                decorators=anns,
                is_test=_is_test_symbol(name, rel_path, anns)
                or (is_method and name.lower() in ("test", "main", "run") and lang is Language.GO),
                is_cli_entry=_CLI_HINT_RE.search(sig) is not None
                or any(_CLI_HINT_RE.search(a) for a in anns),
                is_api_endpoint=any(_API_HINT_RE.search(a) for a in anns),
                signature=sig,
            )
        )
        body = _field(node, "body")
        if body is not None:
            _walk_children(body, lang, spec, source, rel_path, f"{qual}.", None, out)
    elif node.type in (
        "program",
        "source_file",
        "module",
        "compilation_unit",
        "crate",
        "top_level",
        "block",
        "field_declaration",
        "field_definition_list",
    ):
        _walk_children(node, lang, spec, source, rel_path, qual_prefix, in_class, out)
    elif node.type in ("lexical_declaration", "variable_declaration"):
        # JS: const f = () => {}; capture arrow functions
        for child in node.named_children:
            if child.type == "variable_declarator":
                vn = _field(child, "name")
                value = _field(child, "value")
                if (
                    vn is not None
                    and value is not None
                    and value.type in ("arrow_function", "function_expression")
                ):
                    name = _text(vn)
                    qual = f"{qual_prefix}{name}"
                    out.append(
                        RawDefinition(
                            kind="function",
                            name=name,
                            qualified_name=qual,
                            file=rel_path,
                            span=_span_of(value),
                            params=_params_of(value, lang),
                            is_test=_is_test_symbol(name, rel_path, []),
                            signature=_line(source, node.start_point[0]),
                        )
                    )
                elif vn is not None and value is not None and value.type in t.class_types:
                    _extract(value, lang, spec, source, rel_path, qual_prefix, in_class, out)


def _walk_children(node, lang, spec, source, rel_path, qual_prefix, in_class, out) -> None:
    for child in node.named_children:
        _extract(child, lang, spec, source, rel_path, qual_prefix, in_class, out)


def extract_generic(source: str, rel_path: str, lang: Language) -> RawStructure:
    spec = languages.language_for_path(rel_path)
    structure = RawStructure(file=rel_path, module_name=rel_path, language=lang.value)
    try:
        ts_lang = languages.get_language(spec.ts_id) if spec else None
        from tree_sitter import Parser

        parser = Parser(ts_lang)
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
        if root.has_error:
            structure.parse_error = True
            structure.raw_error = "tree-sitter parse errors present"

        structure.module_name = _module_name_for(lang, rel_path, root)
        structure.package = structure.module_name

        defs: list[RawDefinition] = []
        _walk_children(root, lang, spec, source, "", None, defs)
        # dedupe by (qualified_name, start_line)
        seen: set[tuple[str, int]] = set()
        uniq: list[RawDefinition] = []
        for d in defs:
            key = (d.qualified_name, d.span.start_line)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        structure.definitions = uniq

        structure.imports = IMPORT_HANDLERS[lang](root, rel_path)

        calls: list[RawCall] = []
        stack = [root]
        while stack:
            node = stack.pop()
            _calls_of_node(node, lang, calls)
            stack.extend(node.named_children)
        # dedupe by (callee, line)
        seen_c: set[tuple[str, int]] = set()
        uniq_c: list[RawCall] = []
        for c in calls:
            key = (c.callee, c.line, c.receiver or "")
            if key in seen_c:
                continue
            seen_c.add(key)
            uniq_c.append(c)
        structure.calls = uniq_c

        raises: list[RawRaise] = []
        stack = [root]
        while stack:
            node = stack.pop()
            r = _raise_of_node(node, lang)
            if r:
                raises.append(r)
            stack.extend(node.named_children)
        structure.raises = raises

        # JS/TS: detect route registrations (app.get("/path", handler))
        if lang in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            for d in structure.definitions:
                for c in structure.calls:
                    if (
                        c.callee in ("get", "post", "put", "delete", "patch")
                        and c.is_method_call
                        and c.line >= d.span.start_line_1based
                        and c.line <= d.span.end_line_1based
                    ):
                        d.is_api_endpoint = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("generic parse failed for %s: %s", rel_path, exc)
        structure.parse_error = True
        structure.raw_error = f"generic: {exc}"
    return structure
