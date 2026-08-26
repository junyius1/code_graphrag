"""Python extractor: tree-sitter (CST structure/spans) + stdlib ast (semantics).

Definitions, qualified names, spans and decorators come from the tree-sitter CST
(a single code path shared with every language). Imports (with relative level),
call edges, raises, docstrings, module-level constants and parameter details come
from the stdlib ``ast``, which resolves those for Python more reliably than a
generic CST walk. The two are merged by source position.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from code_graphrag.config.models import Language
from code_graphrag.logging_setup import get_logger
from code_graphrag.parsing.common import (
    RawAssignment,
    RawCall,
    RawDefinition,
    RawImport,
    RawParam,
    RawRaise,
    RawStructure,
    Span,
)

logger = get_logger(__name__)

_DECOR_LIST = "decoration_list"
_DECOR = "decoration"


def _span_of(node) -> Span:
    return Span(
        start_line=node.start_point[0],
        start_col=node.start_point[1],
        end_line=node.end_point[0],
        end_col=node.end_point[1],
    )


def _decorators_from_node(node) -> list[str]:
    decs: list[str] = []
    decor_field = node.child_by_field_name("decorators")
    if decor_field is None:
        for child in node.children:
            if child.type == _DECOR_LIST:
                decor_field = child
                break
    if decor_field is not None:
        for child in decor_field.named_children:
            text = child.text.decode("utf-8", "replace")
            decs.append(text.strip().lstrip("@").strip())
    return decs


_ANNOTATED_PARAM_TYPES = (
    "typed_parameter",
    "typed_default_parameter",
    "keyword_parameter",
)
_DEFAULTED_PARAM_TYPES = ("typed_default_parameter", "default_parameter")
_SPLAT_PARAM_TYPES = (
    "list_splat_pattern",
    "dictionary_splat_pattern",
    "rest_parameter",
    "star_parameter",
)


def _params_from_node(node) -> list[RawParam]:
    """Extract parameter names/annotations from a tree-sitter ``parameters`` node.

    Handles every parameter node type produced by the current Python grammar:
    plain identifiers, annotated params, params with defaults, *args/**kwargs,
    and tuple-unpacking patterns.
    """
    params: list[RawParam] = []
    pend = node.child_by_field_name("parameters")
    if pend is None:
        return params
    for i, child in enumerate(pend.named_children):
        name = None
        annotation = None
        ctype = child.type
        if ctype in _ANNOTATED_PARAM_TYPES:
            n = child.child_by_field_name("name")
            t = child.child_by_field_name("type")
            if n is not None:
                name = n.text.decode("utf-8", "replace")
                annotation = t.text.decode("utf-8", "replace") if t else None
            else:
                # Some grammars (e.g. typed_parameter) expose no name/type
                # fields; recover them from the child nodes: first identifier
                # is the name, last annotation-like node is the type.
                named = [c for c in child.named_children]
                name = None
                annotation = None
                for c in named:
                    if c.type in ("identifier", "pattern", "tuple_pattern"):
                        name = c.text.decode("utf-8", "replace")
                        break
                if name is None and named:
                    name = named[0].text.decode("utf-8", "replace")
                for c in reversed(named):
                    if c.type in (
                        "type",
                        "type_hint",
                        "annotation",
                        "string",
                        "number",
                        "subscript",
                    ):
                        annotation = c.text.decode("utf-8", "replace")
                        break
        elif ctype == "identifier":
            name = child.text.decode("utf-8", "replace")
        elif ctype == "default_parameter":
            n = child.child_by_field_name("name")
            name = n.text.decode("utf-8", "replace") if n else None
        elif ctype in ("pattern", "tuple_pattern"):
            name = child.text.decode("utf-8", "replace")
        elif ctype in _SPLAT_PARAM_TYPES:
            name = child.text.decode("utf-8", "replace").lstrip("*").lstrip(")")
        if name is None:
            continue
        # has_default: the param node is a *_default_parameter, or a bare
        # default_value sibling immediately follows it.
        has_default = ctype in _DEFAULTED_PARAM_TYPES
        if not has_default:
            nxt = pend.named_children[i + 1] if i + 1 < len(pend.named_children) else None
            if nxt is not None and nxt.type == "default_value":
                has_default = True
        params.append(RawParam(name=name, annotation=annotation, has_default=has_default))
    return params


def _bases_from_node(node) -> list[str]:
    args = node.child_by_field_name("superclasses")
    if args is None:
        return []
    bases: list[str] = []
    for child in args.named_children:
        bases.append(child.text.decode("utf-8", "replace"))
    return bases


def _module_from_file(rel_path: str) -> str:
    name = rel_path.replace("\\", "/")
    if name.endswith(".pyi"):
        name = name[: -len(".pyi")]
    elif name.endswith(".py"):
        name = name[: -len(".py")]
    parts = name.split("/")
    parts = [p for p in parts if p and p != "__init__"]
    return ".".join(parts) if parts else rel_path


def _is_test_file(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    base = parts[-1]
    return (
        any(p in ("tests", "test", "spec") for p in parts[:-1])
        or base.startswith("test_")
        or base.endswith("_test.py")
    )


def _walk_definitions(
    node,
    module: str,
    qual_prefix: str,
    file: str,
    out: list[RawDefinition],
    enclosing_class: str | None,
    in_test_file: bool,
    source: bytes,
) -> None:
    for child in node.named_children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            name = name_node.text.decode("utf-8", "replace") if name_node else "?"
            qual = f"{qual_prefix}{name}"
            bases = _bases_from_node(child)
            body = child.child_by_field_name("body")
            out.append(
                RawDefinition(
                    kind="class",
                    name=name,
                    qualified_name=qual,
                    file=file,
                    span=_span_of(child),
                    decorators=_decorators_from_node(child),
                    base=bases[0] if bases else None,
                    annotations={"bases": ", ".join(bases)} if bases else {},
                    is_test=in_test_file,
                    signature=_line(source, child.start_point[0]),
                )
            )
            if body is not None:
                _walk_definitions(body, module, f"{qual}.", file, out, name, in_test_file, source)
        elif child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            name = name_node.text.decode("utf-8", "replace") if name_node else "?"
            qual = f"{qual_prefix}{name}"
            is_method = enclosing_class is not None
            ret = child.child_by_field_name("return_type")
            out.append(
                RawDefinition(
                    kind="method" if is_method else "function",
                    name=name,
                    qualified_name=qual,
                    file=file,
                    span=_span_of(child),
                    params=_params_from_node(child),
                    decorators=_decorators_from_node(child),
                    annotations={"returns": ret.text.decode("utf-8", "replace")}
                    if ret is not None
                    else {},
                    is_test=in_test_file,
                    is_cli_entry=_is_cli_decorated(child),
                    is_api_endpoint=_is_api_decorated(child),
                    signature=_line(source, child.start_point[0]),
                )
            )
        elif child.type in ("block", "module"):
            _walk_definitions(
                child, module, qual_prefix, file, out, enclosing_class, in_test_file, source
            )


def _line(source: bytes, line0: int) -> str:
    try:
        return source.splitlines()[line0].decode("utf-8", "replace").strip()
    except IndexError:
        return ""


def _is_cli_decorated(node) -> bool:
    for d in _decorators_from_node(node):
        if any(k in d for k in ("command", "option", "fire", "argparse", "click")):
            return True
    return False


def _is_api_decorated(node) -> bool:
    for d in _decorators_from_node(node):
        if any(
            k in d.lower()
            for k in (
                "route",
                "api_view",
                "get",
                "post",
                "put",
                "delete",
                "router",
                "blueprint",
                "endpoint",
            )
        ):
            return True
    return False


# ---- stdlib ast pass -------------------------------------------------------


def _ast_imports(tree: ast.AST, file: str) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(
                    RawImport(
                        module=a.name,
                        level=0,
                        symbols=[a.asname or a.name],
                        alias={a.asname or a.name: a.name},
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = ("." * node.level) + (module or "")
            symbols = [a.asname or a.name for a in node.names]
            imports.append(
                RawImport(
                    module=module,
                    level=node.level,
                    symbols=symbols,
                    alias={a.asname or a.name: a.name for a in node.names},
                    line=node.lineno,
                )
            )
    return imports


@dataclass
class _CallCtx:
    callee: str
    is_method: bool
    receiver: str | None
    line: int


def _ast_calls(tree: ast.AST) -> list[_CallCtx]:
    ctxs: list[_CallCtx] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                ctxs.append(
                    _CallCtx(callee=func.id, is_method=False, receiver=None, line=node.lineno)
                )
            elif isinstance(func, ast.Attribute):
                base = _attr_base(func.value)
                ctxs.append(
                    _CallCtx(
                        callee=func.attr,
                        is_method=True,
                        receiver=base,
                        line=node.lineno,
                    )
                )
    return ctxs


def _attr_base(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attr_base(node.value)
        return f"{prefix}.{node.attr}"
    return ast.unparse(node) if hasattr(ast, "unparse") else "?"


def _ast_raises(tree: ast.AST) -> list[RawRaise]:
    raises: list[RawRaise] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if isinstance(exc, ast.Call):
                exc = exc.func  # raise ValueError("msg") -> ValueError
            if isinstance(exc, ast.Name):
                name = exc.id
            elif isinstance(exc, ast.Attribute):
                name = exc.attr
            else:
                name = ast.unparse(exc) if hasattr(ast, "unparse") else "?"
            raises.append(RawRaise(exception=name, line=node.lineno))
    return raises


def _ast_docstrings_and_decorators(tree: ast.AST) -> dict[tuple[int, str], dict]:
    """Map (start_line_0based, name) -> {docstring, decorators} for defs/classes."""
    out: dict[tuple[int, str], dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            decs = [ast.unparse(d) for d in node.decorator_list if hasattr(ast, "unparse")]
            out[(node.lineno - 1, node.name)] = {"docstring": doc, "decorators": decs}
    return out


def _ast_module_variables(tree: ast.AST) -> list[RawAssignment]:
    """Module-level assignments that look like configuration or named constants."""
    consts: list[RawAssignment] = []
    for node in tree.body:
        targets: list[str] = []
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            targets, value = [node.targets[0].id], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        else:
            continue
        name = targets[0]
        is_const = name.isupper()
        val = ast.unparse(value) if value is not None and hasattr(ast, "unparse") else "?"
        if is_const or _looks_like_config(name, val):
            consts.append(
                RawAssignment(
                    name=name, value_preview=val[:120], is_constant=is_const, line=node.lineno
                )
            )
    return consts


def _looks_like_config(name: str, value: str) -> bool:
    if name.startswith("__"):
        return False
    hints = (
        "path",
        "url",
        "uri",
        "port",
        "host",
        "timeout",
        "limit",
        "size",
        "rate",
        "key",
        "secret",
        "token",
        "endpoint",
        "base",
        "dir",
        "file",
        "retry",
        "max_",
        "min_",
        "batch",
    )
    if any(h in name.lower() for h in hints):
        return True
    return len(value) < 80 and ("." in value or "/" in value or value.isdigit())


# ---- public entrypoint -----------------------------------------------------


def extract_python(source: str, rel_path: str) -> RawStructure:
    from code_graphrag.parsing import languages

    spec = languages.language_for_path(rel_path)
    module = _module_from_file(rel_path)
    in_test_file = _is_test_file(rel_path)

    structure = RawStructure(file=rel_path, module_name=module, language=Language.PYTHON.value)

    # 1) tree-sitter: definitions + spans + decorators + params + bases
    try:
        lang = languages.get_language(spec.ts_id) if spec else None
        from tree_sitter import Parser

        parser = Parser(lang)
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
        if root.has_error:
            structure.parse_error = True
            structure.raw_error = "tree-sitter parse errors present"
        defs: list[RawDefinition] = []
        _walk_definitions(
            root, module, "", rel_path, defs, None, in_test_file, source.encode("utf-8")
        )
        structure.definitions = defs
    except Exception as exc:  # noqa: BLE001 - never fail the whole index on one file
        logger.warning("tree-sitter parse failed for %s: %s", rel_path, exc)
        structure.parse_error = True
        structure.raw_error = f"tree-sitter: {exc}"

    # 2) stdlib ast: imports, calls, raises, docstrings, constants
    try:
        ast_tree = ast.parse(source)
        structure.imports = _ast_imports(ast_tree, rel_path)
        for c in _ast_calls(ast_tree):
            structure.calls.append(
                RawCall(
                    callee=c.callee,
                    is_method_call=c.is_method,
                    receiver=c.receiver,
                    line=c.line,
                )
            )
        structure.raises = _ast_raises(ast_tree)
        structure.assignments = _ast_module_variables(ast_tree)
        info = _ast_docstrings_and_decorators(ast_tree)
        for d in structure.definitions:
            key = (d.span.start_line, d.name)
            meta = info.get(key)
            if meta:
                d.docstring = meta["docstring"]
                if meta["decorators"] and not d.decorators:
                    d.decorators = meta["decorators"]
    except SyntaxError as exc:
        logger.warning("python ast parse failed for %s: %s", rel_path, exc)
        structure.parse_error = True
        structure.raw_error = (structure.raw_error or "") + f"ast: {exc}"

    return structure
