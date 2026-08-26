"""Python parsing: tree-sitter CST + stdlib AST fusion, per-construct extraction."""

from __future__ import annotations

from code_graphrag.parsing.extractors.python import extract_python

SRC = '''
"""Module docstring."""
from __future__ import annotations
import json
from services import UserService

MAX_RETRIES = 3


class Base:
    """Base class."""
    pass


class Service(Base):
    """A service."""

    def __init__(self, cfg: dict, retries: int = 0) -> None:
        self.cfg = cfg
        self.retries = retries

    def register(self, name: str) -> str:
        """Register a name."""
        if not name:
            raise ValueError("empty")
        return self._save(name)

    def _save(self, name: str) -> str:
        return json.dumps({"name": name})


def helper(x: int) -> int:
    return x * 2
'''


def test_python_definitions():
    st = extract_python(SRC, "mod.py")
    names = {d.name for d in st.definitions}
    assert {"Base", "Service", "register", "_save", "helper", "__init__"} <= names
    kinds = {d.name: d.kind for d in st.definitions}
    assert kinds["Base"] == "class"
    assert kinds["Service"] == "class"
    assert kinds["helper"] == "function"
    assert kinds["register"] == "method"


def test_python_qualified_names():
    st = extract_python(SRC, "mod.py")
    qn = {d.name: d.qualified_name for d in st.definitions}
    assert qn["register"].endswith("Service.register")
    assert qn["helper"] in ("mod.helper", "helper")
    # module name derived from file
    assert st.module_name in ("mod", "mod.py")


def test_python_imports():
    st = extract_python(SRC, "mod.py")
    mods = {i.module for i in st.imports}
    assert "json" in mods
    assert "services" in mods
    # from-import captures the symbol
    sv = next(i for i in st.imports if i.module == "services")
    assert "UserService" in sv.symbols


def test_python_calls():
    st = extract_python(SRC, "mod.py")
    callees = {c.callee for c in st.calls}
    # self._save call, json.dumps call
    assert any("_save" in c for c in callees)
    assert any("dumps" in c or c == "json.dumps" for c in callees)


def test_python_inheritance():
    st = extract_python(SRC, "mod.py")
    svc = next(d for d in st.definitions if d.name == "Service")
    assert svc.base is not None
    assert "Base" in svc.base


def test_python_raises():
    st = extract_python(SRC, "mod.py")
    raised = {r.exception for r in st.raises}
    assert "ValueError" in raised


def test_python_constants():
    st = extract_python(SRC, "mod.py")
    consts = [a.name for a in st.assignments if a.is_constant]
    assert "MAX_RETRIES" in consts


def test_python_source_locations():
    st = extract_python(SRC, "mod.py")
    reg = next(d for d in st.definitions if d.name == "register")
    # 1-based line numbers must be positive and the class must span its methods
    assert reg.span.start_line_1based > 0
    assert reg.span.end_line_1based >= reg.span.start_line_1based
    svc = next(d for d in st.definitions if d.name == "Service")
    assert svc.span.start_line_1based < reg.span.start_line_1based
    assert svc.span.end_line_1based >= reg.span.end_line_1based


def test_python_params_and_defaults():
    st = extract_python(SRC, "mod.py")
    init = next(d for d in st.definitions if d.name == "__init__")
    param_names = [p.name for p in init.params]
    assert "self" in param_names
    assert "cfg" in param_names
    assert "retries" in param_names
    # retries has a default, cfg does not
    retries = next(p for p in init.params if p.name == "retries")
    cfg_p = next(p for p in init.params if p.name == "cfg")
    assert retries.has_default is True
    assert cfg_p.has_default is False


def test_python_control_flow_captures_calls_inside_if():
    """Calls under if/else blocks must still be captured (nested body)."""
    st = extract_python(SRC, "mod.py")
    reg_calls = [c for c in st.calls]
    # the raise + return inside the if-body of register should be represented
    assert any(c.callee.startswith("self._save") or c.callee == "_save" for c in reg_calls)


def test_python_bad_code_is_failure_not_exception():
    bad = "def broken(:\n  pass\n"
    st = extract_python(bad, "bad.py")
    # extractor must not raise; it flags a parse error
    assert st is not None
    assert st.parse_error is True
