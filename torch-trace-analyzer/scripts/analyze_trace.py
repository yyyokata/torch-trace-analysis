#!/usr/bin/env python3
import ast
import json
import gzip
import sys
import io
import os
import re
import copy
import tarfile
import argparse
import subprocess
import textwrap
import tokenize
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Module identity registration.
#
# `frontend_html.py` (sibling module in this scripts/ directory) does a normal
# ``from analyze_trace import ASTFrontend, ...`` at module load time so the
# split between DAG core and HTML rendering is a clean, explicit dependency
# (no globals().update / no spec_from_file_location dynamic loading).
#
# Some entry points load this file directly via ``importlib.util.spec_from_
# file_location("analyze_trace_runtime", ...)`` (see
# ``testset/test_dag_rules.py::_load_analyze_module``).  In that scenario
# ``__name__`` is NOT ``"analyze_trace"`` and a downstream
# ``from analyze_trace import ...`` would otherwise re-execute this file via
# the normal import machinery, producing two parallel copies of every class.
#
# Registering the half-built module under the canonical name ``analyze_trace``
# (without overwriting an existing entry) makes the partial module visible
# to ``frontend_html`` so imports resolve to *this* instance regardless of
# how it was loaded.  All names referenced by frontend_html (ASTFrontend,
# _build_class_map, _strip_inline_comment, build_static_module_tree, ...) are
# defined further down in this file BEFORE ``from frontend_html import ...``
# at the bottom triggers the cross-module import.
#
# Three loading modes need to work:
#
#   1. Subprocess CLI: ``python3 scripts/analyze_trace.py ...``
#      ``__name__`` is ``"__main__"`` and the module is in ``sys.modules``.
#
#   2. Normal import: ``import analyze_trace``
#      ``__name__`` is ``"analyze_trace"`` so the registration is a no-op.
#
#   3. importlib.util.spec_from_file_location: ``test_dag_rules.py``
#      uses this with module name ``"analyze_trace_runtime"`` and does NOT
#      pre-register the module in ``sys.modules``.  We therefore look the
#      module up via the frame globals (``globals()``), which always points
#      at the currently-executing module dict regardless of how it was
#      created.
# ---------------------------------------------------------------------------
def _self_register_as_analyze_trace():
    """Make the currently-executing module reachable as ``analyze_trace``."""
    self_mod_name = globals().get("__name__")
    self_mod = sys.modules.get(self_mod_name) if self_mod_name else None
    if self_mod is None:
        # importlib loaders sometimes defer sys.modules registration.  Build
        # a lightweight proxy whose ``__getattr__`` delegates to our LIVE
        # frame globals so any name bound later in this file becomes
        # immediately visible through ``analyze_trace.<name>``.  Using
        # ``__getattr__`` (rather than copying ``__dict__``) avoids the
        # read-only ``Module.__dict__`` restriction in CPython.
        import types as _types

        _live_globals = globals()

        class _AnalyzeTraceProxy(_types.ModuleType):
            __slots__ = ()

            def __getattr__(self, name):
                try:
                    return _live_globals[name]
                except KeyError:
                    raise AttributeError(
                        f"module 'analyze_trace' has no attribute {name!r}"
                    ) from None

        self_mod = _AnalyzeTraceProxy("analyze_trace")
    sys.modules.setdefault("analyze_trace", self_mod)


_self_register_as_analyze_trace()
del _self_register_as_analyze_trace

# Ensure this scripts/ directory is on sys.path so the bottom-of-file
# ``from frontend_html import ...`` works even when analyze_trace.py is
# loaded by absolute path (e.g. via importlib.util.spec_from_file_location).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ---------------------------------------------------------------------------
# PR3 — ConstantResolver is now the only production evaluator path.
# Keep the module-level switch for compatibility with older tests / scripts,
# but default it to ``True`` and no longer maintain a legacy fallback path.
# ---------------------------------------------------------------------------
USE_NEW_EVAL: bool = True

# Retain the report hook for diagnostics compatibility. PR3 no longer records
# legacy/new diffs, so the log stays empty unless future instrumentation adds
# explicit entries.
_AB_EVAL_DIFFS: list = []

import torch.nn as _torch_nn

# PyTorch 内置容器类（无独立 forward 数据流，不作为 leaf stub）
_NATIVE_CONTAINER_NAMES: frozenset = frozenset({
    'ModuleDict', 'ModuleList', 'Sequential',
    'ParameterDict', 'ParameterList',
})

def _is_nn_leaf_stub(class_name: str) -> bool:
    """True iff nn.<class_name> 是叶子计算 Module（非容器、且是 nn.Module 子类）。
    用于动态判断 native stub 是否应加入 torch_native_module_classes。
    """
    if class_name in _NATIVE_CONTAINER_NAMES:
        return False
    cls = getattr(_torch_nn, class_name, None)
    if cls is None:
        return False
    try:
        return issubclass(cls, _torch_nn.Module)
    except TypeError:
        return False


class ASTFrontend:
    """统一的 AST 元数据前端。

    Phase A 目标：在不接入现有 DAG 逻辑的前提下，提供独立的类/方法/
    `__init__` 赋值元数据查询接口，供后续逐步替换手写正则解析。
    """

    _NN_MODULE_BASES = {"nn.Module", "torch.nn.Module", "Module"}

    def __init__(self, source: str = None, path: str = None):
        if source is None and path is None:
            raise ValueError("ASTFrontend requires either source or path")

        # Allow both `source` and `path` to be supplied simultaneously:
        #   - When `source` is provided, parse from it (in-memory content).
        #   - `path` is retained as filename metadata for error messages / logs,
        #     and used to read source from disk only when `source` is None.
        # Previously this constructor raised ValueError when both were given,
        # which forced every call site like `ASTFrontend(source=..., path=...)`
        # into the regex fallback path. That bug is fixed here.
        self.path = path
        if source is None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        self.source = source
        # Lazy-built caches to keep _node_to_text O(1) per call.
        self._source_bytes = None
        self._line_starts = None
        self.tree = ast.parse(self.source, filename=path or "<memory>")
        self.class_registry = {}
        self._build_class_registry()

    def _build_class_registry(self):
        for node in self.tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [self._node_to_text(base) for base in node.bases]
            methods = []
            method_map = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {
                        "name": item.name,
                        "lineno": getattr(item, "lineno", None),
                        "end_lineno": getattr(item, "end_lineno", getattr(item, "lineno", None)),
                    }
                    methods.append(method_info)
                    method_map[item.name] = method_info
            self.class_registry[node.name] = {
                "bases": bases,
                "methods": methods,
                "method_map": method_map,
                "is_nn_module": None,
                "node": node,
            }

        for class_name in list(self.class_registry.keys()):
            self.class_registry[class_name]["is_nn_module"] = self._resolve_is_nn_module(class_name, set())

    def _node_to_text(self, node):
        # Fast path: ast.get_source_segment internally re-splits self.source on
        # every call, which is O(n) per node. With thousands of AST nodes that
        # quickly becomes O(n^2). We bypass it by slicing pre-computed line
        # offsets instead. Falls back to ast.unparse if positions are missing.
        try:
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", None)
            col_offset = getattr(node, "col_offset", None)
            end_col_offset = getattr(node, "end_col_offset", None)
            if (
                lineno is not None
                and end_lineno is not None
                and col_offset is not None
                and end_col_offset is not None
            ):
                line_starts = self._line_starts
                if line_starts is None:
                    line_starts = self._build_line_starts()
                src = self._source_bytes
                start = line_starts[lineno - 1] + col_offset
                end = line_starts[end_lineno - 1] + end_col_offset
                if 0 <= start <= end <= len(src):
                    seg = src[start:end].decode("utf-8", errors="replace")
                    if seg:
                        return seg.strip()
        except Exception:
            pass
        try:
            return ast.unparse(node)
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts))
            if isinstance(node, ast.Constant):
                return repr(node.value)
            return "<unknown>"

    def _build_line_starts(self):
        # Build byte-offset-of-start-of-each-line so _node_to_text can slice
        # self.source by (lineno, col_offset, end_lineno, end_col_offset)
        # without re-splitting the entire source on every call. col_offset is
        # a *byte* offset into the line (per Python AST contract), so we
        # operate on the UTF-8 encoded bytes here.
        src_bytes = self.source.encode("utf-8")
        self._source_bytes = src_bytes
        starts = [0]
        for i, b in enumerate(src_bytes):
            if b == 0x0A:  # '\n'
                starts.append(i + 1)
        # Sentinel for last line end-of-file
        starts.append(len(src_bytes))
        self._line_starts = starts
        return starts

    def _resolve_is_nn_module(self, class_name, visiting):
        info = self.class_registry.get(class_name)
        if not info:
            return False
        cached = info.get("is_nn_module")
        if cached is not None:
            return cached
        if class_name in visiting:
            return False
        visiting.add(class_name)
        try:
            for base in info.get("bases", []):
                if base in self._NN_MODULE_BASES or base.endswith(".Module"):
                    return True
                base_leaf = base.split(".")[-1]
                if base_leaf == "Module":
                    return True
                if base_leaf in self.class_registry and self._resolve_is_nn_module(base_leaf, visiting):
                    return True
            return False
        finally:
            visiting.discard(class_name)

    def is_nn_module(self, class_name) -> bool:
        info = self.class_registry.get(class_name)
        return bool(info and info.get("is_nn_module"))

    def get_method_lines(self, class_name, method_name):
        info = self.class_registry.get(class_name)
        if not info:
            return None
        method = info.get("method_map", {}).get(method_name)
        if not method:
            return None
        start = method.get("lineno")
        end = method.get("end_lineno")
        if start is None or end is None:
            return None
        return start, end

    def get_init_assignments_ast(self, class_name):
        method_node = self._get_method_node(class_name, "__init__")
        if method_node is None:
            return []

        assignments = []
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            ctor_name = self._node_to_text(node.value.func)
            for target in node.targets:
                attr = self._extract_self_attr_name(target)
                if not attr:
                    continue
                assignments.append({
                    "attr": attr,
                    "class": ctor_name,
                    "lineno": getattr(node, "lineno", None),
                    "args": [self._node_to_text(arg) for arg in node.value.args],
                    "kwargs": {
                        kw.arg: self._node_to_text(kw.value)
                        for kw in node.value.keywords
                        if kw.arg is not None
                    },
                })
        assignments.sort(key=lambda x: (x.get("lineno") is None, x.get("lineno") or 0, x["attr"]))
        return assignments

    def get_reachable_helpers(self, class_name, entry_method="forward"):
        info = self.class_registry.get(class_name)
        if not info:
            return set()
        method_map = info.get("method_map", {})
        if entry_method not in method_map:
            return set()

        reachable = set()
        visited = set()

        def dfs(method_name):
            if method_name in visited:
                return
            visited.add(method_name)
            method_node = self._get_method_node(class_name, method_name)
            if method_node is None:
                return
            for call in ast.walk(method_node):
                if not isinstance(call, ast.Call):
                    continue
                helper_name = self._extract_self_method_name(call.func)
                if helper_name and helper_name in method_map and helper_name != method_name:
                    if helper_name not in reachable:
                        reachable.add(helper_name)
                    dfs(helper_name)

        dfs(entry_method)
        reachable.discard(entry_method)
        return reachable

    def get_module_calls(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return []
        info = self.class_registry.get(class_name) or {}
        method_map = info.get("method_map", {})
        calls = []
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Call):
                continue
            call_info = self._extract_module_call_info(node)
            if not call_info:
                continue
            attr_name = call_info.get("attr")
            if attr_name == "forward":
                continue
            if attr_name in method_map:
                continue
            calls.append(call_info)
        calls.sort(key=lambda item: (
            item.get("line") is None,
            item.get("line") or 0,
            item.get("col") is None,
            item.get("col") or 0,
            item.get("attr") or item.get("name") or "",
        ))
        return calls

    def _joinedstr_affixes(self, node):
        if not isinstance(node, ast.JoinedStr):
            return None
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append(None)
        prefix_parts = []
        for part in parts:
            if part is None:
                break
            prefix_parts.append(part)
        suffix_parts = []
        for part in reversed(parts):
            if part is None:
                break
            suffix_parts.append(part)
        return "".join(prefix_parts), "".join(reversed(suffix_parts))

    def _dynamic_attr_expr_matches(self, expr, target_attr):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value == target_attr
        affixes = self._joinedstr_affixes(expr)
        if affixes is None:
            return False
        prefix, suffix = affixes
        if prefix and not target_attr.startswith(prefix):
            return False
        if suffix and not target_attr.endswith(suffix):
            return False
        return bool(prefix or suffix)

    def _is_self_getattr_call(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
        )

    def get_first_call_loc(self, class_name, attr_name):
        method_names = ["forward"]
        if self.get_method_lines(class_name, "forward") is None:
            return None
        method_names.extend(sorted(self.get_reachable_helpers(class_name, "forward")))
        target_attr = attr_name.split("[", 1)[0] if "[" in attr_name else attr_name
        best = None
        for method_name in method_names:
            for call in self.get_module_calls(class_name, method_name):
                called_attr = call.get("attr")
                if called_attr == attr_name or called_attr == target_attr:
                    line = call.get("line")
                    if line is None:
                        continue
                    cand = (self.path or "", line)
                    if best is None or line < best[1]:
                        best = cand
            method_node = self._get_method_node(class_name, method_name)
            if method_node is None:
                continue
            alias_bindings = defaultdict(list)
            nodes = sorted(
                ast.walk(method_node),
                key=lambda node: (
                    getattr(node, "lineno", 10 ** 9),
                    getattr(node, "col_offset", 10 ** 9),
                ),
            )
            for node in nodes:
                if isinstance(node, ast.Assign) and self._is_self_getattr_call(node.value):
                    if self._dynamic_attr_expr_matches(node.value.args[1], target_attr):
                        assign_line = getattr(node, "lineno", None)
                        for target in node.targets:
                            if isinstance(target, ast.Name) and assign_line is not None:
                                alias_bindings[target.id].append(assign_line)
                    continue
                if isinstance(node, ast.Call) and self._is_self_getattr_call(node.func):
                    if self._dynamic_attr_expr_matches(node.func.args[1], target_attr):
                        line = getattr(node, "lineno", None)
                        if line is None:
                            continue
                        cand = (self.path or "", line)
                        if best is None or line < best[1]:
                            best = cand
                        continue
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    alias_lines = alias_bindings.get(node.func.id) or []
                    line = getattr(node, "lineno", None)
                    if line is None:
                        continue
                    if any(alias_line <= line for alias_line in alias_lines):
                        cand = (self.path or "", line)
                        if best is None or line < best[1]:
                            best = cand
        return best

    def _lookup_var_env_entry(self, env, var_name, phys_lineno=None):
        chain = env.get(var_name) or []
        if not isinstance(chain, list):
            chain = [chain]
        matched = None
        for item in chain:
            if not item:
                continue
            line = item[1] if len(item) > 1 else None
            if phys_lineno is None:
                matched = item
                continue
            if line is not None and line <= phys_lineno:
                matched = item
        return matched

    def build_var_env(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return {}
        env = {}
        alias_env = {}
        stmt_infos = self._get_stmt_infos(class_name, method_name)
        for info in stmt_infos:
            line = info.get("line")
            rhs_vars = info.get("rhs_vars") or []
            targets = [target for target in (info.get("targets") or []) if target and target != "_"]
            producer_attr = info.get("producer_attr")
            if not producer_attr:
                for rhs_var in rhs_vars:
                    root = alias_env.get(rhs_var, rhs_var)
                    entry = self._lookup_var_env_entry(env, root, line)
                    if entry:
                        producer_attr = entry[0]
                        break
            if producer_attr:
                for target in targets:
                    env.setdefault(target, []).append((producer_attr, line))
            if info.get("kind") == "assign" and len(rhs_vars) == 1:
                root = alias_env.get(rhs_vars[0], rhs_vars[0])
                for target in targets:
                    alias_env[target] = root
        return env

    def build_alias_env(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return {}
        aliases = {}
        for info in self._get_stmt_infos(class_name, method_name):
            if info.get("kind") != "assign":
                continue
            if info.get("producer_attr"):
                continue
            rhs_vars = info.get("rhs_vars") or []
            targets = info.get("targets") or []
            if len(rhs_vars) == 1:
                for target in targets:
                    if target and target != "_":
                        aliases[target] = rhs_vars[0]
        return aliases

    def build_dict_slot_env(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return {}
        slot_env = {}
        var_env = self.build_var_env(class_name, method_name)
        alias_env = self.build_alias_env(class_name, method_name)
        for info in self._get_stmt_infos(class_name, method_name):
            if info.get("kind") != "dict_write":
                continue
            dict_var = info.get("dict_var")
            dict_key = info.get("dict_key")
            if not dict_var or dict_key is None:
                continue
            producer_attr = info.get("producer_attr")
            line = info.get("line")
            if not producer_attr:
                for rhs_var in info.get("rhs_vars") or []:
                    root = alias_env.get(rhs_var, rhs_var)
                    entry = self._lookup_var_env_entry(var_env, root, line)
                    if entry:
                        producer_attr = entry[0]
                        break
            if producer_attr:
                slot_env[(dict_var, dict_key)] = (producer_attr, line)
        return slot_env

    def extract_return_vars(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return []
        out = []
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
            vars_ = []
            for value in values:
                vars_.extend(self._extract_name_list(value))
            out.append({
                "line": getattr(node, "lineno", None),
                "vars": vars_,
            })
        out.sort(key=lambda item: (item.get("line") is None, item.get("line") or 0))
        return out

    def get_self_param_aliases(self, class_name, method_name="__init__"):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return {}
        aliases = {}
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.value, ast.Name):
                continue
            attr_name = self._extract_self_attr_name(node.targets[0])
            if attr_name:
                aliases[attr_name] = node.value.id
        return aliases

    def get_dynamic_indexed_self_attrs(self, class_name):
        out = set()
        info = self.class_registry.get(class_name)
        if not info:
            return out
        class_node = info.get("node")
        if class_node is None:
            return out
        for item in class_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name == "__init__":
                continue
            for node in ast.walk(item):
                if not isinstance(node, ast.Subscript):
                    continue
                owner = node.value
                if not (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self"):
                    continue
                idx_node = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
                if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
                    continue
                # Also treat ``-N`` as a literal integer index (UnaryOp(USub, Constant(int))).
                if (
                    isinstance(idx_node, ast.UnaryOp)
                    and isinstance(idx_node.op, ast.USub)
                    and isinstance(idx_node.operand, ast.Constant)
                    and isinstance(idx_node.operand.value, int)
                ):
                    continue
                out.add(owner.attr)
        return out

    def get_loop_expansion_records(self, class_name, method_name):
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            return []
        records = []
        const_table = self.get_constant_table(self.path or "<memory>")
        const_resolver = ConstantResolver(const_table)

        def _parse_loop_spec(for_node):
            iter_node = for_node.iter
            if isinstance(iter_node, ast.Call):
                func_name = self._node_to_text(iter_node.func).strip()
                if func_name == "range" and len(iter_node.args) == 1:
                    return {
                        "loop_kind": "range",
                        "iter_expr": self._node_to_text(iter_node.args[0]).strip(),
                    }
                if func_name == "enumerate" and len(iter_node.args) >= 1:
                    return {
                        "loop_kind": "enumerate",
                        "iter_expr": self._node_to_text(iter_node.args[0]).strip(),
                    }
            iter_text = self._node_to_text(iter_node).strip()
            return {"loop_kind": "iter", "iter_expr": iter_text}

        def _append_record(stmt, container_attr, class_name):
            records.append({
                "line": getattr(stmt, "lineno", None),
                "container_attr": container_attr,
                "class_full": class_name,
                "class_leaf": class_name.split(".")[-1] if class_name else "",
            })

        def _walk_body(body, inherited_spec=None, local_ctor_vars=None):
            if local_ctor_vars is None:
                local_ctor_vars = {}
            for stmt in body or []:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        local_ctor_vars[stmt.targets[0].id] = self._node_to_text(stmt.value.func)
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                        owner = call.func.value
                        if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self" and len(call.args) == 1:
                            arg0 = call.args[0]
                            if isinstance(arg0, ast.Name) and arg0.id in local_ctor_vars:
                                rec = dict(inherited_spec or {})
                                _append_record(stmt, owner.attr, local_ctor_vars[arg0.id])
                                records[-1].update(rec)
                            elif isinstance(arg0, ast.Call):
                                rec = dict(inherited_spec or {})
                                _append_record(stmt, owner.attr, self._node_to_text(arg0.func))
                                records[-1].update(rec)
                    continue
                if isinstance(stmt, ast.For):
                    _walk_body(stmt.body, _parse_loop_spec(stmt), local_ctor_vars)
                    _walk_body(stmt.orelse, inherited_spec, local_ctor_vars)
                    continue
                if isinstance(stmt, ast.If):
                    scope = Scope(file=self.path or "<memory>", cls=class_name, method="__init__")
                    bool_val = const_resolver.eval_bool(stmt.test, scope)
                    if bool_val is True:
                        _walk_body(stmt.body, inherited_spec, local_ctor_vars)
                    elif bool_val is False:
                        _walk_body(stmt.orelse, inherited_spec, local_ctor_vars)
                    else:
                        _walk_body(stmt.body, inherited_spec, local_ctor_vars)
                        _walk_body(stmt.orelse, inherited_spec, local_ctor_vars)
                    continue
                if isinstance(stmt, ast.With):
                    _walk_body(stmt.body, inherited_spec, local_ctor_vars)
                    continue
                if isinstance(stmt, ast.Try):
                    _walk_body(stmt.body, inherited_spec, local_ctor_vars)
                    for h in stmt.handlers:
                        _walk_body(h.body, inherited_spec, local_ctor_vars)
                    _walk_body(stmt.orelse, inherited_spec, local_ctor_vars)
                    _walk_body(stmt.finalbody, inherited_spec, local_ctor_vars)

        _walk_body(method_node.body)
        records = [
            r for r in records
            if r.get("loop_kind") in {"range", "enumerate", "iter"}
        ]
        records.sort(key=lambda item: (
            item.get("line") is None,
            item.get("line") or 0,
            item.get("container_attr") or "",
            item.get("class_leaf") or "",
        ))
        return records

    def _get_method_node(self, class_name, method_name):
        info = self.class_registry.get(class_name)
        if not info:
            return None
        class_node = info.get("node")
        if class_node is None:
            return None
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                return item
        return None

    def _get_stmt_infos(self, class_name, method_name):
        cache = getattr(self, "_stmt_info_cache", None)
        if cache is None:
            cache = {}
            self._stmt_info_cache = cache
        key = (class_name, method_name)
        if key in cache:
            return cache[key]
        method_node = self._get_method_node(class_name, method_name)
        if method_node is None:
            cache[key] = []
            return cache[key]
        infos = []
        for node in ast.walk(method_node):
            if isinstance(node, ast.Assign):
                info = self._stmt_info_from_assign(node)
                if info:
                    infos.append(info)
            elif isinstance(node, ast.AugAssign):
                info = self._stmt_info_from_augassign(node)
                if info:
                    infos.append(info)
            elif isinstance(node, ast.Expr):
                info = self._stmt_info_from_expr(node)
                if info:
                    infos.append(info)
            elif isinstance(node, ast.Return):
                infos.append({
                    "kind": "return",
                    "line": getattr(node, "lineno", None),
                    "rhs_text": self._node_to_text(node.value) if node.value is not None else "",
                    "rhs_vars": self._extract_name_list(node.value) if node.value is not None else [],
                })
        infos.sort(key=lambda item: (item.get("line") is None, item.get("line") or 0, item.get("kind") or ""))
        cache[key] = infos
        return infos

    def _stmt_info_from_assign(self, node):
        rhs_text = self._node_to_text(node.value)
        rhs_vars = self._extract_name_list(node.value)
        info = {
            "kind": "assign",
            "line": getattr(node, "lineno", None),
            "rhs_text": rhs_text,
            "rhs_vars": rhs_vars,
            "rhs_node": node.value,
            "targets": [],
            "producer_attr": self._extract_called_self_attr(node.value.func) if isinstance(node.value, ast.Call) else None,
        }
        dict_write = False
        for target in node.targets:
            names = self._extract_target_names(target)
            if names:
                info["targets"].extend(names)
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                dict_write = True
                info["kind"] = "dict_write"
                info["dict_var"] = target.value.id
                idx_node = target.slice
                if isinstance(idx_node, ast.Index):
                    idx_node = idx_node.value
                info["dict_key"] = self._literal_key(idx_node)
        return info

    def _stmt_info_from_augassign(self, node):
        return {
            "kind": "augassign",
            "line": getattr(node, "lineno", None),
            "rhs_text": self._node_to_text(node.value),
            "rhs_vars": self._extract_name_list(node.value),
            "targets": self._extract_target_names(node.target),
            "producer_attr": self._extract_called_self_attr(node.value.func) if isinstance(node.value, ast.Call) else None,
        }

    def _stmt_info_from_expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call):
            return None
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "append":
            return None
        if not isinstance(func.value, ast.Name):
            return None
        rhs_node = call.args[0] if call.args else None
        return {
            "kind": "append",
            "line": getattr(node, "lineno", None),
            "target_var": func.value.id,
            "rhs_text": self._node_to_text(rhs_node) if rhs_node is not None else "",
            "rhs_vars": self._extract_name_list(rhs_node) if rhs_node is not None else [],
            "rhs_node": rhs_node,
            "call_node": call,
            "producer_attr": self._extract_called_self_attr(rhs_node.func) if isinstance(rhs_node, ast.Call) else None,
        }

    def _extract_target_names(self, target):
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            out = []
            for elt in target.elts:
                out.extend(self._extract_target_names(elt))
            return out
        return []

    def _extract_name_list(self, node):
        if node is None:
            return []
        out = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out.append(sub.id)
        return out

    def _literal_key(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        # AST-only fallback: parse the textual representation as an
        # expression and inspect the resulting node.  No regex is used.
        text = self._node_to_text(node).strip()
        try:
            tree = ast.parse(text, mode='eval')
        except SyntaxError:
            return '*'
        body = tree.body
        if isinstance(body, ast.Constant):
            if isinstance(body.value, str):
                return body.value
            if isinstance(body.value, int):
                return body.value
        # Negative integer literal e.g. ``-1`` parses as UnaryOp(USub, Constant(int)).
        if (
            isinstance(body, ast.UnaryOp)
            and isinstance(body.op, ast.USub)
            and isinstance(body.operand, ast.Constant)
            and isinstance(body.operand.value, int)
        ):
            return -body.operand.value
        return '*'

    def _line_excerpt(self, lineno):
        if lineno is None:
            return None
        lines = self.source.splitlines()
        if not (1 <= lineno <= len(lines)):
            return None
        return lines[lineno - 1]

    def _extract_self_attr_name(self, target):
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            return target.attr
        return None

    def _normalize_index_expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return str(node.value)
            if isinstance(node.value, str):
                return "'" + node.value.replace("'", "\\'") + "'"
        # AST-only fallback: parse textual representation and inspect the node.
        text = self._node_to_text(node).strip()
        try:
            tree = ast.parse(text, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        if isinstance(body, ast.Constant):
            if isinstance(body.value, int):
                return str(body.value)
            if isinstance(body.value, str):
                return "'" + body.value.replace("'", "\\'") + "'"
        return None

    def _extract_self_method_name(self, func_node):
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) and func_node.value.id == "self":
            return func_node.attr
        return None

    def _extract_module_call_info(self, node):
        if not isinstance(node, ast.Call):
            return None
        line = getattr(node, "lineno", None)
        col = getattr(node.func, "col_offset", getattr(node, "col_offset", None))
        func_end_col = getattr(node.func, "end_col_offset", None)
        info = {
            "line": line,
            "col": col,
            "func_end_col": func_end_col,
            "args": [self._node_to_text(arg) for arg in node.args],
            "args_text": ", ".join(self._node_to_text(arg) for arg in node.args),
            "node": node,
        }

        attr_name = self._extract_called_self_attr(node.func)
        if attr_name:
            info["attr"] = attr_name
            info["kind"] = "self"
            if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Attribute):
                idx_node = node.func.slice
                if isinstance(idx_node, ast.Index):
                    idx_node = idx_node.value
                info["container_attr"] = node.func.value.attr
                info["index_expr"] = self._node_to_text(idx_node).strip()
                info["kind"] = "self_indexed"
            elif isinstance(node.func, ast.Call):
                info["kind"] = "getattr_literal"
            return info

        if isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name) and node.func.func.id == "getattr":
            if len(node.func.args) >= 2 and isinstance(node.func.args[0], ast.Name) and node.func.args[0].id == "self":
                info["kind"] = "getattr_dynamic"
                info["name_expr"] = self._node_to_text(node.func.args[1]).strip()
                return info

        if isinstance(node.func, ast.Name):
            info["kind"] = "name"
            info["name"] = node.func.id
            return info

        if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name):
            idx_node = node.func.slice
            if isinstance(idx_node, ast.Index):
                idx_node = idx_node.value
            info["kind"] = "dict_index"
            info["dict_name"] = node.func.value.id
            info["index_expr"] = self._node_to_text(idx_node).strip()
            return info

        if isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Attribute):
            get_attr = node.func.func
            if isinstance(get_attr.value, ast.Name) and get_attr.attr == "get":
                info["kind"] = "dict_get"
                info["dict_name"] = get_attr.value.id
                info["index_expr"] = self._node_to_text(node.func.args[0]).strip() if node.func.args else ""
                return info

        return None

    def _extract_called_self_attr(self, func_node):
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) and func_node.value.id == "self":
            return func_node.attr
        if isinstance(func_node, ast.Subscript) and isinstance(func_node.value, ast.Attribute):
            base = func_node.value
            if isinstance(base.value, ast.Name) and base.value.id == "self":
                idx_node = func_node.slice
                if isinstance(idx_node, ast.Index):
                    idx_node = idx_node.value
                idx = self._normalize_index_expr(idx_node)
                return f"{base.attr}[{idx}]" if idx is not None else base.attr
        # Phase E2: helper-method chain forms.
        # ``self.<attr>.<helper>(...)``  → return ``<attr>``
        # ``self.<attr>[idx].<helper>(...)`` → return ``<attr>[idx]``
        # Without these the dead-child filter mistakenly prunes ``self.<attr>``
        # whenever the only call site is via a helper method on the submodule
        # (e.g. ``self.seq_trans.dense_query(...)``).
        if isinstance(func_node, ast.Attribute):
            inner = func_node.value
            # self.<attr>.<helper>(...)
            if (isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self"):
                return inner.attr
            # self.<attr>[idx].<helper>(...)
            if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Attribute):
                base = inner.value
                if isinstance(base.value, ast.Name) and base.value.id == "self":
                    idx_node = inner.slice
                    if isinstance(idx_node, ast.Index):
                        idx_node = idx_node.value
                    idx = self._normalize_index_expr(idx_node)
                    return f"{base.attr}[{idx}]" if idx is not None else base.attr
        if isinstance(func_node, ast.Call) and isinstance(func_node.func, ast.Name) and func_node.func.id == "getattr":
            if len(func_node.args) >= 2 and isinstance(func_node.args[0], ast.Name) and func_node.args[0].id == "self":
                lit = func_node.args[1]
                if isinstance(lit, ast.Constant) and isinstance(lit.value, str):
                    return lit.value
        return None

    # ------------------------------------------------------------------
    # PR1: ConstantTable lazy accessor
    # ------------------------------------------------------------------
    # In PR1 the new evaluator stack (ConstantTable + ConstantResolver) is
    # added as a *coexistent* skeleton next to the legacy regex+AST
    # evaluator.  No production code path consumes the new stack yet; this
    # accessor is the integration seam that future PRs (PR2 A/B comparison,
    # PR3 regex deletion) will hook into.
    #
    # The ``fname`` argument is accepted for API symmetry with the design
    # document's ``ConstantTable.build_all`` (which is intrinsically
    # cross-file).  In PR1 we build a *single-file* table per ASTFrontend
    # and cache it on the frontend instance so repeated calls are O(1).
    # When PR2 wires a global ``ast_frontends: Dict[fname, ASTFrontend]``
    # into ``build_static_module_tree`` we will switch to the cross-file
    # build path; the public method signature stays the same.
    def get_constant_table(self, fname: str = None):
        """Return a lazily-built :class:`ConstantTable` for this frontend.

        Parameters
        ----------
        fname : str, optional
            Filename label for entries inserted into the table.  Defaults
            to ``self.path`` or the literal string ``"<memory>"``.

        Returns
        -------
        ConstantTable
            A populated single-file constant table.  Subsequent calls
            return the same cached instance unless ``fname`` differs (in
            which case a new table is built and cached separately).
        """
        if not hasattr(self, "_constant_table_cache"):
            self._constant_table_cache = {}
        key = fname if fname is not None else (self.path or "<memory>")
        cached = self._constant_table_cache.get(key)
        if cached is not None:
            return cached
        table = ConstantTable.build_all(
            source_files={key: self.source.split("\n")},
            ast_frontends={key: self},
            nn_module_classes={
                cname for cname, info in self.class_registry.items()
                if info.get("is_nn_module")
            },
        )
        self._constant_table_cache[key] = table
        return table

    def parse_local_stmt(self, line: str):
        text = (line or "").strip()
        if not text:
            return None
        try:
            mod = ast.parse(text)
        except Exception:
            return None
        if len(mod.body) != 1:
            return None
        return mod.body[0]

    def _expr_leaf_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_leaf_name(node.value)
            return (base + "." if base else "") + node.attr
        return None

    def _local_node_to_text(self, node):
        # Local snippet nodes are parsed from standalone logical lines, so their
        # coordinates point into the snippet rather than the full source file.
        # Use ast.unparse instead of _node_to_text to avoid source-slice drift.
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def parse_local_ctor_assign(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if target is None:
            return None
        attr = self._extract_self_attr_name(target)
        if not attr:
            return None
        return {
            "attr": attr,
            "class_full": self._local_node_to_text(stmt.value.func),
            "kwargs": {kw.arg: self._local_node_to_text(kw.value) for kw in stmt.value.keywords if kw.arg is not None},
        }

    def parse_local_lg_assign(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if target is None:
            return None
        attr = self._extract_self_attr_name(target)
        if not attr:
            return None
        func_text = self._local_node_to_text(stmt.value.func)
        if func_text in {
            "LG.feature_column", "LG.dense_feature", "LG.get_sample_rate", "LG.get_bias"
        }:
            return {"attr": attr}
        return None

    def parse_local_fc_get_vector_assign(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if target is None:
            return None
        attr = self._extract_self_attr_name(target)
        if not attr:
            return None
        func = stmt.value.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get_vector"):
            return None
        owner = func.value
        if not isinstance(owner, ast.Subscript):
            return None
        owner_text = self._local_node_to_text(owner.value).strip()
        if not re.search(r'(?:^|_)(?:dict|fc)\w*$', owner_text) and not re.search(r'(?:dict|fc)', owner_text):
            return None
        return {"attr": attr}

    def parse_local_container_ctor(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if target is None:
            return None
        attr = self._extract_self_attr_name(target)
        if not attr:
            return None
        func_leaf = (self._expr_leaf_name(stmt.value.func) or "").split(".")[-1]
        if func_leaf not in {"ModuleList", "ModuleDict"}:
            return None
        return {"attr": attr, "kind": func_leaf, "call": stmt.value}

    def parse_local_append_ctor(self, stmt):
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "append":
            return None
        owner = call.func.value
        if not (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self"):
            return None
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
            return None
        return {
            "attr": owner.attr,
            "class_full": self._local_node_to_text(call.args[0].func),
        }

    def parse_local_subscript_ctor(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if not isinstance(target, ast.Subscript):
            return None
        owner = target.value
        if not (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self"):
            return None
        return {
            "attr": owner.attr,
            "key_expr": self._local_node_to_text(target.slice),
            "class_full": self._local_node_to_text(stmt.value.func),
        }

    def parse_local_var_ctor(self, stmt):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return None
        target = stmt.targets[0] if len(stmt.targets) == 1 else None
        if not isinstance(target, ast.Name):
            return None
        return {
            "name": target.id,
            "class_full": self._local_node_to_text(stmt.value.func),
        }

    def parse_local_setattr_ctor(self, stmt):
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        call = stmt.value
        func_name = self._expr_leaf_name(call.func)
        if func_name != "setattr" or len(call.args) != 3:
            return None
        owner = call.args[0]
        if not (isinstance(owner, ast.Name) and owner.id == "self"):
            return None
        name_expr = self._local_node_to_text(call.args[1]).strip()
        value = call.args[2]
        if isinstance(value, ast.Call):
            return {
                "name_expr": name_expr,
                "class_full": self._local_node_to_text(value.func),
                "two_step": False,
            }
        if isinstance(value, ast.Name):
            return {
                "name_expr": name_expr,
                "local_var": value.id,
                "two_step": True,
            }
        return None




# ===========================================================================
# PR1: ConstantTable + ConstantResolver skeleton
# ===========================================================================
# Reference: ast_refactor_workdir/output/ast_evaluator_redesign.md (sections
# 3-4).  These two classes form the "facts table + pure evaluator" pair that
# will eventually replace the legacy regex+AST mixed evaluator (`_eval_int_atom`
# / `_eval_list_len` / `_extract_kw_int_args` / `_extract_kw_list_lens` /
# `_resolve_range_n` / `_resolve_iter_len`) per the 5-stage PR plan.
#
# Scope of PR1 (this commit):
#   * Class skeletons exist and are *callable* (not just `pass` placeholders).
#   * ConstantTable's 4-pass build path scans real AST nodes for the 14 tables
#     listed in the design.  Multi-file dataclass field defaults / cross-file
#     positional binding chains can be built but are intentionally NOT wired
#     to any production caller.
#   * ConstantResolver dispatches over the 9 AST node types in section 4.2
#     (Constant / Name / Attribute / Subscript / BinOp / UnaryOp / Call /
#     List / Starred) with `_eval_cache` + recursion-guard placeholders.
#   * Failure path returns ``None`` -- never falls back to regex.
#
# Out of scope for PR1 (deferred to PR2+):
#   * No legacy code path is replaced.  The 6 legacy evaluator functions are
#     untouched; they continue to be the sole production evaluator.
#   * Integration into `build_static_module_tree` is limited to the new
#     ``ASTFrontend.get_constant_table()`` lazy accessor (above).  PR2 adds
#     the ``--use-new-eval`` A/B switch.
#   * `local_self_dict_literals` (referenced in section 4.6) is left as a
#     stub method.  PR5 wires it for the scenario-5 transformer cases.
#   * `EvalTrace` debug instrumentation is left as a no-op hook.
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Scope:
    """Evaluation context for ``ConstantResolver`` look-ups.

    Every public ``eval_*`` call requires a fully-formed ``Scope``.  When
    ``cls`` / ``method`` / ``parent_cls`` / ``parent_attr`` are unknown
    they MUST be passed as ``None`` rather than the empty string -- the
    table look-ups treat ``None`` as "not in any class/method", whereas
    an empty string would silently match unrelated entries.
    """

    file: str
    cls: Optional[str] = None
    method: Optional[str] = None
    parent_cls: Optional[str] = None
    parent_attr: Optional[str] = None

    @property
    def class_key(self) -> Tuple[str, Optional[str]]:
        return (self.file, self.cls)

    @property
    def method_key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.file, self.cls, self.method)

    @property
    def instance_key(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.parent_cls, self.parent_attr)


@dataclass(frozen=True)
class IntValue:
    """A resolved integer with provenance metadata for diagnostics."""

    value: int
    origin: str = "literal"


@dataclass(frozen=True)
class ListValue:
    """A resolved list-length (with optional original AST element nodes)."""

    length: int
    items: Optional[Tuple[Any, ...]] = None
    origin: str = "list_literal"


# Sentinel used by ConstantResolver._eval_cache to break recursive cycles
# (see case22 in the design document).  When a recursive eval re-enters the
# same (scope, node) key the cache returns this object and the outer call
# treats it as a soft-fail (None).
_RECURSING = object()


class ConstantTable:
    """Pre-scanned facts about static integer / list constants in a code base.

    The table is built once via :meth:`build_all` and is read-only thereafter.
    All 14 tables documented in section 3.1 of ``ast_evaluator_redesign.md``
    are exposed as plain ``Dict`` attributes; ``ConstantResolver`` consults
    them through narrow accessor helpers.

    The 14 tables:
        1.  ``file_int_consts``        — file-level ``NAME = <int>``
        2.  ``file_str_list_consts``   — file-level ``NAME = [<str>, ...]``
        3.  ``file_dict_literals``     — file-level ``NAME = {<str>: ...}``
        4.  ``global_int_consts``      — cross-file unique-int names
        5.  ``class_init_params``      — ``__init__`` formal parameter lists
        6.  ``dataclass_defaults``     — ``@dataclass`` field defaults
        7.  ``ctor_kw_args``           — class-level kwargs (int values)
        8.  ``ctor_kw_list_args``      — class-level kwargs (list-length)
        9.  ``instance_kw_int``        — per-instance kwargs (int values)
        10. ``instance_kw_list_len``   — per-instance kwargs (list-length)
        11. ``instance_const_chain``   — per-instance dataclass field chains
        12. ``local_aliases``          — method-level ``var = <expr>`` AST
        13. ``local_dataclass_inst``   — method-level dataclass instances
        14. ``self_to_param``          — ``self.attr`` → formal-param map
    """

    def __init__(self):
        # ── File-level (table 1-3) ─────────────────────────────────────
        self.file_int_consts: Dict[str, Dict[str, IntValue]] = {}
        self.file_str_list_consts: Dict[str, Dict[str, ListValue]] = {}
        self.file_dict_literals: Dict[str, Dict[str, Dict[str, ast.expr]]] = {}

        # ── Global unique (table 4) ────────────────────────────────────
        self.global_int_consts: Dict[str, IntValue] = {}

        # ── Class-level (tables 5-8) ───────────────────────────────────
        # class_init_params[(file, cls)] -> ['self', 'config', ...]
        self.class_init_params: Dict[Tuple[str, str], List[str]] = {}
        # dataclass_defaults[(file, cls)] -> {field: IntValue}
        self.dataclass_defaults: Dict[Tuple[str, str], Dict[str, IntValue]] = {}
        # ctor_kw_args[ClassName] -> {kw: {IntValue, ...}}
        self.ctor_kw_args: Dict[str, Dict[str, Set[IntValue]]] = {}
        # ctor_kw_list_args[ClassName] -> {kw: {list_len, ...}}
        self.ctor_kw_list_args: Dict[str, Dict[str, Set[int]]] = {}

        # ── Per-instance (tables 9-11) ─────────────────────────────────
        self.instance_kw_int: Dict[Tuple[str, str], Dict[str, IntValue]] = {}
        self.instance_kw_list_len: Dict[Tuple[str, str], Dict[str, int]] = {}
        # Auxiliary: retain per-instance concrete list literals so callers can
        # recover the real item sequence (for example ``['x', 'y']`` vs
        # ``['x', 'y', 'z']``) instead of only the aggregated class-level max len.
        self.instance_kw_list_items: Dict[Tuple[str, str], Dict[str, ListValue]] = {}
        # instance_const_chain[(parent_cls, parent_attr)] ->
        #   {param_name: {field: IntValue}}
        self.instance_const_chain: Dict[
            Tuple[str, str], Dict[str, Dict[str, IntValue]]
        ] = {}

        # ── Method-level (tables 12-13) ────────────────────────────────
        # local_aliases[(file, cls, method)] -> {var_name: ast.expr}
        self.local_aliases: Dict[Tuple[str, str, str], Dict[str, ast.expr]] = {}
        # local_dataclass_inst[(file, cls, method)] -> {var: {field: IntValue}}
        self.local_dataclass_inst: Dict[
            Tuple[str, str, str], Dict[str, Dict[str, IntValue]]
        ] = {}

        # ── self.attr → param map (table 14) ───────────────────────────
        self.self_to_param: Dict[Tuple[str, str], Dict[str, str]] = {}
        # local_self_attr_exprs[(file, cls, method)] -> {attr: [(line, expr), ...]}
        # Used by P2 to resolve list_len through ``self.attr`` indirection, e.g.
        # ``self.layers = [..]`` / ``self.layers = [.. for _ in range(n)]`` /
        # ``self.names = names``.
        self.local_self_attr_exprs: Dict[
            Tuple[str, str, str], Dict[str, List[Tuple[int, ast.expr]]]
        ] = {}

        # ── Phase E1.2 auxiliary: per-class ClassDef AST node (used for
        #    cross-class dict-literal lookups during Pass 4 dataclass
        #    propagation).
        self.class_defs: Dict[Tuple[str, str], ast.ClassDef] = {}

        # ── Annotation fallback is forbidden: when a parameter cannot be
        #    resolved from concrete caller/default semantics, resolvers must
        #    conservatively return ``None``. Type annotations are metadata only;
        #    they must never be used to infer runtime values.
        # ── Auxiliary (referenced in section 4.6) ──────────────────────
        # local_self_dict_literals[(file, cls, method)] ->
        #   {self_attr_name: {key: ast.expr}}
        # Stays empty in PR1; PR5 will populate it.
        self.local_self_dict_literals: Dict[
            Tuple[str, str, str], Dict[str, Dict[str, ast.expr]]
        ] = {}

        # ── Pass-1-private aggregator: per-file int-name → {value, ...}
        # Used by Pass 2 to converge to a globally-unique int.  Not part of
        # the 14 public tables.
        self._global_int_candidates: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------
    # Public build entry point
    # ------------------------------------------------------------------
    @classmethod
    def build_all(cls,
                  source_files: Dict[str, List[str]],
                  ast_frontends: Dict[str, "ASTFrontend"],
                  nn_module_classes: Set[str]) -> "ConstantTable":
        """Run all 4 passes and return a populated table.

        Parameters
        ----------
        source_files : dict[str, list[str]]
            ``{fname: source_lines}``.  Currently used only for parity with
            the design doc; PR1 reads exclusively from ``ast_frontends``.
        ast_frontends : dict[str, ASTFrontend]
            One entry per ``fname``.  Frontends MUST already be parsed.
        nn_module_classes : set[str]
            Names of classes derived (transitively) from ``nn.Module``.
        """
        table = cls()
        table._ast_frontends = dict(ast_frontends)
        # Pass 1 — purely file-level scans.
        for fname, fe in ast_frontends.items():
            table._pass1_file_consts(fname, fe)
        # Pass 2 — cross-file global-int convergence (depends on Pass 1).
        table._pass2_global_converge()
        # Pass 3 — class-level chains (depends on Pass 1 + 2).
        for fname, fe in ast_frontends.items():
            table._pass3_class_chains(fname, fe, nn_module_classes)
        # Pass 4 — call-site formal-parameter binding (depends on Pass 3).
        for fname, fe in ast_frontends.items():
            table._pass4_callsite_bind(fname, fe, nn_module_classes)
        return table

    # ------------------------------------------------------------------
    # Pass 1 — file-level constants, dataclass defaults
    # ------------------------------------------------------------------
    def _pass1_file_consts(self, fname: str, fe: "ASTFrontend") -> None:
        """Scan the top-level ``ast.Module`` body of *fname*.

        Populates:
          * ``file_int_consts``
          * ``file_str_list_consts``
          * ``file_dict_literals``
          * ``dataclass_defaults`` (for ``@dataclass`` decorated classes)
          * ``_global_int_candidates`` (consumed by Pass 2)
        """
        for node in fe.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    self._record_file_const(fname, tgt.id, node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self._record_file_const(fname, node.target.id, node.value)
            elif isinstance(node, ast.ClassDef):
                if self._is_dataclass(node):
                    self._scan_dataclass_defaults(fname, node)

    def _record_file_const(self, fname: str, name: str, value_node: ast.expr) -> None:
        # int literal
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, int) and not isinstance(value_node.value, bool):
            iv = IntValue(value_node.value, "file_const")
            self.file_int_consts.setdefault(fname, {})[name] = iv
            self._global_int_candidates.setdefault(name, set()).add(value_node.value)
            return
        # negative int literal (UnaryOp(USub, Constant(int)))
        if (isinstance(value_node, ast.UnaryOp)
                and isinstance(value_node.op, ast.USub)
                and isinstance(value_node.operand, ast.Constant)
                and isinstance(value_node.operand.value, int)
                and not isinstance(value_node.operand.value, bool)):
            v = -value_node.operand.value
            iv = IntValue(v, "file_const")
            self.file_int_consts.setdefault(fname, {})[name] = iv
            self._global_int_candidates.setdefault(name, set()).add(v)
            return
        # list of str literals
        if isinstance(value_node, ast.List) and value_node.elts and all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            for elt in value_node.elts
        ):
            self.file_str_list_consts.setdefault(fname, {})[name] = ListValue(
                length=len(value_node.elts),
                items=tuple(value_node.elts),
                origin="file_str_list",
            )
            return
        # dict literal with all-string keys (values stay as AST nodes — the
        # resolver expands them on demand via ``eval_int`` / ``eval_list_len``)
        if isinstance(value_node, ast.Dict):
            d = {}
            ok = True
            for k, v in zip(value_node.keys, value_node.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    ok = False
                    break
                d[k.value] = v
            if ok and d:
                self.file_dict_literals.setdefault(fname, {})[name] = d

    def _is_dataclass(self, classdef_node: ast.ClassDef) -> bool:
        for dec in classdef_node.decorator_list:
            # @dataclass
            if isinstance(dec, ast.Name) and dec.id == "dataclass":
                return True
            # @dataclasses.dataclass
            if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
                return True
            # @dataclass(...) / @dataclasses.dataclass(...)
            if isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Name) and func.id == "dataclass":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "dataclass":
                    return True
        return False

    def _scan_dataclass_defaults(self, fname: str, classdef_node: ast.ClassDef) -> None:
        """For ``@dataclass class Foo: x: int = 5`` record ``x → IntValue(5)``.

        Only int defaults are captured in PR1 (sections 3.1 + 4.5); PR5
        extends this to recursive dataclass-instance defaults.
        """
        defaults: Dict[str, IntValue] = {}
        for stmt in classdef_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                v = stmt.value
                if isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool):
                    defaults[stmt.target.id] = IntValue(v.value, "dataclass_field")
                elif (isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub)
                      and isinstance(v.operand, ast.Constant)
                      and isinstance(v.operand.value, int)
                      and not isinstance(v.operand.value, bool)):
                    defaults[stmt.target.id] = IntValue(-v.operand.value, "dataclass_field")
        if defaults:
            self.dataclass_defaults[(fname, classdef_node.name)] = defaults

    # ------------------------------------------------------------------
    # Pass 2 — cross-file global-int convergence
    # ------------------------------------------------------------------
    def _pass2_global_converge(self) -> None:
        """Promote names whose value is identical across all defining files.

        A name appears in ``global_int_consts`` only when **every** file
        that defines it gives the *same* int.  This matches the existing
        ``global_int_const_values`` semantics in the legacy evaluator.
        """
        for name, values in self._global_int_candidates.items():
            if len(values) == 1:
                v = next(iter(values))
                self.global_int_consts[name] = IntValue(v, "global_const")

    # ------------------------------------------------------------------
    # Pass 3 — class-level chains
    # ------------------------------------------------------------------
    def _pass3_class_chains(self, fname: str, fe: "ASTFrontend",
                            nn_module_classes: Set[str]) -> None:
        """Scan every class in *fname* that participates in DAG construction.

        For each qualifying class records:
          * ``class_init_params`` (formal parameter names)
          * ``self_to_param``     (``self.X = X`` aliases)
          * ``local_aliases``     (``var = <expr>`` inside methods)
          * ``ctor_kw_args``      (kwargs passed to the class as a ctor)
          * ``ctor_kw_list_args`` (kwargs whose value is a list literal)
        """
        for cname, info in fe.class_registry.items():
            classdef = info.get("node")
            if classdef is None:
                continue
            qualifies = (cname in nn_module_classes
                         or (fname, cname) in self.dataclass_defaults)
            if not qualifies:
                continue
            # Phase E1.2: stash the ClassDef AST node for cross-class lookup.
            self.class_defs[(fname, cname)] = classdef
            self._scan_class_init_params(fname, cname, classdef)
            self._scan_self_to_param(fname, cname, fe)
            self._scan_self_attr_exprs(fname, cname, classdef)
            self._scan_local_aliases(fname, cname, classdef)
            self._scan_self_dict_literals(fname, cname, classdef)
            self._scan_local_dataclass_instances(fname, cname, classdef)
            self._scan_ctor_call_sites(fname, cname, classdef)

    def _scan_class_init_params(self, fname: str, cname: str,
                                classdef: ast.ClassDef) -> None:
        for stmt in classdef.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                params = []
                # Note: stmt.args.args includes `self`; we keep it for index
                # symmetry with the positional binding pass.
                for a in stmt.args.args:
                    params.append(a.arg)
                # Skip *args / **kwargs (intentionally — section 8.1 R1).
                self.class_init_params[(fname, cname)] = params
                return

    def _scan_self_to_param(self, fname: str, cname: str,
                            fe: "ASTFrontend") -> None:
        """Use ASTFrontend's existing ``get_self_param_aliases`` helper."""
        try:
            aliases = fe.get_self_param_aliases(cname, "__init__")
        except Exception:
            aliases = {}
        if aliases:
            self.self_to_param[(fname, cname)] = dict(aliases)

    def _scan_self_attr_exprs(self, fname: str, cname: str,
                              classdef: ast.ClassDef) -> None:
        """Record ``self.attr = <expr>`` bindings with source-order information.

        P2 needs list_len resolution through ``self.attr`` indirection.  We keep
        the original RHS AST node so ``ConstantResolver`` can recurse into it
        later (for example ``self.layers = [.. for _ in range(n)]`` or
        ``self.names = names``).
        """
        for stmt in classdef.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope_key = (fname, cname, stmt.name)
            attr_exprs: Dict[str, List[Tuple[int, ast.expr]]] = {}
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Assign):
                    continue
                if len(sub.targets) != 1:
                    continue
                attr_name = self._extract_self_attr_name(sub.targets[0])
                if attr_name is None:
                    continue
                attr_exprs.setdefault(attr_name, []).append((getattr(sub, "lineno", -1), sub.value))
            if attr_exprs:
                for entries in attr_exprs.values():
                    entries.sort(key=lambda item: item[0])
                self.local_self_attr_exprs[scope_key] = attr_exprs

    def _scan_local_aliases(self, fname: str, cname: str,
                            classdef: ast.ClassDef) -> None:
        """Record ``var = <expr>`` for every method in the class.

        We keep the **right-hand-side AST node** (not its evaluated value)
        so the resolver can recurse into it and benefit from memoization.
        """
        for stmt in classdef.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            mname = stmt.name
            scope_key = (fname, cname, mname)
            method_aliases: Dict[str, ast.expr] = {}
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            method_aliases[tgt.id] = sub.value
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) and sub.value is not None:
                    method_aliases[sub.target.id] = sub.value
            if method_aliases:
                self.local_aliases[scope_key] = method_aliases

    def _scan_self_dict_literals(self, fname: str, cname: str,
                                 classdef: ast.ClassDef) -> None:
        """Record ``self.attr = {<str>: <expr>}`` literals inside methods."""
        for stmt in classdef.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            mname = stmt.name
            scope_key = (fname, cname, mname)
            dicts: Dict[str, Dict[str, ast.expr]] = {}
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Assign) or len(sub.targets) != 1:
                    continue
                tgt = sub.targets[0]
                if not (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    continue
                if not isinstance(sub.value, ast.Dict):
                    continue
                entries: Dict[str, ast.expr] = {}
                ok = True
                for k_node, v_node in zip(sub.value.keys, sub.value.values):
                    if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                        ok = False
                        break
                    entries[k_node.value] = v_node
                if ok and entries:
                    dicts[tgt.attr] = entries
            if dicts:
                self.local_self_dict_literals[scope_key] = dicts

    def _scan_local_dataclass_instances(self, fname: str, cname: str,
                                        classdef: ast.ClassDef) -> None:
        """Materialize local ``var = DataclassCtor(...)`` bindings per method."""
        resolver = ConstantResolver(self)
        for stmt in classdef.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            mname = stmt.name
            scope = Scope(file=fname, cls=cname, method=mname)
            scope_key = (fname, cname, mname)
            out: Dict[str, Dict[str, IntValue]] = {}
            for sub in ast.walk(stmt):
                target = None
                value = None
                if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                    target = sub.targets[0]
                    value = sub.value
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) and sub.value is not None:
                    target = sub.target
                    value = sub.value
                if target is None or value is None:
                    continue
                fields = resolver.eval_dataclass_fields(value, scope)
                if fields:
                    out[target.id] = fields
            if out:
                self.local_dataclass_inst[scope_key] = out

    def _scan_ctor_call_sites(self, fname: str, cname: str,
                              classdef: ast.ClassDef) -> None:
        """For ``self.x = SubClass(kw=<int>, kw2=[a,b,c])`` aggregate kwargs
        into the *class-level* ``ctor_kw_args`` / ``ctor_kw_list_args``.

        Per-instance specialisation happens in Pass 4 (``_pass4_callsite_bind``).
        Both tables retain *all* observed values across instances (Set union)
        so the resolver can detect non-unique results and conservatively fail.
        """
        for stmt in ast.walk(classdef):
            if not isinstance(stmt, ast.Assign):
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            sub_cls = self._call_class_name(value.func)
            if sub_cls is None:
                continue
            for kw in value.keywords:
                if kw.arg is None:
                    continue  # **kwargs unpack — skip; PR1 doesn't resolve
                kw_val = kw.value
                # int literal
                if isinstance(kw_val, ast.Constant) and isinstance(kw_val.value, int) and not isinstance(kw_val.value, bool):
                    iv = IntValue(kw_val.value, "kw_arg")
                    self.ctor_kw_args.setdefault(sub_cls, {}).setdefault(kw.arg, set()).add(iv)
                elif (isinstance(kw_val, ast.UnaryOp) and isinstance(kw_val.op, ast.USub)
                      and isinstance(kw_val.operand, ast.Constant)
                      and isinstance(kw_val.operand.value, int)
                      and not isinstance(kw_val.operand.value, bool)):
                    iv = IntValue(-kw_val.operand.value, "kw_arg")
                    self.ctor_kw_args.setdefault(sub_cls, {}).setdefault(kw.arg, set()).add(iv)
                # list literal — record length only
                elif isinstance(kw_val, ast.List):
                    self.ctor_kw_list_args.setdefault(sub_cls, {}).setdefault(kw.arg, set()).add(len(kw_val.elts))

    # ------------------------------------------------------------------
    # Pass 4 — call-site formal-parameter binding
    # ------------------------------------------------------------------
    def _pass4_callsite_bind(self, fname: str, fe: "ASTFrontend",
                             nn_module_classes: Set[str]) -> None:
        """Cross-call-site positional / keyword argument binding.

        For every ``self.x = SubClass(arg0, arg1, kw=...)`` in *cname*:

          1. Look up ``SubClass.__init__`` formal parameter list.
          2. ``positional bind`` ``args[i] → params[i]``.
          3. ``kwargs bind``    ``kw   → param_name``.
          4. If the bound value resolves to a known dataclass instance
             (via ``local_dataclass_inst`` populated lazily here), write
             ``instance_const_chain[(cname, attr)][param] = field_dict``.
          5. If the bound value is an int literal: write to
             ``instance_kw_int``.
          6. If the bound value is a list literal: write list length to
             ``instance_kw_list_len``.

        PR1 implements the basic int / list-literal arms (steps 5-6) so the
        skeleton is callable end-to-end.  Phase E1.2 wires steps 1-4
        (dataclass propagation across call sites — section 4.6).
        """
        for class_name, info in fe.class_registry.items():
            classdef = info.get("node")
            if classdef is None or class_name not in nn_module_classes:
                continue
            # Phase E1.2: collect per-class map of ``self.<dict_field> = {<lit>}``
            # from the already-populated ``local_self_dict_literals`` cache
            # (populated by _scan_self_dict_literals during _pass3_class_chains).
            # Merge all methods of this class into a flat {attr_name: {str: expr}} map.
            self_dict_fields: Dict[str, Dict[str, ast.expr]] = {}
            for (sf, sc, _sm), method_dicts in self.local_self_dict_literals.items():
                if sf == fname and sc == class_name:
                    for attr, entries in method_dicts.items():
                        self_dict_fields.setdefault(attr, {}).update(entries)
            for stmt in ast.walk(classdef):
                if not isinstance(stmt, ast.Assign):
                    continue
                target = stmt.targets[0] if len(stmt.targets) == 1 else None
                attr_name = self._extract_self_attr_name(target) if target is not None else None
                if attr_name is None:
                    continue
                value = stmt.value
                if not isinstance(value, ast.Call):
                    continue
                sub_cls = self._call_class_name(value.func)
                if sub_cls is None:
                    continue
                # --- Step 1-6: positional/keyword binding + per-instance
                # literal/dataclass propagation.
                # Locate ``SubClass.__init__`` formal parameter list across
                # any defining file (the dataclass / subclass may live in a
                # different file than the parent class).
                params = self._lookup_init_params(sub_cls)
                if not params:
                    continue
                # Skip ``self`` slot for binding.
                bind_params = params[1:] if params and params[0] == "self" else params
                bound: Dict[str, ast.expr] = {}
                for i, arg_node in enumerate(value.args):
                    if i < len(bind_params):
                        bound[bind_params[i]] = arg_node
                for kw in value.keywords:
                    if kw.arg is not None:
                        bound[kw.arg] = kw.value
                # Special-case ``SubClass(**self.<dict_field>)`` —
                # treat the dict literal as an inline kwargs spread.
                for kw in value.keywords:
                    if kw.arg is None:
                        # ``**something``: only handle ``**self.<attr>``
                        # whose RHS is a dict literal previously assigned
                        # to ``self.<attr>``.
                        kv = kw.value
                        if (isinstance(kv, ast.Attribute)
                                and isinstance(kv.value, ast.Name)
                                and kv.value.id == "self"):
                            spread = self_dict_fields.get(kv.attr)
                            if spread:
                                for sk, sv in spread.items():
                                    bound.setdefault(sk, sv)

                if not bound:
                    continue
                scope_key = (fname, class_name, "__init__")
                scope = Scope(file=fname, cls=class_name, method="__init__")
                inst_key = (class_name, attr_name)
                for param, expr_node in bound.items():
                    iv = self._resolve_bound_int(expr_node, scope)
                    if iv is not None:
                        self.instance_kw_int.setdefault(inst_key, {})[param] = iv
                    list_value = self._resolve_bound_list_value(expr_node, scope_key)
                    if list_value is not None:
                        self.instance_kw_list_len.setdefault(inst_key, {})[param] = list_value.length
                        self.instance_kw_list_items.setdefault(inst_key, {})[param] = list_value
                    fields = self._resolve_bound_dataclass_fields(expr_node, scope_key)
                    if not fields:
                        continue
                    slot = self.instance_const_chain.setdefault(inst_key, {})
                    existing = slot.get(param)
                    if existing is None:
                        slot[param] = fields
                    else:
                        merged = dict(existing)
                        for fk, fv in fields.items():
                            merged.setdefault(fk, fv)
                        slot[param] = merged
    # ------------------------------------------------------------------
    # Helpers (used by Pass 3 & 4)
    # ------------------------------------------------------------------
    def _lookup_init_params(self, sub_cls: str) -> List[str]:
        for (_fname, _cname), params in self.class_init_params.items():
            if _cname == sub_cls:
                return params
        return []

    def _resolve_bound_int(self,
                           expr_node: ast.expr,
                           scope: Scope) -> Optional[IntValue]:
        resolver = ConstantResolver(self)
        iv = resolver.eval_int(expr_node, scope)
        if iv is None:
            return None
        return IntValue(iv.value, "callsite_bind")

    def _resolve_bound_list_value(self,
                                  expr_node: ast.expr,
                                  scope_key: Tuple[str, str, str]) -> Optional[ListValue]:
        fname, cname, mname = scope_key
        resolver = ConstantResolver(self)
        list_node = resolver._resolve_to_list_literal(
            expr_node,
            Scope(file=fname, cls=cname, method=mname),
        )
        if list_node is None:
            return None
        return ListValue(
            length=len(list_node.elts),
            items=tuple(list_node.elts),
            origin="callsite_bind",
        )

    def _resolve_bound_dataclass_fields(self,
                                        expr_node: ast.expr,
                                        scope_key: Tuple[str, str, str]) -> Optional[Dict[str, IntValue]]:
        fields = self._resolve_dataclass_call_fields(expr_node, scope_key)
        if fields:
            return fields
        if not isinstance(expr_node, ast.Name):
            return None
        local_inst = self.local_dataclass_inst.get(scope_key, {})
        if expr_node.id in local_inst:
            return dict(local_inst[expr_node.id])
        resolver = ConstantResolver(self)
        target_node = resolver._resolve_alias_chain(expr_node, scope_key)
        if isinstance(target_node, ast.Name) and target_node.id in local_inst:
            return dict(local_inst[target_node.id])
        fields = self._resolve_dataclass_call_fields(target_node, scope_key) if target_node is not None else None
        if fields:
            return fields
        return None

    def _resolve_dataclass_call_fields(self,
                                       expr_node: ast.expr,
                                       scope_key: Tuple[str, str, str]) -> Optional[Dict[str, IntValue]]:
        if not isinstance(expr_node, ast.Call):
            return None
        fname, cname, mname = scope_key
        resolver = ConstantResolver(self)
        return resolver.eval_dataclass_fields(
            expr_node,
            Scope(file=fname, cls=cname, method=mname),
        )

    @staticmethod
    def _call_class_name(func_node: ast.expr) -> Optional[str]:
        """Best-effort extract of the called class name from ``Call.func``.

        Handles plain ``Cls(...)`` and ``mod.Cls(...)`` / ``a.b.Cls(...)``;
        returns ``None`` for ``Cls(...).method(...)`` chains and other
        non-constructor expressions.
        """
        if isinstance(func_node, ast.Name):
            # Heuristic: ``ClassName`` style (PEP 8) — caller decides.
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    @staticmethod
    def _extract_self_attr_name(target: ast.expr) -> Optional[str]:
        """For ``self.X = ...`` return ``"X"``; otherwise ``None``."""
        if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            return target.attr
        return None


class ConstantResolver:
    """Pure AST-driven constant evaluator.

    The resolver is **read-only**; it consults :class:`ConstantTable` but
    never mutates it.  Failure semantics: every public ``eval_*`` returns
    ``None`` when evaluation cannot be completed — *no regex fallback*.

    Dispatch covers the 9 AST node types listed in section 4.2:

        ============  =====================================
        Node type     ``int`` evaluation path
        ============  =====================================
        Constant      literal int, possibly inside a list
        Name          local alias / file const / global const
        Attribute     arbitrary-depth ``self.<a>.<b>.<c>...``
        Subscript     ``LIST[i]`` or ``self.x[i]`` (literal i)
        BinOp         ``+ - * //`` arithmetic over ints
        UnaryOp       unary ``-`` over ints
        Call          ``len(...)`` / ``int(...)`` / dataclass()
        List          (``list_len`` only) literal list length
        Starred       (``list_len`` only) inner list length
        ============  =====================================

    Anything else returns ``None``.
    """

    # Conservative recursion depth limit (section 4.3).
    _MAX_ATTR_CHAIN_DEPTH = 8
    MAX_ALIAS_DEPTH = 8

    def __init__(self, table: ConstantTable, runtime_overrides: Optional[Dict[str, int]] = None,
                 diagnostics: Optional[List[str]] = None):
        self.table = table
        self.runtime_overrides = runtime_overrides or {}
        self.diagnostics = diagnostics if diagnostics is not None else []
        self._diagnostic_keys: Set[str] = set()
        # Cache key: (file, cls, method, parent_cls or "", parent_attr or "",
        # id(node), kind).  Using ``id(node)`` skips the O(n) ``ast.unparse``
        # cost while remaining stable for the lifetime of the table (section
        # 3.3, R2).
        self._eval_cache: Dict[Tuple[Any, ...], Any] = {}
        self.fail_reasons: Counter[str] = Counter()
        self.fail_samples: Dict[str, List[str]] = defaultdict(list)

    def _fail(self, reason: str, node: Optional[ast.AST] = None, scope: Optional[Scope] = None) -> None:
        allowed = {
            "name_unresolved",
            "param_unbound",
            "call_not_constant",
            "list_len_unsupported_expr",
            "alias_unresolved",
            "annotation_not_value",
        }
        if reason not in allowed:
            reason = "list_len_unsupported_expr"
        self.fail_reasons[reason] += 1
        samples = self.fail_samples[reason]
        if len(samples) < 5:
            node_desc = type(node).__name__ if node is not None else "<none>"
            scope_desc = ""
            if scope is not None:
                scope_desc = f"{scope.file}:{scope.cls}.{scope.method}:{scope.parent_cls}.{scope.parent_attr}"
            samples.append(f"{node_desc}@{scope_desc}")
        return None

    def get_fail_reason_counts(self) -> Dict[str, int]:
        return dict(self.fail_reasons)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, node: ast.expr, scope: Scope, kind: str = "int") -> Any:
        """Generic entry point used by the integration tests in PR1.

        ``kind`` ∈ ``{"int", "list_len", "fields"}``; returns ``None`` on
        any failure.  Most callers should prefer the typed wrappers below.
        """
        if kind == "int":
            return self.eval_int(node, scope)
        if kind == "list_len":
            return self.eval_list_len(node, scope)
        if kind == "fields":
            return self.eval_dataclass_fields(node, scope)
        return None

    def eval_int(self, node: ast.expr, scope: Scope) -> Optional[IntValue]:
        return self._cached(node, scope, "int", self._eval_int_uncached)

    def eval_list_len(self, node: ast.expr, scope: Scope) -> Optional[int]:
        return self._cached(node, scope, "list_len", self._eval_list_len_uncached)

    def eval_bool(self, node: ast.expr, scope: Scope) -> Optional[bool]:
        return self._cached(node, scope, "bool", self._eval_bool_uncached)

    def eval_dataclass_fields(self, node: ast.expr, scope: Scope) -> Optional[Dict[str, IntValue]]:
        return self._cached(node, scope, "fields", self._eval_fields_uncached)

    def _eval_bool_uncached(self, node: ast.expr, scope: Scope) -> Optional[bool]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return node.value
            return None
        if isinstance(node, ast.Name):
            method_aliases = self.table.local_aliases.get(scope.method_key)
            if method_aliases and node.id in method_aliases:
                rhs = method_aliases[node.id]
                return self._eval_bool_uncached(rhs, scope)
            default_val = self._lookup_param_default(node.id, scope)
            if default_val is not None:
                return self._eval_bool_uncached(default_val, scope)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sub = self.eval_bool(node.operand, scope)
            return (not sub) if sub is not None else None
        if isinstance(node, ast.BoolOp):
            vals = [self.eval_bool(v, scope) for v in node.values]
            if any(v is None for v in vals):
                return None
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
        return None

    def _lookup_param_default(self, param_name: str, scope: Scope) -> Optional[ast.expr]:
        method_node = self._get_method_node_for_scope(scope)
        if method_node is None:
            return None
        args = method_node.args
        n_defaults = len(args.defaults)
        n_args = len(args.args)
        for i, arg in enumerate(args.args):
            if arg.arg != param_name:
                continue
            default_offset = i - (n_args - n_defaults)
            if default_offset >= 0:
                return args.defaults[default_offset]
            return None
        return None

    def _get_method_node_for_scope(self, scope: Scope):
        if scope.file is None or scope.cls is None or scope.method is None:
            return None
        frontends = getattr(self.table, "_ast_frontends", {}) or {}
        fe = frontends.get(scope.file)
        if fe is None:
            return None
        return fe._get_method_node(scope.cls, scope.method)

    # ------------------------------------------------------------------
    # Cache + recursion guard
    # ------------------------------------------------------------------
    def _cached(self, node, scope, kind, fn):
        key = (
            scope.file,
            scope.cls or "",
            scope.method or "",
            scope.parent_cls or "",
            scope.parent_attr or "",
            id(node),
            kind,
        )
        if key in self._eval_cache:
            cached = self._eval_cache[key]
            if cached is _RECURSING:
                # Cycle detected — fail soft (case22).
                return None
            return cached
        # Insert sentinel so a recursive re-entry on the same node sees it.
        self._eval_cache[key] = _RECURSING
        try:
            result = fn(node, scope)
        except Exception:
            result = None
        # Replace sentinel with actual result.
        self._eval_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # int dispatcher
    # ------------------------------------------------------------------
    def _eval_int_uncached(self, node: ast.expr, scope: Scope) -> Optional[IntValue]:
        if isinstance(node, ast.Constant):
            return self._visit_Constant_int(node, scope)
        if isinstance(node, ast.Name):
            return self._visit_Name_int(node, scope)
        if isinstance(node, ast.Attribute):
            return self._visit_Attribute_int(node, scope)
        if isinstance(node, ast.Subscript):
            return self._visit_Subscript_int(node, scope)
        if isinstance(node, ast.BinOp):
            return self._visit_BinOp_int(node, scope)
        if isinstance(node, ast.UnaryOp):
            return self._visit_UnaryOp_int(node, scope)
        if isinstance(node, ast.Call):
            return self._visit_Call_int(node, scope)
        return self._fail("list_len_unsupported_expr", node, scope)

    def _visit_Constant_int(self, node: ast.Constant, scope: Scope) -> Optional[IntValue]:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return IntValue(node.value, "literal")
        return self._fail("list_len_unsupported_expr", node, scope)

    def _method_param_has_annotation(self, param_name: str, scope: Scope) -> bool:
        method_node = self._get_method_node_for_scope(scope)
        if method_node is None:
            return False
        for arg in method_node.args.args:
            if arg.arg == param_name:
                return arg.annotation is not None
        return False

    def _visit_Name_int(self, node: ast.Name, scope: Scope) -> Optional[IntValue]:
        # ① Method-local alias chain.
        resolved_alias = self._resolve_alias_chain(node, scope.method_key)
        if resolved_alias is None:
            return self._fail("alias_unresolved", node, scope)
        if resolved_alias is not node:
            sub = self.eval_int(resolved_alias, scope)
            if sub is not None:
                return IntValue(sub.value, "alias")
            return self._fail("alias_unresolved", node, scope)
        # ② File-level constant.
        file_consts = self.table.file_int_consts.get(scope.file, {})
        if node.id in file_consts:
            return file_consts[node.id]
        # ③ Global unique constant.
        if node.id in self.table.global_int_consts:
            return self.table.global_int_consts[node.id]
        # ④ Class formal parameter — try per-instance first, then class-level
        #    union (only when unique).
        if scope.cls is not None:
            params = self.table.class_init_params.get((scope.file, scope.cls), [])
            if node.id in params:
                # Per-instance override via instance_kw_int.
                inst_key = scope.instance_key
                if inst_key[0] is not None and inst_key[1] is not None:
                    inst_int = self.table.instance_kw_int.get(inst_key, {})
                    if node.id in inst_int:
                        return inst_int[node.id]
                # Fallback to class-level ctor_kw_args union when unique.
                kw_set = self.table.ctor_kw_args.get(scope.cls, {}).get(node.id)
                if kw_set and len(kw_set) == 1:
                    return next(iter(kw_set))
                if self._method_param_has_annotation(node.id, scope):
                    return self._fail("annotation_not_value", node, scope)
                return self._fail("param_unbound", node, scope)
        return self._fail("name_unresolved", node, scope)

    def _visit_Attribute_int(self, node: ast.Attribute, scope: Scope) -> Optional[IntValue]:
        chain = self._flatten_attr_chain(node)
        if chain is None:
            return None
        base, attrs = chain
        if len(attrs) > self._MAX_ATTR_CHAIN_DEPTH:
            return None
        if isinstance(base, ast.Name) and base.id == "self":
            return self._resolve_self_chain(attrs, scope)
        return None

    def _visit_Subscript_int(self, node: ast.Subscript, scope: Scope) -> Optional[IntValue]:
        # Only literal int indices into a known list literal are supported.
        # Section 4.2 explicitly allows: LIST_NAME[i] / self.x[i] when the
        # base is ast.List / file_str_list_consts / local_aliases→list.
        idx_node = node.slice
        # Python 3.8 wraps subscripts in ast.Index.
        if hasattr(ast, "Index") and isinstance(idx_node, ast.Index):
            idx_node = idx_node.value  # type: ignore[attr-defined]
        if not (isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int)):
            return None
        idx = idx_node.value
        # Resolve the base to an ast.List literal.
        base_list = self._resolve_to_list_literal(node.value, scope)
        if base_list is None:
            return None
        if idx < 0:
            idx += len(base_list.elts)
        if not (0 <= idx < len(base_list.elts)):
            return None
        return self.eval_int(base_list.elts[idx], scope)

    def _visit_BinOp_int(self, node: ast.BinOp, scope: Scope) -> Optional[IntValue]:
        l = self.eval_int(node.left, scope)
        r = self.eval_int(node.right, scope)
        if l is None or r is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return IntValue(l.value + r.value, "binop")
            if isinstance(node.op, ast.Sub):
                return IntValue(l.value - r.value, "binop")
            if isinstance(node.op, ast.Mult):
                return IntValue(l.value * r.value, "binop")
            if isinstance(node.op, ast.FloorDiv):
                if r.value == 0:
                    return None
                return IntValue(l.value // r.value, "binop")
        except Exception:
            return None
        return None

    def _visit_UnaryOp_int(self, node: ast.UnaryOp, scope: Scope) -> Optional[IntValue]:
        if isinstance(node.op, ast.USub):
            sub = self.eval_int(node.operand, scope)
            if sub is None:
                return None
            return IntValue(-sub.value, "unaryop")
        if isinstance(node.op, ast.UAdd):
            return self.eval_int(node.operand, scope)
        return None

    def _visit_Call_int(self, node: ast.Call, scope: Scope) -> Optional[IntValue]:
        # ``len([...])`` — degenerates into list_len.
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
            n = self.eval_list_len(node.args[0], scope)
            if n is not None:
                return IntValue(n, "len_call")
            return None
        # ``int(<int_literal_or_str>)`` — narrow cast.
        if isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                if isinstance(arg.value, int) and not isinstance(arg.value, bool):
                    return IntValue(arg.value, "int_cast")
                if isinstance(arg.value, str):
                    try:
                        return IntValue(int(arg.value), "int_cast")
                    except ValueError:
                        return None
            return self.eval_int(arg, scope)
        # Dataclass(...) — section 4.2 says return None for int kind (the
        # caller should use eval_dataclass_fields instead).
        return self._fail("call_not_constant", node, scope)

    # ------------------------------------------------------------------
    # list_len dispatcher
    # ------------------------------------------------------------------
    def _eval_list_len_uncached(self, node: ast.expr, scope: Scope) -> Optional[int]:
        if isinstance(node, ast.List):
            return self._visit_List_list_len(node, scope)
        if isinstance(node, ast.ListComp):
            return self._visit_ListComp_list_len(node, scope)
        if isinstance(node, ast.Name):
            return self._visit_Name_list_len(node, scope)
        if isinstance(node, ast.Attribute):
            return self._visit_Attribute_list_len(node, scope)
        if isinstance(node, ast.BinOp):
            return self._visit_BinOp_list_len(node, scope)
        if isinstance(node, ast.Starred):
            return self._visit_Starred_list_len(node, scope)
        if isinstance(node, ast.Call):
            return self._fail("call_not_constant", node, scope)  # only int-returning calls are handled
        return self._fail("list_len_unsupported_expr", node, scope)

    def _visit_List_list_len(self, node: ast.List, scope: Scope) -> Optional[int]:
        total = 0
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                n = self.eval_list_len(elt.value, scope)
                if n is None:
                    return None
                total += n
            else:
                total += 1
        return total

    def _visit_ListComp_list_len(self, node: ast.ListComp, scope: Scope) -> Optional[int]:
        if node.generators is None or not node.generators:
            return 0
        total = 1
        for gen in node.generators:
            if gen.ifs:
                return None
            n = self.eval_list_len(gen.iter, scope)
            if n is None:
                if (isinstance(gen.iter, ast.Call)
                        and isinstance(gen.iter.func, ast.Name)
                        and gen.iter.func.id == "range"
                        and len(gen.iter.args) == 1):
                    iv = self.eval_int(gen.iter.args[0], scope)
                    if iv is None or iv.value < 0:
                        return None
                    n = iv.value
                else:
                    return None
            total *= n
        return total

    def _visit_Name_list_len(self, node: ast.Name, scope: Scope) -> Optional[int]:
        # ① local alias → recurse on its final RHS
        resolved_alias = self._resolve_alias_chain(node, scope.method_key)
        if resolved_alias is None:
            return self._fail("alias_unresolved", node, scope)
        if resolved_alias is not node:
            n = self.eval_list_len(resolved_alias, scope)
            if n is not None:
                return n
            return self._fail("alias_unresolved", node, scope)
        # ② file-level str-list literal
        file_lists = self.table.file_str_list_consts.get(scope.file, {})
        if node.id in file_lists:
            return file_lists[node.id].length
        # ③ class-level kwarg union (only when single value). Prefer
        #    per-instance specialisation from instance_kw_list_len when the
        #    current scope is evaluating inside a child class for a specific
        #    parent instance.
        if scope.cls is not None:
            params = self.table.class_init_params.get((scope.file, scope.cls), [])
            if node.id in params:
                inst_key = scope.instance_key
                if inst_key[0] is not None and inst_key[1] is not None:
                    inst_lens = self.table.instance_kw_list_len.get(inst_key, {})
                    if node.id in inst_lens:
                        return inst_lens[node.id]
                kw_set = self.table.ctor_kw_list_args.get(scope.cls, {}).get(node.id)
                if kw_set:
                    # Phase E1.2: when multiple list lengths exist across instances,
                    # pick the maximum as a conservative upper bound (section 4.4).
                    return max(kw_set)
                if self._method_param_has_annotation(node.id, scope):
                    return self._fail("annotation_not_value", node, scope)
                return self._fail("param_unbound", node, scope)
        return self._fail("name_unresolved", node, scope)

    def _visit_Attribute_list_len(self, node: ast.Attribute, scope: Scope) -> Optional[int]:
        chain = self._flatten_attr_chain(node)
        if chain is None:
            return None
        base, attrs = chain
        if not (isinstance(base, ast.Name) and base.id == "self"):
            return None
        direct = self._lookup_self_attr_expr(attrs, scope, getattr(node, "lineno", None))
        if direct is not None:
            n = self.eval_list_len(direct, scope)
            if n is not None:
                return n
        if len(attrs) != 1:
            return self._fail("list_len_unsupported_expr", node, scope)
        cls_key = (scope.file, scope.cls)
        param = self.table.self_to_param.get(cls_key, {}).get(attrs[0])
        if param is None:
            return self._fail("name_unresolved", node, scope)
        # Per-instance overrides class-level.
        inst_key = (scope.parent_cls, scope.parent_attr)
        inst_lens = self.table.instance_kw_list_len.get(inst_key, {})
        if param in inst_lens:
            return inst_lens[param]
        kw_set = self.table.ctor_kw_list_args.get(scope.cls, {}).get(param)
        if kw_set and len(kw_set) == 1:
            return next(iter(kw_set))
        return self._fail("param_unbound", node, scope)

    def _visit_BinOp_list_len(self, node: ast.BinOp, scope: Scope) -> Optional[int]:
        if isinstance(node.op, ast.Add):
            l = self.eval_list_len(node.left, scope)
            r = self.eval_list_len(node.right, scope)
            if l is None or r is None:
                return None
            return l + r
        if isinstance(node.op, ast.Mult):
            for a, b in [(node.left, node.right), (node.right, node.left)]:
                la = self.eval_list_len(a, scope)
                ib = self.eval_int(b, scope)
                if la is not None and ib is not None and ib.value >= 0:
                    return la * ib.value
        return None

    def _visit_Starred_list_len(self, node: ast.Starred, scope: Scope) -> Optional[int]:
        return self.eval_list_len(node.value, scope)

    # ------------------------------------------------------------------
    # dataclass-fields dispatcher
    # ------------------------------------------------------------------
    def _eval_fields_uncached(self, node: ast.expr, scope: Scope) -> Optional[Dict[str, IntValue]]:
        if not isinstance(node, ast.Call):
            return None
        cls_name = ConstantTable._call_class_name(node.func)
        if cls_name is None:
            return None
        # Look up dataclass defaults across all files.
        defaults: Optional[Dict[str, IntValue]] = None
        field_order: List[str] = []
        for (fname, cname), d in self.table.dataclass_defaults.items():
            if cname == cls_name:
                defaults = dict(d)
                field_order = list(d.keys())
                break
        if defaults is None:
            return None
        fields = dict(defaults)
        # Positional args.
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred) or i >= len(field_order):
                return None  # conservative
            v = self.eval_int(arg, scope)
            if v is not None:
                fields[field_order[i]] = v
        # Keyword args / **dict spreads.
        for kw in node.keywords:
            if kw.arg is None:
                spread_items = self._resolve_to_dict_items(kw.value, scope)
                if spread_items is None:
                    return None
                for sk, sv in spread_items.items():
                    v = self.eval_int(sv, scope)
                    if v is not None:
                        fields[sk] = v
                continue
            v = self.eval_int(kw.value, scope)
            if v is not None:
                fields[kw.arg] = v
        return fields

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_alias_chain(self,
                             node: ast.expr,
                             scope_key: Tuple[str, str, str],
                             visited: Optional[Set[str]] = None,
                             depth: int = 0) -> Optional[ast.expr]:
        """Follow method-local ``Name`` aliases to their final RHS.

        Returns ``None`` on cycles or when the chain exceeds
        ``MAX_ALIAS_DEPTH``.  Non-name expressions are already concrete and are
        returned unchanged.
        """
        if not isinstance(node, ast.Name):
            return node
        if depth >= self.MAX_ALIAS_DEPTH:
            return None
        method_aliases = self.table.local_aliases.get(scope_key, {})
        if node.id not in method_aliases:
            return node
        if visited is None:
            visited = set()
        if node.id in visited:
            return None
        visited.add(node.id)
        return self._resolve_alias_chain(
            method_aliases[node.id],
            scope_key,
            visited,
            depth + 1,
        )

    def _flatten_attr_chain(self, node: ast.Attribute) -> Optional[Tuple[ast.expr, List[str]]]:
        """Convert ``a.b.c.d`` into ``(<Name 'a'>, ['b', 'c', 'd'])``.

        Returns ``None`` if the chain bottoms out at something other than a
        bare ``ast.Name`` (e.g. a function call result).
        """
        attrs: List[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Attribute):
            attrs.append(cur.attr)
            cur = cur.value
        attrs.reverse()
        if not isinstance(cur, (ast.Name, ast.Call)):
            return None
        return (cur, attrs)

    def _resolve_self_chain(self, attrs: List[str], scope: Scope) -> Optional[IntValue]:
        if scope.cls is None:
            return None
        cls_key = (scope.file, scope.cls)
        # ── Length-1 chain: self.X ────────────────────────────────────
        if len(attrs) == 1:
            param = self.table.self_to_param.get(cls_key, {}).get(attrs[0])
            if param is None:
                return None
            # Per-instance int override beats class-level union.
            inst_key = (scope.parent_cls, scope.parent_attr)
            inst_int = self.table.instance_kw_int.get(inst_key, {})
            if param in inst_int:
                return inst_int[param]
            kw_set = self.table.ctor_kw_args.get(scope.cls, {}).get(param)
            if kw_set and len(kw_set) == 1:
                return next(iter(kw_set))
            return None
        # ── Length-2+ chain: self.X.Y...Z ─────────────────────────────
        head = attrs[0]
        param = self.table.self_to_param.get(cls_key, {}).get(head)
        if param is None:
            return None
        if scope.parent_cls is None or scope.parent_attr is None:
            return None
        inst_key = (scope.parent_cls, scope.parent_attr)
        chain = self.table.instance_const_chain.get(inst_key, {}).get(param)
        if chain is None:
            override = self._lookup_runtime_override(attrs, scope)
            if override is not None:
                return IntValue(override, "runtime_override")
            self._warn_unresolved_runtime_config(attrs, scope)
            return None
        cur: Any = chain
        for key in attrs[1:]:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        if isinstance(cur, IntValue):
            return cur
        if isinstance(cur, int):
            return IntValue(cur, "dataclass_field")
        return None

    def _lookup_self_attr_expr(self,
                               attrs: List[str],
                               scope: Scope,
                               before_line: Optional[int] = None) -> Optional[ast.expr]:
        if scope.cls is None or scope.method is None or len(attrs) != 1:
            return None
        entries = self.table.local_self_attr_exprs.get(scope.method_key, {}).get(attrs[0], [])
        if not entries:
            return None
        if before_line is not None:
            for lineno, expr in reversed(entries):
                if lineno <= before_line:
                    return expr
        return entries[-1][1]

    def _runtime_override_candidates(self, attrs: List[str], scope: Scope) -> List[str]:
        joined = ".".join(attrs)
        leaf = attrs[-1] if attrs else ""
        candidates: List[str] = []
        if scope.cls and joined:
            candidates.append(f"{scope.cls}.{joined}")
        if scope.parent_cls and scope.parent_attr and joined:
            candidates.append(f"{scope.parent_cls}.{scope.parent_attr}.{joined}")
        if leaf:
            candidates.append(leaf)
        if joined and joined not in candidates:
            candidates.append(joined)
        return candidates

    def _lookup_runtime_override(self, attrs: List[str], scope: Scope) -> Optional[int]:
        for key in self._runtime_override_candidates(attrs, scope):
            if key not in self.runtime_overrides:
                continue
            value = self.runtime_overrides[key]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _warn_unresolved_runtime_config(self, attrs: List[str], scope: Scope) -> None:
        joined = ".".join(attrs)
        owner = scope.cls or "<unknown>"
        inst = f" for {scope.parent_cls}.{scope.parent_attr}" if scope.parent_cls and scope.parent_attr else ""
        diag_key = f"{owner}:{inst}:{joined}"
        if diag_key in self._diagnostic_keys:
            return
        self._diagnostic_keys.add(diag_key)
        self.diagnostics.append(
            f"Cannot statically resolve {owner}.{joined}{inst}. "
            f"Please provide runtime_overrides or explicit constructor arguments."
        )

    def _resolve_to_dict_items(self, node: ast.expr, scope: Scope) -> Optional[Dict[str, ast.expr]]:
        if isinstance(node, ast.Dict):
            out: Dict[str, ast.expr] = {}
            for k_node, v_node in zip(node.keys, node.values):
                if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                    return None
                out[k_node.value] = v_node
            return out
        if isinstance(node, ast.Name):
            target_node = self._resolve_alias_chain(node, scope.method_key)
            if target_node is None:
                return None
            if target_node is not node:
                return self._resolve_to_dict_items(target_node, scope)
            file_dicts = self.table.file_dict_literals.get(scope.file, {})
            if node.id in file_dicts:
                return dict(file_dicts[node.id])
            return None
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            return dict(self.table.local_self_dict_literals.get(scope.method_key, {}).get(node.attr, {})) or None
        return None

    def _resolve_to_list_literal(self, node: ast.expr, scope: Scope) -> Optional[ast.List]:
        """Try to follow *node* down to a concrete ``ast.List`` literal."""
        if isinstance(node, ast.List):
            return node
        if isinstance(node, ast.Name):
            target_node = self._resolve_alias_chain(node, scope.method_key)
            if target_node is None:
                return None
            if target_node is not node:
                return self._resolve_to_list_literal(target_node, scope)
            if scope.cls is not None:
                params = self.table.class_init_params.get((scope.file, scope.cls), [])
                if node.id in params:
                    inst_key = scope.instance_key
                    if inst_key[0] is not None and inst_key[1] is not None:
                        inst_lists = self.table.instance_kw_list_items.get(inst_key, {})
                        list_value = inst_lists.get(node.id)
                        if list_value is not None and list_value.items is not None:
                            return ast.List(elts=list(list_value.items), ctx=ast.Load())
            file_lists = self.table.file_str_list_consts.get(scope.file, {})
            if node.id in file_lists:
                items = file_lists[node.id].items
                if items is not None:
                    # Re-wrap into an ast.List for uniform handling.
                    fake = ast.List(elts=list(items), ctx=ast.Load())
                    return fake
        if isinstance(node, ast.Attribute):
            chain = self._flatten_attr_chain(node)
            if chain is not None:
                base, attrs = chain
                if isinstance(base, ast.Name) and base.id == "self":
                    direct = self._lookup_self_attr_expr(attrs, scope, getattr(node, "lineno", None))
                    if direct is not None:
                        return self._resolve_to_list_literal(direct, scope)
        return None


# ===========================================================================
# End of PR1 ConstantTable + ConstantResolver skeleton
# ===========================================================================


def _strip_inline_comment(code_line: str) -> str:
    """剥离行内 # 注释（尽量不影响字符串中的 #）。

    目的：用于括号计数/静态解析时避免注释中的 " :( " 等非平衡括号破坏 open_count。
    """
    # tokenize.untokenize 会保留大部分原始空白/结构，比简单 split('#') 更安全
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code_line).readline))
        toks = [t for t in toks if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        # 兜底：不处理引号场景，仅用于防止解析直接崩
        return code_line.split('#', 1)[0]


def _join_logical_lines(raw_lines, base_lineno):
    """Join multi-line statements by tracking open brackets/parens.

    Returns list of (start_lineno, joined_text) tuples (start_lineno is the
    absolute line number in the source file of the FIRST physical line that
    contributed to the joined logical line).
    """
    logical = []
    buf = ""
    buf_start = None
    open_count = 0
    for offset, line in enumerate(raw_lines):
        phys_lineno = base_lineno + offset
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if not buf:
                logical.append((phys_lineno, line))
            continue
        if not buf:
            buf_start = phys_lineno
        stripped_nc = _strip_inline_comment(stripped).strip()
        if not stripped_nc:
            continue
        buf += (" " if buf else "") + stripped_nc
        open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
        open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
        if open_count <= 0:
            logical.append((buf_start, buf))
            buf = ""
            buf_start = None
            open_count = 0
    if buf:
        logical.append((buf_start, buf))
    return logical


def load_trace(filepath):
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(filepath, "r") as f:
            data = json.load(f)
    return data


def format_duration(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    elif us >= 1e3:
        return f"{us/1e3:.3f} ms"
    else:
        return f"{us:.2f} us"


def pct_str(part, total):
    return f"{part / total * 100:.2f}%" if total > 0 else "N/A"


# ---------------------------------------------------------------------------
# Phase 0: Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(data, events):
    meta = {}
    # Device info
    dp = data.get("deviceProperties", [])
    if dp:
        meta["device_name"] = dp[0].get("name", "Unknown")
        meta["num_sms"] = dp[0].get("numSms", 0)
    # Distributed info
    di = data.get("distributedInfo", {})
    if di:
        meta["rank"] = di.get("rank", 0)
        meta["world_size"] = di.get("world_size", 1)
    # Thread names & process labels
    thread_names = {}
    process_labels = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "thread_name":
            thread_names[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")
        elif e.get("name") == "process_labels":
            process_labels[e.get("pid")] = e.get("args", {}).get("labels", "")
    meta["thread_names"] = thread_names
    meta["process_labels"] = process_labels
    return meta


def detect_trace_type(events):
    has_fwdbwd = any(e.get("cat") == "fwdbwd" for e in events)
    has_backward = any("backward" in e.get("name", "").lower() or "Backward" in e.get("name", "")
                       for e in events if e.get("cat") in ("cpu_op", "python_function"))
    if has_fwdbwd or has_backward:
        return "training"
    return "inference"


def detect_enhanced_trace(events):
    has_code_location = False
    has_stack_traces = False
    for e in events:
        if e.get("name") == "thread_name" and e.get("ph") == "M":
            if "Code Location" in e.get("args", {}).get("name", ""):
                has_code_location = True
        if has_code_location and has_stack_traces:
            break
    for e in events:
        if e.get("cat") == "kernel":
            st = e.get("args", {}).get("stack", {})
            if "stack_traces" in st:
                has_stack_traces = True
                break
    return has_code_location, has_stack_traces


def load_model_code(code_path):
    if code_path is None:
        return {}
    source_files = {}
    if code_path.endswith(".tar.gz") or code_path.endswith(".tgz"):
        with tarfile.open(code_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    content = tar.extractfile(member).read().decode("utf-8", errors="replace")
                    basename = os.path.basename(member.name)
                    source_files[basename] = content.split("\n")
    elif os.path.isdir(code_path):
        for root, _, files in os.walk(code_path):
            for f in files:
                if f.endswith(".py"):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r", errors="replace") as fp:
                        source_files[f] = fp.read().split("\n")
    return source_files


def _build_ast_frontends(source_files):
    ast_frontends = {}
    for fname in source_files.keys():
        try:
            ast_frontends[fname] = ASTFrontend(
                source='\n'.join(source_files.get(fname, [])),
                path=fname,
            )
        except Exception:
            ast_frontends[fname] = None
    return ast_frontends


def build_module_like_set(source_files, ast_frontends=None):
    class_defs = {}
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)
    for fname, lines in source_files.items():
        fe = ast_frontends.get(fname)
        if fe is not None:
            for node in fe.tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = []
                    for base in node.bases:
                        try:
                            bases.append(fe._node_to_text(base))
                        except Exception:
                            bases.append(getattr(base, "id", getattr(base, "attr", "")))
                    class_defs[(fname, node.name)] = bases
            continue
        try:
            tree = ast.parse("\n".join(lines), filename=fname)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        bases.append(getattr(base, "id", getattr(base, "attr", "")))
                class_defs[(fname, node.name)] = bases

    module_like = set()
    changed = True
    while changed:
        changed = False
        known_names = {c for _, c in module_like}
        for key, bases in class_defs.items():
            if key in module_like:
                continue
            if any(
                b in {"nn.Module", "torch.nn.Module", "Module"}
                or b.endswith(".Module")
                or b in known_names
                for b in bases
            ):
                module_like.add(key)
                changed = True
    return module_like


# ---------------------------------------------------------------------------
# Iter18: AST-based static module-tree extractor (inlined from fake_runner.py
# Phase 1).
#
# This is a *ground-truth lookup* that runs in-process during static analysis.
# It does not replace the existing string/regex-based logic in
# build_static_module_tree(); instead, its output is exposed on each
# tree[cname] entry as ``_ast_*`` keys so downstream consumers (and the
# existing edge/expansion code) can consult an authoritative AST view when
# their heuristics are uncertain (e.g. ModuleList element type for a
# DenseTower built with ``[Dense(...) for _ in output_dims]``, dynamic
# ``setattr(self, f"group_tower_{i}", GroupTower(...))`` inside a ``for i in
# range(N)`` loop, etc.).
#
# The class is a near-verbatim port of fake_runner.py's ``ModuleTreeExtractor``
# adapted to consume ``source_files`` (a dict of ``{filename: [lines]}``)
# directly rather than reading from disk.  Behaviour and outputs are
# semantically identical to the standalone tool whose Phase 1 has been
# verified end-to-end against ``fake_runner_output/module_tree.json``.
# ---------------------------------------------------------------------------

class _AST_ModuleTreeExtractor:
    """Inlined fake_runner.py Phase 1 ``ModuleTreeExtractor``.

    Walks ``__init__`` of every ``nn.Module`` subclass and records:
      • ``self.attr = SomeModule(...)``               → direct child
      • ``setattr(self, name, SomeModule(...))``      → dynamic / literal child
      • ``self.container.append(SomeModule(...))``    → container element
      • Local-var tracking: ``v = SomeClass(...); self.x.append(v)``
                           ``v = SomeClass(...); setattr(self, n, v)``
      • For-loop / if / with bodies are walked (no condition evaluation).

    Exposes:
      • ``classes``                  : {cname: {file, node, bases}}
      • ``nn_module_classes``        : set[cname]
      • ``class_assignments[cname]`` : list[dict] (raw __init__ assigns)
      • ``ast_class_attrs[cname]``   : {attr -> child_class}     (literal)
      • ``ast_container_elems[cname][container] -> [child_class, ...]``
      • ``ast_container_kinds[cname][container] -> "ModuleList"|"ModuleDict"|"Sequential"``
      • ``ast_dynamic_attrs[cname]`` : list[child_class]   (setattr w/ non-literal name)
      • ``module_tree``              : list[dict]  (recursive tree from build_tree)
    """

    _NON_MODULE_TYPES = {
        "OrderedDict", "dict", "list", "set", "tuple",
        "defaultdict", "Counter", "deque",
    }

    def __init__(self, source_files, ast_frontends=None):
        # source_files: {fname: [lines]}
        self.source_files = source_files
        self.ast_frontends = ast_frontends or {}
        self.classes = {}            # cname -> {file, node, bases}
        self.class_assignments = {}  # cname -> list of dicts
        self.ast_class_attrs = defaultdict(dict)
        self.ast_container_elems = defaultdict(lambda: defaultdict(list))
        self.ast_container_kinds = defaultdict(dict)
        self.ast_dynamic_attrs = defaultdict(list)
        self.nn_module_classes = set()
        self.module_tree = []
        self._parse_all_files()
        self._derive_nn_module_set()
        self._build_class_assignments()

    # ---- Parsing -----------------------------------------------------

    def _parse_all_files(self):
        for fname, lines in self.source_files.items():
            tree = None
            fe = self.ast_frontends.get(fname)
            if fe is not None:
                tree = fe.tree
            else:
                try:
                    source = "\n".join(lines)
                    tree = ast.parse(source, filename=fname)
                except SyntaxError:
                    tree = None
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Last-write-wins on duplicate class names (matches
                    # _build_class_map's behaviour).
                    self.classes[node.name] = {
                        "file": fname,
                        "node": node,
                        "bases": [self._get_base_name(b) for b in node.bases],
                    }

    def _get_base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return ""

    def _derive_nn_module_set(self):
        # Iteratively close over: any class whose any base is nn.Module/Module
        # or another already-known nn.Module subclass.
        known = set()
        # Direct hits
        for cname, info in self.classes.items():
            for b in info["bases"]:
                tail = b.split(".")[-1]
                if tail in ("Module", "nn.Module"):
                    known.add(cname)
                    break
                if b in ("nn.Module", "torch.nn.Module"):
                    known.add(cname)
                    break
        # Closure
        changed = True
        while changed:
            changed = False
            for cname, info in self.classes.items():
                if cname in known:
                    continue
                for b in info["bases"]:
                    tail = b.split(".")[-1]
                    if tail in known or b in known:
                        known.add(cname)
                        changed = True
                        break
        self.nn_module_classes = known

    # ---- __init__ walking -------------------------------------------

    def _build_class_assignments(self):
        for cname in self.classes:
            assigns = self._find_init_assignments(cname)
            self.class_assignments[cname] = assigns
            # Materialise lookups
            for a in assigns:
                attr = a["attr"]
                child = a["class"]
                kind = a.get("container_type", "")
                if a.get("_is_container_element"):
                    cont = a.get("_container_attr") or attr.split("[")[0]
                    if child:
                        self.ast_container_elems[cname][cont].append(child)
                    continue
                if a.get("dynamic"):
                    if child:
                        self.ast_dynamic_attrs[cname].append(child)
                    continue
                if a.get("is_container") and kind:
                    self.ast_container_kinds[cname][attr] = kind
                    # The container itself has no immediate child class —
                    # elements are filled by .append() patterns.
                    continue
                if child and attr and "<dynamic>" not in attr:
                    # First write wins — mirrors how the regex pass uses
                    # ``attrs.setdefault``.
                    self.ast_class_attrs[cname].setdefault(attr, child)

    def _find_init_assignments(self, class_name):
        if class_name not in self.classes:
            return []
        cls_node = self.classes[class_name]["node"]
        assignments = []
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self._walk_init_body(item.body, assignments, class_name)
        return assignments

    def _walk_init_body(self, stmts, assignments, parent_class, local_vars=None):
        if local_vars is None:
            local_vars = {}
        for stmt in stmts:
            # local_var = SomeClass(...) and self.attr = SomeClass(...)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._check_self_assign(target, stmt.value, assignments)
                    if isinstance(target, ast.Name):
                        cls, _, _ = self._infer_class_from_value(stmt.value)
                        if cls:
                            local_vars[target.id] = cls
            # call expressions (setattr / append)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name) and call.func.id == "setattr":
                    self._check_setattr(call, assignments, local_vars)
                elif isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                    self._check_container_append(call, assignments, local_vars)
            # Recurse into compound statements
            elif isinstance(stmt, ast.For):
                self._walk_init_body(stmt.body, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.orelse, assignments, parent_class, local_vars)
            elif isinstance(stmt, (ast.If, ast.With)):
                body = getattr(stmt, "body", []) or []
                orelse = getattr(stmt, "orelse", []) or []
                self._walk_init_body(body, assignments, parent_class, local_vars)
                self._walk_init_body(orelse, assignments, parent_class, local_vars)
            elif isinstance(stmt, ast.Try):
                self._walk_init_body(stmt.body, assignments, parent_class, local_vars)
                for h in stmt.handlers:
                    self._walk_init_body(h.body, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.orelse, assignments, parent_class, local_vars)
                self._walk_init_body(stmt.finalbody, assignments, parent_class, local_vars)

    def _check_self_assign(self, target, value, assignments):
        if not isinstance(target, ast.Attribute):
            return
        if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
            return
        attr_name = target.attr
        class_name, is_container, container_type = self._infer_class_from_value(value)
        if class_name or is_container:
            assignments.append({
                "attr": attr_name,
                "class": class_name,
                "is_container": is_container,
                "container_type": container_type,
                "line": getattr(target, "lineno", 0),
            })

    def _check_setattr(self, call_node, assignments, local_vars):
        if len(call_node.args) < 3:
            return
        obj, name_arg, value_arg = call_node.args[0], call_node.args[1], call_node.args[2]
        if not (isinstance(obj, ast.Name) and obj.id == "self"):
            return
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            attr_name = name_arg.value
            dynamic = False
        else:
            attr_name = "<dynamic>"
            dynamic = True
        class_name, is_container, container_type = self._infer_class_from_value(value_arg)
        # Resolve `setattr(self, n, var)` where var is a local-var ctor.
        if not class_name and isinstance(value_arg, ast.Name) and local_vars:
            class_name = local_vars.get(value_arg.id, "")
        if class_name or is_container:
            assignments.append({
                "attr": attr_name,
                "class": class_name,
                "is_container": is_container,
                "container_type": container_type,
                "line": getattr(call_node, "lineno", 0),
                "dynamic": dynamic,
            })

    def _check_container_append(self, call_node, assignments, local_vars):
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            return
        if not isinstance(func.value, ast.Attribute):
            return
        if not (isinstance(func.value.value, ast.Name) and func.value.value.id == "self"):
            return
        container_attr = func.value.attr
        if not call_node.args:
            return
        element_value = call_node.args[0]
        class_name, _, _ = self._infer_class_from_value(element_value)
        if not class_name and isinstance(element_value, ast.Name) and local_vars:
            class_name = local_vars.get(element_value.id, "")
        if class_name:
            assignments.append({
                "attr": f"{container_attr}[*]",
                "class": class_name,
                "is_container": False,
                "container_type": "",
                "line": getattr(call_node, "lineno", 0),
                "dynamic": False,
                "_is_container_element": True,
                "_container_attr": container_attr,
            })

    def _infer_class_from_value(self, value):
        """Return (class_name, is_container, container_type)."""
        if isinstance(value, ast.Call):
            func = value.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
                if isinstance(func.value, ast.Name) and func.value.id == "nn":
                    func_name = f"nn.{func.attr}"
            if func_name in ("ModuleList", "nn.ModuleList"):
                return ("", True, "ModuleList")
            elif func_name in ("ModuleDict", "nn.ModuleDict"):
                return ("", True, "ModuleDict")
            elif func_name in ("Sequential", "nn.Sequential"):
                return ("", True, "Sequential")
            if func_name in self._NON_MODULE_TYPES:
                return ("", False, "")
            if func_name and func_name[0:1].isupper():
                return (func_name, False, "")
            if isinstance(func, ast.Attribute) and func.attr[0:1].isupper():
                return (func.attr, False, "")
        return ("", False, "")

    # ---- Tree builder ------------------------------------------------

    def build_tree(self, root_class, root_path=""):
        self.module_tree = []
        if root_class and root_class in self.classes:
            self._build_recursive(root_class, root_path, depth=0,
                                  ancestor_classes=set())
        return self.module_tree

    def _build_recursive(self, class_name, prefix, depth, ancestor_classes):
        if class_name in ancestor_classes:
            return
        new_ancestors = ancestor_classes | {class_name}
        for assign in self.class_assignments.get(class_name, []):
            attr = assign["attr"]
            path = f"{prefix}.{attr}" if prefix else attr
            child_class = assign["class"]
            entry = {
                "attr_path": path,
                "class": child_class if child_class else assign.get("container_type", "Unknown"),
                "depth": depth + 1,
                "is_container": assign.get("is_container", False),
                "container_type": assign.get("container_type", ""),
                "parent_class": class_name,
                "file": self.classes.get(class_name, {}).get("file", ""),
                "line": assign.get("line", 0),
                "dynamic": assign.get("dynamic", False),
            }
            self.module_tree.append(entry)
            if (child_class and child_class in self.classes
                    and child_class in self.nn_module_classes):
                self._build_recursive(child_class, path, depth + 1, new_ancestors)


def build_main_thread_hierarchy(events):
    profiler_steps = [
        e for e in events
        if str(e.get("name", "")).startswith("ProfilerStep")
        and e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
    ]
    if not profiler_steps:
        return None, [], defaultdict(list)
    main_tid = max(
        {e["tid"] for e in profiler_steps},
        key=lambda tid: sum(1 for e in profiler_steps if e["tid"] == tid),
    )
    main_events = sorted(
        [
            e for e in events
            if e.get("tid") == main_tid and e.get("ph") == "X" and e.get("dur") is not None
        ],
        key=lambda e: (e["ts"], -(e.get("dur") or 0)),
    )

    children = defaultdict(list)
    stack = []
    for e in main_events:
        start = e["ts"]
        end = start + (e.get("dur") or 0)
        while stack and stack[-1][1] <= start:
            stack.pop()
        if stack:
            children[id(stack[-1][0])].append(e)
        stack.append((e, end))

    return main_tid, main_events, children


def phase_name_from_label(name):
    lower = str(name).lower()
    if "optimize" in lower or "optimizer.step" in lower:
        return "optimize"
    if re.search(r"(^|[^a-z])backward([^a-z]|$)", lower):
        return "backward"
    if re.search(r"(^|[^a-z])forward([^a-z]|$)", lower):
        return "forward"
    return None


def extract_step_phase_intervals(main_events, children):
    profiler_steps = [
        e for e in main_events
        if str(e.get("name", "")).startswith("ProfilerStep") and e.get("cat") == "user_annotation"
    ]
    step_infos = []
    for profiler_step in profiler_steps:
        direct_children = children[id(profiler_step)]
        if not direct_children:
            continue
        step_event = max(direct_children, key=lambda e: e.get("dur", 0))
        info = {
            "profiler_name": profiler_step["name"],
            "step_interval": (profiler_step["ts"], profiler_step["ts"] + profiler_step["dur"]),
            "forward": [],
            "backward": [],
            "optimize": [],
        }
        for level2 in children[id(step_event)]:
            level2_name = str(level2.get("name", ""))
            level2_name_lower = level2_name.lower()
            if "forward_backward" in level2_name_lower:
                for level3 in children[id(level2)]:
                    phase = phase_name_from_label(level3.get("name", ""))
                    if phase in ("forward", "backward"):
                        info[phase].append((level3["ts"], level3["ts"] + level3["dur"], level3.get("name", "")))
            else:
                phase = phase_name_from_label(level2_name)
                if phase == "optimize":
                    info["optimize"].append((level2["ts"], level2["ts"] + level2["dur"], level2.get("name", "")))
        step_infos.append(info)
    return step_infos


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def classify_kernel_phase(kernel_start, kernel_end, step_infos):
    matched_step = None
    for info in step_infos:
        step_start, step_end = info["step_interval"]
        if kernel_end >= step_start and kernel_start <= step_end:
            matched_step = info
            break
    if matched_step is None:
        return "other"

    phase_overlap = {"forward": 0.0, "backward": 0.0, "optimize": 0.0}
    for phase in ("forward", "backward", "optimize"):
        for start, end, _label in matched_step[phase]:
            phase_overlap[phase] += overlap(kernel_start, kernel_end, start, end)

    best_phase = max(phase_overlap, key=phase_overlap.get)
    return best_phase if phase_overlap[best_phase] > 0 else "other"


def _normalize_runtime_module_name(mod_name):
    short = (mod_name or "").replace("nn.Module: ", "").replace("nn.Module:", "")
    m_callsite = re.search(r',\s*callsite:\s*(\d+)', short)
    callsite = int(m_callsite.group(1)) if m_callsite else None
    short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
    m_index = re.search(r'_(\d+)$', short)
    runtime_index = int(m_index.group(1)) if m_index else None
    base = re.sub(r'_\d+$', '', short)
    stripped = re.sub(r'^(FSDP|DDP|DistributedDataParallel|Wrapped)', '', base)
    return {
        "runtime_name": short,
        "class_name": stripped,
        "runtime_index": runtime_index,
        "callsite": callsite,
    }


def _build_external_id_to_module_map(events):
    by_tid = defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        if e.get("cat") == "kernel":
            continue
        by_tid[e.get("tid")].append(e)

    ext_to_module = {}
    for tid, tid_events in by_tid.items():
        tid_events.sort(key=lambda e: (e["ts"], -(e.get("dur") or 0), 0 if e.get("cat") == "python_function" else 1))
        module_stack = []
        for e in tid_events:
            start = e["ts"]
            end = start + (e.get("dur") or 0)
            while module_stack and module_stack[-1][1] <= start:
                module_stack.pop()

            is_module = e.get("cat") == "python_function" and str(e.get("name", "")).startswith("nn.Module:")
            if is_module:
                module_stack.append((e, end))

            ext_id = e.get("args", {}).get("External id")
            if ext_id is not None and module_stack:
                ext_to_module.setdefault(ext_id, module_stack[-1][0].get("name"))

    return ext_to_module


# ==========================================================================
# Timing Pipeline Redesign — 4-step instance-level attribution (cherry-picked)
# Restored from 9a103e3 after accidental deletion in 08cb9e3 (HTML cleanup).
# --------------------------------------------------------------------------
# Step 1: build_fwdbwd_flow_index           — pair fwdbwd flow events
# Step 2: build_kernel_attribution_table    — kernel → InstanceKey weights
# Step 3: rollup_instance_timing            — bottom-up self/inclusive sum
# Step 4: build_timing_panel_data           — instance + class output
# Top-level entry: build_instance_timing_pipeline
#
# Design doc: timing_fix_workdir/output/timing_redesign.md
# ==========================================================================


_STACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')


def _parse_user_frames(traces, source_files):
    """Parse a list of trace strings into per-trace user-code frame chains.

    Returns a list of frame chains, one per input trace, preserving the
    original outer-first ordering inside each chain. Frames whose file is
    not in source_files (third-party libs, torch internals) are dropped.
    Empty chains (traces with no user frames) are filtered out.

    Contract change (P0-A): previously this function returned a single flat
    list and stopped at the first non-empty trace, which silently dropped
    every additional trace recorded for a kernel and prevented multi-leaf
    weight accumulation.  We now return all per-trace chains so downstream
    can attribute weight across distinct leaf InstanceKeys (and accumulate
    repeats when multiple traces resolve to the same key).

    Returns:
        list[list[(fname_basename, lineno, func_name)]]
    """
    if not traces or not source_files:
        return []
    chains = []
    for trace in traces:
        chain = []
        for ln in trace.split("\n"):
            m = _STACK_FRAME_RE.search(ln)
            if not m:
                continue
            fpath, lineno_s, func = m.groups()
            fname = os.path.basename(fpath)
            if fname not in source_files:
                continue
            chain.append((fname, int(lineno_s), func))
        if chain:
            chains.append(chain)
    return chains


def _frame_class(fname, lineno, class_map):
    cls, _ = _find_class_for_line(fname, lineno, class_map)
    return cls


def build_fwdbwd_flow_index(events):
    """Step 1: pair fwdbwd flow events (ph='s' on forward tid, ph='f' on
    backward tid) and resolve them to forward/backward op time ranges.

    Returns: {
        "by_bwd_tid": {tid: [entry, ...]},   # sorted by bwd_start_us
        "all": [entry, ...],                 # sorted by bwd_start_us
    }
    where each entry = {
        "flow_id": int,
        "fwd_ts": float, "fwd_tid": int,
        "bwd_ts": float, "bwd_tid": int,
    }
    The `_ts` values are the flow source/dest timestamps (μs).  Callers can
    look up the enclosing forward/backward op by interval-containment.
    """
    if not events:
        return {"by_bwd_tid": {}, "all": []}
    pairs = {}
    for e in events:
        if e.get("cat") != "fwdbwd":
            continue
        ph = e.get("ph")
        fid = e.get("id")
        if fid is None or ph not in ("s", "f"):
            continue
        slot = pairs.setdefault(fid, {})
        slot[ph] = e
    entries = []
    for fid, pp in pairs.items():
        s_ev = pp.get("s")
        f_ev = pp.get("f")
        if not s_ev or not f_ev:
            continue
        try:
            entries.append({
                "flow_id": fid,
                "fwd_ts": float(s_ev.get("ts", 0.0)),
                "fwd_tid": s_ev.get("tid"),
                "bwd_ts": float(f_ev.get("ts", 0.0)),
                "bwd_tid": f_ev.get("tid"),
            })
        except Exception:
            continue
    entries.sort(key=lambda x: x["bwd_ts"])
    by_tid = defaultdict(list)
    for ent in entries:
        by_tid[ent["bwd_tid"]].append(ent)
    return {"by_bwd_tid": dict(by_tid), "all": entries}


def _find_enclosing_op(events_by_tid_sorted, tid, ts):
    """Find the smallest cpu_op event on `tid` whose [ts, ts+dur] contains
    the timestamp. Returns event dict or None.

    `events_by_tid_sorted`: {tid: [event]} sorted by ts ascending.
    """
    cands = events_by_tid_sorted.get(tid, [])
    if not cands:
        return None
    # Linear scan with early exit (events count per tid is bounded; OK).
    best = None
    best_dur = float("inf")
    for ev in cands:
        ev_ts = ev.get("ts", 0.0)
        if ev_ts > ts:
            break
        dur = ev.get("dur") or 0.0
        if ev_ts <= ts <= ev_ts + dur and dur < best_dur:
            best = ev
            best_dur = dur
    return best


def _build_cpu_op_index_by_tid(events):
    """Group cpu_op events by tid and sort by ts. Used for fwdbwd flow
    interval resolution.
    """
    by_tid = defaultdict(list)
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        if e.get("dur") is None:
            continue
        by_tid[e.get("tid")].append(e)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: x.get("ts", 0.0))
    return dict(by_tid)


def _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, bwd_kernel_ts, bwd_kernel_tid):
    """For a backward kernel at (ts, tid), find the corresponding forward
    scope time range.

    Returns (fwd_start, fwd_end, fwd_tid) or None.
    """
    cands = fwdbwd_index["by_bwd_tid"].get(bwd_kernel_tid, [])
    if not cands:
        return None
    # Find the fwdbwd entry whose enclosing backward op contains bwd_kernel_ts.
    # Strategy: pick the entry with the largest bwd_ts <= kernel_ts whose
    # enclosing bwd op covers the kernel.
    chosen = None
    chosen_op = None
    for ent in cands:
        if ent["bwd_ts"] > bwd_kernel_ts:
            continue
        bwd_op = _find_enclosing_op(cpu_op_by_tid, ent["bwd_tid"], ent["bwd_ts"])
        if bwd_op is None:
            continue
        bwd_start = bwd_op.get("ts", 0.0)
        bwd_end = bwd_start + (bwd_op.get("dur") or 0.0)
        if bwd_start <= bwd_kernel_ts <= bwd_end:
            chosen = ent
            chosen_op = bwd_op
            # Don't break — later entries (same op) override since they
            # represent finer-grained scope.
    if chosen is None:
        return None
    fwd_op = _find_enclosing_op(cpu_op_by_tid, chosen["fwd_tid"], chosen["fwd_ts"])
    if fwd_op is None:
        return None
    fwd_start = fwd_op.get("ts", 0.0)
    fwd_end = fwd_start + (fwd_op.get("dur") or 0.0)
    return (fwd_start, fwd_end, chosen["fwd_tid"])


# --------------------------------------------------------------------------
# Step 2: kernel attribution to InstanceKey
# --------------------------------------------------------------------------

def _extract_instance_keys_from_stack(frames, class_map):
    """From a list of (fname, lineno, func_name) frames (outer→inner),
    walk class boundaries and produce a list of InstanceKey ordered from
    outermost to deepest.

    InstanceKey = (class_name, callsite_file, callsite_line, ancestors_tuple)
      - callsite_file/line: parent frame's (file, line) — i.e. where the
        parent class's code calls into this child class.
      - ancestors_tuple: ((file, line), ...) of all preceding callsites
        outermost→inner, NOT including self.

    Single-frame fallback: when only one user frame is present and it maps
    to a class, emit a weak key for that class (callsite_file=<self>,
    callsite_line=that frame's line, ancestors=()).  This lets us attribute
    e.g. autograd-leaf kernels whose stack has only the leaf class's forward
    method.

    Returns [] only if no frame maps to any user class.
    """
    if not frames or not class_map:
        return []
    # Compute class for each frame.
    frame_classes = []
    for f in frames:
        cls = _frame_class(f[0], f[1], class_map)
        frame_classes.append(cls)
    keys = []
    # Find the first frame that maps to a class.
    first_idx = -1
    for i, cls in enumerate(frame_classes):
        if cls:
            first_idx = i
            break

    if first_idx != -1:
        # Outermost class attribution. If it was called from a user frame
        # that doesn't map to a class (e.g. main.py), use that as callsite.
        # If it's the very first frame in the stack, use a <root> anchor.
        cls = frame_classes[first_idx]
        if first_idx > 0:
            cf, cl = frames[first_idx-1][0], frames[first_idx-1][1]
        else:
            # Use the class's start line as a stable anchor for the root.
            cf, cl = "<root>", class_map.get((frames[0][0], cls), {}).get("start", 0)
        
        root_key = (cls, cf, cl, ())
        keys.append(root_key)
        
        ancestors = [(cf, cl)]
        last_class = cls
        last_frame = frames[first_idx]
        for i in range(first_idx + 1, len(frames)):
            cur_class = frame_classes[i]
            if cur_class and last_class and cur_class != last_class:
                key = (cur_class, last_frame[0], last_frame[1], tuple(ancestors))
                keys.append(key)
                ancestors.append((last_frame[0], last_frame[1]))
            if cur_class:
                last_class = cur_class
            last_frame = frames[i]

    if not keys:
        # No class boundary detected. Emit a weak key for the deepest
        # frame whose class is known so the kernel still gets attributed.
        for i in range(len(frames) - 1, -1, -1):
            cls = frame_classes[i]
            if cls:
                key = (cls, "<self>", frames[i][1], ())
                keys.append(key)
                break
    return keys


def _is_backward_trace(traces):
    if not traces:
        return False
    for t in traces:
        if "Gradient" in t or "Backward" in t or "backward" in t:
            return True
    return False


def build_kernel_stack_cost_table(events, source_files, fwdbwd_index):
    """Step 1a: 将所有 kernel 对齐到"堆栈-开销"记录。

    Returns:
        dict[kernel_idx, {
            "dur_us":    float,
            "phase":     "fwd" | "bwd" | "other",
            "chains":    list[list[(fname, lineno, func_name)]],
            "mod_name":  str | None,
            "unmatched": bool,
        }]
    """
    if not events:
        return {}

    cpu_op_by_tid = _build_cpu_op_index_by_tid(events) if fwdbwd_index else {}
    ext_to_module = _build_external_id_to_module_map(events)
    bwd_tids = set((fwdbwd_index or {}).get("by_bwd_tid", {}).keys())

    # Forward kernels with usable user-frame chains; used by Path B to borrow
    # the nearest forward stack within the matched fwdbwd scope.
    fwd_kernel_buckets = defaultdict(list)  # tid -> [(ts, idx, chains)]
    kernel_rows = {}

    for idx, e in enumerate(events):
        if e.get("cat") != "kernel":
            continue
        dur = float(e.get("dur") or 0.0)
        if dur <= 0:
            continue
        ts = float(e.get("ts") or 0.0)
        tid = e.get("tid")
        traces = e.get("args", {}).get("stack", {}).get("stack_traces", [])
        is_bwd = (tid in bwd_tids) or _is_backward_trace(traces)
        chains = _parse_user_frames(traces, source_files)
        kernel_rows[idx] = {
            "event": e,
            "ts": ts,
            "tid": tid,
            "dur_us": dur,
            "traces": traces,
            "chains": chains,
            "is_bwd": is_bwd,
        }
        if (not is_bwd) and chains:
            fwd_kernel_buckets[tid].append((ts, idx, chains))

    for tid in fwd_kernel_buckets:
        fwd_kernel_buckets[tid].sort(key=lambda x: x[0])

    table = {}
    for idx, meta in kernel_rows.items():
        phase = "bwd" if meta["is_bwd"] else "fwd"
        if meta["traces"]:
            table[idx] = {
                "dur_us": meta["dur_us"],
                "phase": "bwd" if _is_backward_trace(meta["traces"]) else "fwd",
                "chains": meta["chains"],
                "mod_name": None,
                "unmatched": False,
            }
            continue

        borrowed_chains = []
        if meta["is_bwd"] and fwdbwd_index:
            scope = _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, meta["ts"], meta["tid"])
            if scope is not None:
                fwd_start, fwd_end, fwd_tid = scope
                nearest = None
                nearest_dist = None
                for cand_ts, cand_idx, cand_chains in fwd_kernel_buckets.get(fwd_tid, []):
                    if not (fwd_start <= cand_ts <= fwd_end):
                        continue
                    dist = abs(cand_ts - meta["ts"])
                    if nearest is None or dist < nearest_dist:
                        nearest = cand_chains
                        nearest_dist = dist
                if nearest:
                    borrowed_chains = nearest

        if borrowed_chains:
            table[idx] = {
                "dur_us": meta["dur_us"],
                "phase": "bwd",
                "chains": borrowed_chains,
                "mod_name": None,
                "unmatched": False,
            }
            continue

        ext_id = meta["event"].get("args", {}).get("External id")
        mod_name = ext_to_module.get(ext_id)
        if not mod_name:
            mod_name, _pidx = find_module_parent(meta["event"], events)
        normalized = _normalize_runtime_module_name(mod_name) if mod_name else None
        class_name = normalized.get("class_name") if normalized else None
        table[idx] = {
            "dur_us": meta["dur_us"],
            "phase": phase,
            "chains": [],
            "mod_name": class_name,
            "unmatched": class_name is None,
        }

    return table


def build_kernel_attribution_table(events, source_files, class_map, step_infos, fwdbwd_index):
    """Step 2: for every kernel, decide which InstanceKey(s) it belongs to
    and with what weight (weights sum to 1 per kernel).

    Returns (attribution, debug_stats):
      attribution: {kernel_idx: {InstanceKey: weight}}
      debug_stats: counts dict
    """
    if not source_files or not class_map:
        return {}, {"total_kernels": 0, "fwd_attributed": 0, "bwd_attributed": 0,
                    "wrapped_fallback": 0, "fwd_unattributed": 0, "bwd_unattributed": 0,
                    "bwd_via_flow_narrowed": 0}

    cpu_op_by_tid = _build_cpu_op_index_by_tid(events) if fwdbwd_index else {}
    ext_to_module = _build_external_id_to_module_map(events)

    # Index forward kernels by enclosing forward scope (fwd_tid, [fwd_start,fwd_end])
    # — actually we don't index globally; instead, for each backward kernel
    # we find its fwd scope and re-scan forward kernels within that range.
    # To make that efficient, pre-bucket forward kernels (with stack_traces)
    # by tid, sorted by ts, and only those classified as 'forward' phase.
    fwd_kernel_buckets = defaultdict(list)  # tid -> [(ts, ts+dur, kernel_idx, frames)]
    kernel_meta = {}  # idx -> dict
    for idx, e in enumerate(events):
        if e.get("cat") != "kernel":
            continue
        dur = float(e.get("dur") or 0.0)
        if dur <= 0:
            continue
        ts = float(e.get("ts") or 0.0)
        traces = e.get("args", {}).get("stack", {}).get("stack_traces", [])
        is_bwd = _is_backward_trace(traces)
        kernel_meta[idx] = {
            "ts": ts, "dur": dur, "tid": e.get("tid"),
            "is_bwd": is_bwd, "traces": traces, "ext_id": e.get("args", {}).get("External id"),
        }
        if not is_bwd and traces:
            chains = _parse_user_frames(traces, source_files)
            # The fwd_kernel_buckets entry is currently informational
            # (no downstream consumer yet); record the first non-empty
            # chain so the bucket schema stays stable.
            if chains:
                fwd_kernel_buckets[e.get("tid")].append((ts, ts + dur, idx, chains[0]))
    for tid in fwd_kernel_buckets:
        fwd_kernel_buckets[tid].sort(key=lambda x: x[0])

    attribution = {}
    stats = {
        "total_kernels": 0, "fwd_attributed": 0, "bwd_attributed": 0,
        "wrapped_fallback": 0, "fwd_unattributed": 0, "bwd_unattributed": 0,
        "bwd_via_flow_narrowed": 0,
        "total_kernel_dur_us": 0.0,
    }

    def _wrapped_fallback(idx, mod_name):
        info = _normalize_runtime_module_name(mod_name)
        cls = info.get("class_name")
        if not cls:
            return False
        # Synthesize a weak InstanceKey: ancestors empty, callsite_line None.
        rt_idx = info.get("runtime_index")
        callsite = info.get("callsite")
        # Use callsite line if available; otherwise use runtime_index packed as -idx
        cs_line = callsite if callsite is not None else (-(rt_idx or 0))
        key = (cls, "<wrapped>", int(cs_line) if cs_line is not None else 0, ())
        attribution[idx] = {key: 1.0}
        return True

    for idx, meta in kernel_meta.items():
        stats["total_kernels"] += 1
        stats["total_kernel_dur_us"] += meta["dur"]
        traces = meta["traces"]
        is_bwd = meta["is_bwd"]
        if not traces:
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                # Second-level fallback: walk parent chain via _P pointers.
                mod_name, _pidx = find_module_parent(events[idx], events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        chains = _parse_user_frames(traces, source_files)
        if not chains:
            # Try wrapped fallback
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                mod_name, _pidx = find_module_parent(events[idx], events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        # Accumulate leaf-InstanceKey hits across ALL traces of this kernel.
        # Contract (P0-A, no-dedup):
        #   * Every non-empty trace contributes +1 weight to its leaf key.
        #   * If two traces resolve to the same leaf key, that key gets +2.
        #   * Final weight = hit_count / total_hits (sum normalized to 1.0).
        # This makes 1 kernel × N traces equivalent to N kernels each
        # attributed once, as required by the timing contract.
        hit_counts = {}
        total_hits = 0
        for chain in chains:
            chain_keys = _extract_instance_keys_from_stack(chain, class_map)
            if not chain_keys:
                continue
            leaf = chain_keys[-1]
            hit_counts[leaf] = hit_counts.get(leaf, 0) + 1
            total_hits += 1

        if total_hits == 0:
            mod_name = ext_to_module.get(meta["ext_id"])
            if not mod_name:
                mod_name, _pidx = find_module_parent(events[idx], events)
            if mod_name and _wrapped_fallback(idx, mod_name):
                stats["wrapped_fallback"] += 1
                continue
            if is_bwd:
                stats["bwd_unattributed"] += 1
            else:
                stats["fwd_unattributed"] += 1
            continue

        if is_bwd and fwdbwd_index and fwdbwd_index["all"]:
            # Try to narrow via fwdbwd flow: find forward scope, then look
            # for forward kernels whose stack reproduces the leaf keys —
            # if there's a unique match, we keep the current weights.
            scope = _resolve_fwdbwd_scope(fwdbwd_index, cpu_op_by_tid, meta["ts"], meta["tid"])
            if scope is not None:
                stats["bwd_via_flow_narrowed"] += 1
            # Backward stack_traces already encode forward callsite chains
            # (recorded by autograd), so the leaf keys are normally already
            # correct. Flow-based narrowing is informational; we keep the
            # accumulated weights.

        # Normalize hit counts to weights summing to 1.0.
        attribution[idx] = {k: cnt / total_hits for k, cnt in hit_counts.items()}
        if is_bwd:
            stats["bwd_attributed"] += 1
        else:
            stats["fwd_attributed"] += 1

    return attribution, stats


# --------------------------------------------------------------------------
# Step 3: bottom-up self/inclusive rollup
# --------------------------------------------------------------------------

def rollup_instance_timing(kernel_attribution, events, step_infos, class_map, roots=None):
    """Step 3: aggregate kernel durations into per-InstanceKey self_us, then
    compute inclusive_us by walking the InstanceKey ancestors chain.

    Returns: {InstanceKey: instance_record}
    instance_record = {
        "class_name", "callsite_file", "callsite_line", "ancestors",
        "self_us": {"forward","backward","optimize","other"},
        "inclusive_us": {"forward","backward","optimize","other"},
        "child_keys": set,
    }
    """
    num_steps = max(1, len(step_infos))
    timings = {}

    def _ensure(key):
        if key not in timings:
            class_name, csf, csl, ancestors = key
            timings[key] = {
                "class_name": class_name,
                "callsite_file": csf,
                "callsite_line": csl,
                "ancestors": ancestors,
                "self_us": {"forward": 0.0, "backward": 0.0, "optimize": 0.0, "other": 0.0},
                "inclusive_us": {"forward": 0.0, "backward": 0.0, "optimize": 0.0, "other": 0.0},
                "child_keys": set(),
            }
        return timings[key]

    # Phase A: self_us
    for idx, weights in kernel_attribution.items():
        e = events[idx]
        dur = float(e.get("dur") or 0.0)
        ts = float(e.get("ts") or 0.0)
        phase = classify_kernel_phase(ts, ts + dur, step_infos)
        if phase not in ("forward", "backward", "optimize", "other"):
            phase = "other"
        for key, w in weights.items():
            rec = _ensure(key)
            rec["self_us"][phase] += dur * w

    # Divide by num_steps for per-step average.
    for rec in timings.values():
        for ph in rec["self_us"]:
            rec["self_us"][ph] /= num_steps

    # Phase B: parent-child relationships from ancestors
    # parent of (cls, csf, csl, ancestors) = (parent_cls, parent_csf, parent_csl, ancestors[:-1])
    # where parent_cls is the class containing (csf, csl).
    for key in list(timings.keys()):
        class_name, csf, csl, ancestors = key
        if not ancestors:
            continue  # top-level / weak key
        # Find parent's class via class_map
        parent_cls = _frame_class(csf, csl, class_map) if csf != "<wrapped>" else None
        if not parent_cls:
            continue
        parent_csf, parent_csl = ancestors[-1]
        parent_ancestors = ancestors[:-1]
        parent_key = (parent_cls, parent_csf, parent_csl, parent_ancestors)
        if parent_key not in timings:
            # Synthesize parent record so its inclusive_us aggregates
            # children even when no kernel directly attributed to parent.
            _ensure(parent_key)
        timings[parent_key]["child_keys"].add(key)

    # Phase C: bottom-up sum (topological order — leaves first).
    # Compute depth = len(ancestors); larger depth processed first.
    def _depth(k):
        return len(k[3])
    sorted_keys = sorted(timings.keys(), key=_depth, reverse=True)
    # Initialize inclusive = self
    for k in timings:
        for ph, v in timings[k]["self_us"].items():
            timings[k]["inclusive_us"][ph] = v
    # Add children inclusive
    for k in sorted_keys:
        rec = timings[k]
        for ck in rec["child_keys"]:
            child = timings.get(ck)
            if not child:
                continue
            for ph, v in child["inclusive_us"].items():
                rec["inclusive_us"][ph] += v

    # Phase D: orphan rollup to model roots
    if roots:
        # Find a primary root instance to receive orphans
        primary_root_key = None
        for k in timings:
            if k[0] in roots and k[1] == "<root>":
                primary_root_key = k
                break
        if not primary_root_key:
            for k in timings:
                if k[0] in roots:
                    primary_root_key = k
                    break
        
        if primary_root_key:
            # Find all orphans (nodes with no parent)
            all_children = set()
            for rec in timings.values():
                all_children.update(rec["child_keys"])
            
            for k, rec in timings.items():
                if k != primary_root_key and k not in all_children:
                    # This is an orphan! Add its inclusive time to primary root.
                    # This ensures model-level total inclusive time is preserved.
                    for ph, v in rec["inclusive_us"].items():
                        timings[primary_root_key]["inclusive_us"][ph] += v

    return timings


# --------------------------------------------------------------------------
# Step 4: instance + class panel data
# --------------------------------------------------------------------------

def build_timing_panel_data(instance_timing, class_map, step_dur_us):
    """Step 4: convert instance_timing → timing_data fields:
      - runtime_instance_timings_by_class (instance level)
      - class_durations / class_durations_fwd / class_durations_bwd
        (class level = sum of all instances' inclusive)
    """
    by_class = defaultdict(list)
    for key, rec in instance_timing.items():
        cls = rec["class_name"]
        csf = rec["callsite_file"]
        csl = rec["callsite_line"]
        anc = rec["ancestors"]
        sus_fwd = rec["self_us"]["forward"]
        sus_bwd = rec["self_us"]["backward"]
        sus_opt = rec["self_us"]["optimize"]
        sus_oth = rec["self_us"]["other"]
        inc_fwd = rec["inclusive_us"]["forward"]
        inc_bwd = rec["inclusive_us"]["backward"]
        inc_opt = rec["inclusive_us"]["optimize"]
        inc_oth = rec["inclusive_us"]["other"]
        self_total = sus_fwd + sus_bwd + sus_opt + sus_oth
        inc_total = inc_fwd + inc_bwd + inc_opt + inc_oth
        runtime_name = (f"{cls}@{os.path.basename(csf)}:{csl}"
                        if csf and csf != "<wrapped>" and csl and csl > 0
                        else cls)
        item = {
            "runtime_name": runtime_name,
            "runtime_index": rec.get("runtime_index"),
            "callsite": csl if (csl and csl > 0) else None,
            "callsite_file": csf,
            "callsite_line": csl if (csl and csl > 0) else None,
            "ancestors": [list(a) for a in anc],
            # Backwards-compat: forward_us/backward_us/total_us are EXCLUSIVE (self)
            # values, matching the prior contract under the bottom-up rollup.
            "forward_us": sus_fwd,
            "backward_us": sus_bwd,
            "optimize_us": sus_opt,
            "other_us": sus_oth,
            "total_us": self_total,
            "self_us": self_total,
            # New: inclusive values for downstream UI.
            "inclusive_us": inc_total,
            "inclusive_forward_us": inc_fwd,
            "inclusive_backward_us": inc_bwd,
        }
        by_class[cls].append(item)

    # Sort each class's instances deterministically.
    for cn in by_class:
        by_class[cn].sort(key=lambda x: (
            x.get("callsite_file") or "",
            x.get("callsite_line") or 0,
            x.get("runtime_name") or "",
        ))

    # Build class-level totals from instance inclusive.
    class_durations = {}
    class_durations_fwd = {}
    class_durations_bwd = {}
    for cls, items in by_class.items():
        cls_fwd = sum(it.get("inclusive_forward_us", 0.0) for it in items)
        cls_bwd = sum(it.get("inclusive_backward_us", 0.0) for it in items)
        cls_total = sum(it.get("inclusive_us", 0.0) for it in items)
        class_durations_fwd[cls] = cls_fwd
        class_durations_bwd[cls] = cls_bwd
        class_durations[cls] = cls_total

    return {
        "runtime_instance_timings_by_class": dict(by_class),
        "class_durations": class_durations,
        "class_durations_fwd": class_durations_fwd,
        "class_durations_bwd": class_durations_bwd,
    }


def build_instance_timing_pipeline(events, source_files, class_map, step_infos, step_dur_us, roots=None):
    """Top-level entry: runs Step 1→4 and returns timing_data fragment.

    Output keys:
      - runtime_instance_timings_by_class
      - class_durations / class_durations_fwd / class_durations_bwd
      - _timing_pipeline_stats (debug)
    """
    fwdbwd_index = build_fwdbwd_flow_index(events)
    attribution, stats = build_kernel_attribution_table(
        events, source_files, class_map, step_infos, fwdbwd_index,
    )
    instance_timing = rollup_instance_timing(attribution, events, step_infos, class_map, roots=roots)
    panel = build_timing_panel_data(instance_timing, class_map, step_dur_us)
    panel["_timing_pipeline_stats"] = stats
    num_steps = max(1, len(step_infos))
    panel["step_kernel_us"] = stats.get("total_kernel_dur_us", 0) / num_steps
    
    total_attributed_us = sum(
        rec["self_us"]["forward"] + rec["self_us"]["backward"] + 
        rec["self_us"]["optimize"] + rec["self_us"]["other"] 
        for rec in instance_timing.values()
    )
    panel["unattributed_kernel_us"] = max(0, panel["step_kernel_us"] - total_attributed_us)
    return panel


# --------------------------------------------------------------------------
# Coverage summary — diagnostics over DAG groups
# --------------------------------------------------------------------------

def compute_timing_coverage_summary(groups):
    """Compute timing coverage statistics over DAG sub-groups.

    Walks the DAG ``groups`` list (as produced by ``generate_html_flowchart``
    with ``_return_data_only=True``) and reports how many sub-group instances
    have non-zero timing attributed to them.

    Statistics rules:
        * Sub-groups only — top-level / root groups (``depth <= 0``) are
          excluded so the figure reflects user-facing modules rather than the
          synthetic RootModule wrapper.
        * An instance is considered "timed" when ``kernel_us > 0`` OR
          ``total_us > 0`` (whichever field is present on the group dict).
        * Each list entry in the ``groups`` argument is treated as one
          instance — the caller is expected to pass the flat ``dag_groups``
          list which already contains one entry per instance occurrence.

    Args:
        groups: Iterable of group dicts. Each dict is expected to expose at
            least ``depth``; ``label``/``class_name`` and one of
            ``attr_name``/``id`` are used to populate the missed-instance
            list. ``kernel_us`` and/or ``total_us`` (in microseconds) are
            consulted to decide whether an instance is timed. Missing fields
            are treated as zero.

    Returns:
        dict with keys::

            {
              "total_instances": int,
              "timed_instances": int,
              "coverage_rate": float,           # in [0.0, 1.0]
              "no_timing_instances": [
                  {"class_name": str, "instance_key": str,
                   "kernel_us": float, "total_us": float},
                  ...
              ]
            }

        ``coverage_rate`` is ``0.0`` when ``total_instances == 0`` to avoid
        ``ZeroDivisionError``; callers can therefore safely format the value
        without extra guards.
    """
    total = 0
    timed = 0
    no_timing = []

    if not groups:
        return {
            "total_instances": 0,
            "timed_instances": 0,
            "coverage_rate": 0.0,
            "no_timing_instances": [],
        }

    for g in groups:
        if not isinstance(g, dict):
            continue
        # Skip the synthetic RootModule wrapper / any depth==0 entries; we only
        # want user-facing sub-groups in this report.
        depth = g.get("depth", 0)
        try:
            depth_i = int(depth) if depth is not None else 0
        except (TypeError, ValueError):
            depth_i = 0
        if depth_i <= 0:
            continue

        total += 1

        kernel_us = g.get("kernel_us", 0) or 0
        total_us = g.get("total_us", 0) or 0
        try:
            kernel_us = float(kernel_us)
        except (TypeError, ValueError):
            kernel_us = 0.0
        try:
            total_us = float(total_us)
        except (TypeError, ValueError):
            total_us = 0.0

        if kernel_us > 0 or total_us > 0:
            timed += 1
        else:
            class_name = g.get("class_name") or g.get("label") or ""
            instance_key = (
                g.get("attr_name")
                or g.get("id")
                or ""
            )
            no_timing.append({
                "class_name": class_name,
                "instance_key": instance_key,
                "kernel_us": kernel_us,
                "total_us": total_us,
            })

    coverage_rate = (timed / total) if total > 0 else 0.0
    return {
        "total_instances": total,
        "timed_instances": timed,
        "coverage_rate": coverage_rate,
        "no_timing_instances": no_timing,
    }


def analyze_source_hotspots(events, source_files):
    # Build class/method line range map
    class_map = _build_class_map(source_files)

    line_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    class_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    class_method_dur = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
    total_kernel_dur = 0.0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        total_kernel_dur += dur
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        if not traces:
            continue

        is_bwd = any("Gradient" in t or "Backward" in t or "backward" in t for t in traces)
        phase = "bwd" if is_bwd else "fwd"
        n_traces = max(1, len(traces))
        per_trace_dur = dur / n_traces

        seen_classes = set()
        seen_lines = set()
        for trace in traces:
            for tl in trace.split("\n"):
                m = re.search(r'File "([^"]+)", line (\d+), in (\w+)', tl)
                if not m:
                    continue
                fpath, lineno_s, func = m.groups()
                fname = os.path.basename(fpath)
                lineno = int(lineno_s)
                loc = f"{fname}:{lineno_s}"

                if loc not in seen_lines:
                    seen_lines.add(loc)
                    line_dur[loc][phase] += per_trace_dur

                cname, mname = _find_class_for_line(fname, lineno, class_map)
                if cname:
                    ckey = f"{fname}:{cname}"
                    if ckey not in seen_classes:
                        seen_classes.add(ckey)
                        class_dur[ckey][phase] += per_trace_dur
                    if mname:
                        cmkey = f"{ckey}.{mname}"
                        if cmkey not in seen_classes:
                            seen_classes.add(cmkey)
                            class_method_dur[cmkey][phase] += per_trace_dur

    # Sort classes by total time
    sorted_classes = sorted(class_dur.items(), key=lambda x: -(x[1]["fwd"] + x[1]["bwd"]))
    sorted_methods = sorted(class_method_dur.items(), key=lambda x: -(x[1]["fwd"] + x[1]["bwd"]))

    # For top classes, collect hot lines
    class_hot_lines = defaultdict(list)
    for loc, phases in line_dur.items():
        parts = loc.split(":")
        fname, lineno = parts[0], int(parts[1])
        cname, mname = _find_class_for_line(fname, lineno, class_map)
        if cname:
            total = phases["fwd"] + phases["bwd"]
            class_hot_lines[f"{fname}:{cname}"].append((lineno, total, phases["fwd"], phases["bwd"], mname))

    for k in class_hot_lines:
        class_hot_lines[k].sort(key=lambda x: -x[1])

    return {
        "class_dur": dict(class_dur),
        "class_method_dur": dict(class_method_dur),
        "line_dur": {k: dict(v) for k, v in line_dur.items()},
        "total_kernel_dur": total_kernel_dur,
        "sorted_classes": sorted_classes,
        "sorted_methods": sorted_methods,
        "class_hot_lines": dict(class_hot_lines),
        "class_map": class_map,
    }



def _build_class_map_ast(source_files, ast_frontends=None):
    class_map = {}
    failed_files = set()
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)
    for fname, lines in source_files.items():
        fe = ast_frontends.get(fname)
        if fe is None:
            failed_files.add(fname)
            continue
        file_failed = False
        for cname, info in fe.class_registry.items():
            cls_node = info.get("node")
            start = getattr(cls_node, "lineno", None)
            end = getattr(cls_node, "end_lineno", None)
            if start is None or end is None:
                file_failed = True
                break
            methods = {}
            for method in info.get("methods", []):
                mstart = method.get("lineno")
                mend = method.get("end_lineno")
                mname = method.get("name")
                if not mname or mstart is None or mend is None:
                    file_failed = True
                    break
                methods[mname] = (mstart, mend)
            if file_failed:
                break
            class_map[(fname, cname)] = {"start": start, "end": end, "methods": methods}
        if file_failed:
            failed_files.add(fname)
            for key in [k for k in class_map if k[0] == fname]:
                class_map.pop(key, None)
    return class_map, failed_files


def _build_class_map(source_files, ast_frontends=None):
    ast_map, failed_files = _build_class_map_ast(source_files, ast_frontends=ast_frontends)
    if failed_files:
        print(f"[WARN] AST parse failed for {len(failed_files)} file(s): {sorted(failed_files)}", file=sys.stderr)
    return ast_map


def _build_source_dependency_order(source_files, class_map, ast_frontends=None):
    """Parse forward() methods to extract the order in which child modules are called.
    Returns a dict: {ClassName: [child_attr_name_in_order, ...]}
    Also returns attr_to_class: {(file, attr): ClassName} from __init__ self.xxx = SomeClass(...)
    """
    module_call_order = {}
    attr_to_class = {}  # (fname, attr_name) -> class_name
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    def _norm_index(idx_expr: str):
        idx_expr = (idx_expr or '').strip()
        if not idx_expr:
            return None
        # AST-only path: parse the textual index as a Python expression and
        # only accept integer or string Constant nodes.
        try:
            tree = ast.parse(idx_expr, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        if isinstance(body, ast.Constant):
            if isinstance(body.value, int):
                return str(body.value)
            if isinstance(body.value, str):
                return "'" + body.value.replace("'", "\\'") + "'"
        return None

    for fname, lines in source_files.items():
        fe = _get_ast_frontend(fname)
        for (f, cname), info in class_map.items():
            if f != fname:
                continue
            # Parse __init__ to find self.attr = ClassName(...) mappings
            if fe:
                for item in fe.get_init_assignments_ast(cname):
                    attr_name = item.get("attr")
                    cls_ref = (item.get("class") or "").split('.')[-1]
                    if attr_name and cls_ref:
                        attr_to_class[(fname, attr_name)] = cls_ref

            # Parse forward() reachable helpers to find self.xxx(...) and self.xxx[idx](...) call order
            call_order = []
            if fe and fe.get_method_lines(cname, "forward"):
                reachable = ["forward"] + sorted(fe.get_reachable_helpers(cname, "forward"))
                for method_name in reachable:
                    for call in fe.get_module_calls(cname, method_name):
                        attr_name = call.get("attr")
                        if attr_name and attr_name not in call_order and attr_name != "forward":
                            call_order.append(attr_name)
            module_call_order[cname] = call_order

    return module_call_order, attr_to_class


def _build_data_dependency_edges(source_files, class_map, module_attrs, class_str_list_attrs=None, dynamic_attrs_per_class=None, ast_frontends=None):
    """Parse forward() methods to extract tensor data dependency edges between child modules.

    For each class, analyzes variable assignments from self.xxx() calls and variable
    usage in subsequent self.yyy() calls to determine actual data flow edges.
    Also handles loop iteration patterns (for x in self.attr) and indexed access
    (self.attr[idx](...)).

    Args:
        source_files: {filename: [lines]}
        class_map: from _build_class_map
        module_attrs: {class_name: {attr_name: class_ref}} - known nn.Module attrs

    Returns:
        Tuple (edges_dict, edge_locs_dict, split_info):
        - edges_dict: {class_name: [(from_attr, to_attr), ...]} - data dependency edges per class
        - edge_locs_dict: {class_name: {(from_attr, to_attr): {"file", "from_line", "to_line", "var"}}}
            evidence for each edge: variable name carrying the data, the line that
            *produced* the variable (LHS = self.from_attr(...)) and the line that
            *consumed* it (self.to_attr(..., var, ...)).
        - split_info: {class_name: {original_attr: [split_name_0, split_name_1, ...]}}
            maps original attr names to their split node names for attrs called >1 time
    """
    class_dep_edges = {}
    class_edge_locs = {}
    class_split_info = {}
    if ast_frontends is None:
        ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    if class_str_list_attrs is None:
        class_str_list_attrs = {}

    # -------------------------------------------------------------------
    # dict 数据流追踪（v7 根因修复核心）：
    # - dict 槽位写入：d['k'] = v
    # - dict 槽位读取：x = d['k'] / d.get('k')
    # - 字面量 dict：d = {'k': v, ...}
    # - key 非字面量时：使用该 dict 当前所有槽位 producers 的并集（保守上界，保证不漏边）
    #
    # 说明：这里追踪的是“张量数据流”，不是 dict 这个对象本身。
    # 因此我们不把 dict 作为 var_producers 的普通变量，而是单独维护槽位。
    # -------------------------------------------------------------------

    # 全局（跨 class）共享 dict 槽位：{(fname, dict_name): {key: (producers, from_line, var_evidence)}}
    shared_dict_slots = defaultdict(dict)

    def _as_str_lit(s: str):
        """AST-only string-literal extraction.  Returns the literal value
        for ``"abc"`` / ``'abc'``, otherwise ``None``."""
        s = (s or '').strip()
        if not s:
            return None
        try:
            tree = ast.parse(s, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        if isinstance(body, ast.Constant) and isinstance(body.value, str):
            return body.value
        return None

    def _dict_union_slots(dslots: dict):
        """将 dict 的所有槽位 producers 合并（key 非字面量时使用）。"""
        merged = set()
        best_line = -1
        best_var = None
        for _k, (ps, pl, pv) in dslots.items():
            if ps:
                merged.update(ps)
                if pl is not None and pl > best_line:
                    best_line = pl
                    best_var = pv
        return merged, best_line if best_line >= 0 else None, best_var

    def _collect_dict_reads(expr: str, dict_slots: dict, fname: str):
        """从表达式中收集 dict 读取产生的 producers。

        AST-only walk.  Recognises:
          * ``d['k']`` / ``d[var]``  → ast.Subscript with ast.Name owner
          * ``d.get('k', ...)``      → ast.Call with ast.Attribute func of '.get'

        返回：
          producers_set, evidence_map(producer_attr -> (var_evidence, from_line))
        """
        prod = set()
        ev_map = {}
        if not expr:
            return prod, ev_map
        # Parse as expression; tolerate top-level commas via tuple wrap.
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            try:
                tree = ast.parse("(" + expr + ")", mode='eval')
            except SyntaxError:
                return prod, ev_map

        def _idx_text(idx_node):
            if isinstance(idx_node, ast.Index):  # py<3.9
                idx_node = idx_node.value
            try:
                return ast.unparse(idx_node).strip()
            except Exception:
                return ""

        for sub in ast.walk(tree):
            # d[k] / d['k']
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                dname = sub.value.id
                idx_node = sub.slice
                if isinstance(idx_node, ast.Index):
                    idx_node = idx_node.value
                key_expr = _idx_text(idx_node)
                dslots = dict_slots.get(dname) or shared_dict_slots.get((fname, dname))
                if not dslots:
                    continue
                if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, str):
                    key_lit = idx_node.value
                else:
                    key_lit = _as_str_lit(key_expr)
                if key_lit is not None and key_lit in dslots:
                    ps, pl, pv = dslots[key_lit]
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}['{key_lit}']", pl)
                else:
                    ps, pl, pv = _dict_union_slots(dslots)
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}[{key_expr}]", pl)
            # d.get('k', ...)
            elif (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Name)
                and sub.args
            ):
                dname = sub.func.value.id
                key_node = sub.args[0]
                key_expr = _idx_text(key_node)
                dslots = dict_slots.get(dname) or shared_dict_slots.get((fname, dname))
                if not dslots:
                    continue
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    key_lit = key_node.value
                else:
                    key_lit = _as_str_lit(key_expr)
                if key_lit is not None and key_lit in dslots:
                    ps, pl, pv = dslots[key_lit]
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}.get('{key_lit}')", pl)
                else:
                    ps, pl, pv = _dict_union_slots(dslots)
                    prod.update(ps)
                    for p in ps:
                        prev = ev_map.get(p)
                        if prev is None or (pl is not None and pl > prev[1]):
                            ev_map[p] = (f"{dname}.get({key_expr})", pl)

        return prod, ev_map

    def _collect_expr_producers(expr: str, var_producers: dict, dict_slots: dict, fname: str):
        """统一收集一个表达式的 producers（普通变量 + dict reads）。

        AST-only walk over ``ast.Name`` nodes for variable references,
        plus delegation to ``_collect_dict_reads`` for dict-based flow.
        """
        producers = set()
        best_line = None
        best_var = None

        # 普通变量引用：parse expression and walk Name nodes.
        if expr and var_producers:
            tree = None
            try:
                tree = ast.parse(expr, mode='eval')
            except SyntaxError:
                try:
                    tree = ast.parse("(" + expr + ")", mode='eval')
                except SyntaxError:
                    tree = None
            if tree is not None:
                seen_vars = set()
                for sub in ast.walk(tree):
                    if isinstance(sub, ast.Name):
                        var = sub.id
                        if var in seen_vars:
                            continue
                        seen_vars.add(var)
                        info = var_producers.get(var)
                        if not info:
                            continue
                        ps, ploc = info
                        if ps:
                            producers.update(ps)
                            if ploc is not None and (best_line is None or ploc > best_line):
                                best_line = ploc
                                best_var = var

        # dict 读取
        dps, dev = _collect_dict_reads(expr, dict_slots, fname)
        if dps:
            producers.update(dps)
            # 证据取最新的一条
            for _p, (vname, pl) in dev.items():
                if pl is not None and (best_line is None or pl > best_line):
                    best_line = pl
                    best_var = vname

        return producers, best_line, best_var

    def _parse_dict_literal_items(content: str):
        """解析 {k: v, ...} 的顶层 key/value 列表（只做静态近似，足以用于 DAG 依赖）。"""
        items = []
        s = content.strip()
        if not s:
            return items

        parts = []
        buf = []
        depth = 0
        in_str = None
        esc = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)

        for p in parts:
            # key : value （只按顶层第一个冒号切分）
            depth = 0
            in_str = None
            esc = False
            split_i = None
            for i, ch in enumerate(p):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == in_str:
                        in_str = None
                    continue
                if ch in ("'", '"'):
                    in_str = ch
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                if ch == ':' and depth == 0:
                    split_i = i
                    break
            if split_i is None:
                continue
            k = p[:split_i].strip()
            v = p[split_i+1:].strip()
            items.append((k, v))
        return items

    for fname, lines in source_files.items():
        for (f, cname), info in class_map.items():
            if f != fname:
                continue
            if cname not in module_attrs:
                continue
            attrs = module_attrs[cname]
            fwd_range = info["methods"].get("forward")
            if not fwd_range:
                continue

            # var_producers: {var_name: (set_of_attr_names_that_produce_it, lineno_of_last_production)}
            var_producers = {}
            # Iter15 tensor-alias: tracks tensor aliases registered via
            # ``setattr(self, <name_expr>, <tensor_var>)`` where ``tensor_var``
            # is an existing tracked variable (not a Module constructor call).
            # The alias name (after f-string expansion) acts like a virtual
            # variable: subsequent ``getattr(self, <name_expr>)`` should resolve
            # to the same producer set as the tensor that was stored.
            #   {alias_name_str: (producer_set, registration_lineno)}
            tensor_alias_producers = {}
            # var_lineage: {var_name: [{var, file, line, text}, ...]}
            # Iter11 varlineage: assignment chain for each tracked variable in
            # forward(). Each step records the variable name, the line of the
            # logical statement, and the raw line text. When a variable is
            # produced from a previous variable (e.g. `b = a + bias`), the
            # lineage of `a` is inherited and the new step appended. This is
            # later attached to dep edges so the front-end can render the
            # complete history (a = ..., b = a + ..., c = self.X(b)).
            var_lineage = {}

            def _line_text(ln):
                if not ln or ln <= 0 or ln > len(lines):
                    return ""
                return lines[ln - 1].rstrip("\n")

            def _record_lineage(target_var, parent_vars, lineno):
                """Append a step (target_var produced at lineno) to the chain.
                Inherit chains from parent_vars (deduped, preserving order).
                """
                if not target_var or target_var == '_':
                    return
                base = []
                seen = set()
                if parent_vars:
                    for pv in parent_vars:
                        if not pv or pv == target_var:
                            continue
                        for step in var_lineage.get(pv, []) or []:
                            key = (step.get("var"), step.get("line"))
                            if key in seen:
                                continue
                            seen.add(key)
                            base.append(step)
                step_text = _line_text(lineno)
                key = (target_var, lineno)
                if key not in seen and step_text:
                    base.append({"var": target_var, "file": fname,
                                 "line": lineno, "text": step_text})
                if base:
                    var_lineage[target_var] = base

            def _vars_in(expr_text):
                """Return tracked var names that textually appear in expr_text
                (AST walk, ``ast.Name`` only), preserving expression order so
                the most relevant carrier (last) is at the end."""
                if not expr_text:
                    return []
                hits = []
                seen = set()
                # Search both in var_lineage (rich chain) and var_producers
                # (module-rooted tracking) so we don't miss variables that have
                # producers but no recorded lineage step yet.
                _candidates = set(var_lineage.keys()) | set(var_producers.keys())
                if not _candidates:
                    return []
                # AST-only path: parse the expression and walk Name nodes in
                # source order.  ``ast.parse`` requires a complete expression;
                # wrap as ``(expr,)`` to tolerate top-level comma-separated
                # arg-strings without changing Name semantics.
                try:
                    tree = ast.parse(expr_text, mode='eval')
                except SyntaxError:
                    try:
                        tree = ast.parse("(" + expr_text + ")", mode='eval')
                    except SyntaxError:
                        return []
                for sub in ast.walk(tree):
                    if isinstance(sub, ast.Name):
                        v = sub.id
                        if not v or v == '_' or v not in _candidates:
                            continue
                        if v not in seen:
                            seen.add(v)
                            hits.append(v)
                return hits

            def _collect_ast_expr_consumers(expr_node, current_attr):
                """AST-first consumer extraction for a call argument expression.

                Returns:
                  (producer_set, evidence_map)
                where evidence_map is producer_attr -> (var_evidence, from_line).
                """
                producers = set()
                ev_map = {}
                if expr_node is None:
                    return producers, ev_map

                def _merge(_ps, _ev):
                    if not _ps:
                        return
                    producers.update(_ps)
                    for _p, _meta in (_ev or {}).items():
                        if _p == current_attr:
                            continue
                        _prev = ev_map.get(_p)
                        _line = _meta[1] if _meta else None
                        if _prev is None or (_line is not None and (_prev[1] is None or _line > _prev[1])):
                            ev_map[_p] = _meta

                for sub in ast.walk(expr_node):
                    if isinstance(sub, ast.Name):
                        _v = sub.id
                        if _v in var_producers:
                            _ps, _pl = var_producers[_v]
                            if _ps:
                                _merge(_ps, {p: (_v, _pl) for p in _ps})
                    elif isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                        _dname = sub.value.id
                        _idx_node = sub.slice
                        if isinstance(_idx_node, ast.Index):
                            _idx_node = _idx_node.value
                        _key_expr = _ast_fe_dd._node_to_text(_idx_node).strip() if _ast_fe_dd else ""
                        dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                        if dslots:
                            _key_lit = _as_str_lit(_key_expr)
                            if _key_lit is not None and _key_lit in dslots:
                                _ps, _pl, _pv = dslots[_key_lit]
                                _merge(_ps, {p: (f"{_dname}['{_key_lit}']", _pl) for p in _ps})
                            else:
                                _ps, _pl, _pv = _dict_union_slots(dslots)
                                _merge(_ps, {p: (f"{_dname}[{_key_expr}]", _pl) for p in _ps})
                    elif isinstance(sub, ast.Call):
                        _nested_attr = _ast_fe_dd._extract_called_self_attr(sub.func) if _ast_fe_dd else None
                        if _nested_attr and _nested_attr in known_attrs and _nested_attr != current_attr:
                            _merge({_nested_attr}, {_nested_attr: ("(nested call)", phys_lineno)})
                            continue
                        if isinstance(sub.func, ast.Attribute):
                            _get_attr = sub.func
                            if isinstance(_get_attr.value, ast.Name) and _get_attr.attr == "get":
                                _dname = _get_attr.value.id
                                _key_expr = _ast_fe_dd._node_to_text(sub.args[0]).strip() if (_ast_fe_dd and sub.args) else ""
                                dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                                if dslots:
                                    _key_lit = _as_str_lit(_key_expr)
                                    if _key_lit is not None and _key_lit in dslots:
                                        _ps, _pl, _pv = dslots[_key_lit]
                                        _merge(_ps, {p: (f"{_dname}.get('{_key_lit}')", _pl) for p in _ps})
                                    else:
                                        _ps, _pl, _pv = _dict_union_slots(dslots)
                                        _merge(_ps, {p: (f"{_dname}.get({_key_expr})", _pl) for p in _ps})
                        if isinstance(sub.func, ast.Call) and isinstance(sub.func.func, ast.Name) and sub.func.func.id == "getattr":
                            _gargs = sub.func.args
                            if len(_gargs) >= 2 and isinstance(_gargs[0], ast.Name) and _gargs[0].id == "self":
                                _name_node = _gargs[1]
                                if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                    _nested_attr = _name_node.value
                                    if _nested_attr in known_attrs and _nested_attr != current_attr:
                                        _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})
                                else:
                                    _name_arg = _ast_fe_dd._node_to_text(_name_node).strip() if _ast_fe_dd else ""
                                    if _name_arg in _loop_var_to_real_attrs:
                                        for _nested_attr in _loop_var_to_real_attrs[_name_arg]:
                                            if _nested_attr != current_attr:
                                                _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})
                                    elif dynamic_setattr_attrs:
                                        for _nested_attr in dynamic_setattr_attrs:
                                            if _nested_attr != current_attr:
                                                _merge({_nested_attr}, {_nested_attr: ("(nested getattr call)", phys_lineno)})

                return producers, ev_map
            # dict_slots: {dict_name: {key: (producers_set, from_line, var_evidence)}}
            # 仅用于追踪“经过 dict 槽位传递”的张量数据流。
            dict_slots = defaultdict(dict)
            # edges: set of (from_attr, to_attr) tuples
            edges = set()
            # edge -> evidence dict
            edge_locs = {}
            # Iter11: when `for name in self.xxx_names:` iterates over a known literal
            # string list, the loop var carries an *attribute name* (string), not a
            # tensor. So `x = getattr(self, name)` resolves to the union of real
            # per-element attrs (e.g. {relation_layer, gift_layer, stay_layer, pos_layer, other_layer}).
            # {loop_var_name: [real_attr_str, ...]}
            _loop_var_to_real_attrs = {}
            # Iter15 tensor-alias: parallel tracker that stores the *raw* string
            # items the loop variable iterates over (before filtering to known
            # Module attrs). Used to expand f-string templates like
            # ``f"{group_name}_out"`` into concrete alias names. Distinct from
            # ``_loop_var_to_real_attrs`` because tensor alias names typically
            # are NOT registered Module attrs (e.g. ``gift_layer_out``).
            _loop_var_to_str_items_local = {}
            _str_list_attrs_for_class = (class_str_list_attrs.get(cname) or {})
            # Track which attrs appear in forward
            known_attrs = set(attrs.keys())
            # Iter11: dynamic attrs (synthesised when setattr name is non-literal) are
            # tracked by the upstream scanner via `dynamic_attrs_per_class[cname]`.
            # The DAG node name is now just the class name (e.g. `GroupTower`), with
            # no `__setattr_` prefix.
            dynamic_setattr_attrs = sorted(
                [a for a in (dynamic_attrs_per_class.get(cname) or set()) if a in known_attrs]
            )

            # container -> [elem_attr,...] (ModuleList/ModuleDict 展开后的子节点)
            # 约定：elem_attr 形如 "layers[0]" / "loss_fns['ctr0']" / "blocks[*]"
            container_to_elems = defaultdict(list)

            def _split_indexed_attr(attr_name):
                """Pure-string split of ``container[idx]`` style attr names.
                Returns (container_name, idx_text) or ``None`` when the attr
                does not look like an indexed reference.  No regex is used —
                only matching brackets at the very end of the string and an
                identifier head for the container."""
                if not attr_name or not attr_name.endswith(']'):
                    return None
                lb = attr_name.find('[')
                if lb <= 0 or lb >= len(attr_name) - 1:
                    return None
                head = attr_name[:lb]
                if not (head.isidentifier()):
                    return None
                inner = attr_name[lb + 1:-1]
                return head, inner

            for a in known_attrs:
                split = _split_indexed_attr(a)
                if not split:
                    continue
                container_to_elems[split[0]].append(a)
            # 稳定排序：优先按 [数字] 排序，否则按字典序
            for cont in list(container_to_elems.keys()):
                elems = container_to_elems[cont]
                num_elems = []
                other = []
                for ea in elems:
                    sp = _split_indexed_attr(ea)
                    if sp and sp[0] == cont and sp[1].isdigit():
                        num_elems.append((int(sp[1]), ea))
                    else:
                        other.append(ea)
                if num_elems and not other:
                    container_to_elems[cont] = [ea for _, ea in sorted(num_elems)]
                else:
                    container_to_elems[cont] = sorted(elems)

            def _norm_index_expr(idx_expr: str):
                idx_expr = (idx_expr or '').strip()
                if not idx_expr:
                    return None
                # AST-only path: only int / str Constant nodes are accepted.
                try:
                    tree = ast.parse(idx_expr, mode='eval')
                except SyntaxError:
                    return None
                body = tree.body
                if isinstance(body, ast.Constant):
                    if isinstance(body.value, int):
                        return str(body.value)
                    if isinstance(body.value, str):
                        return "'" + body.value.replace("'", "\\'") + "'"
                return None

            def _resolve_indexed_attr(container: str, idx_expr: str):
                norm = _norm_index_expr(idx_expr)
                if norm is not None:
                    cand = f"{container}[{norm}]"
                    if cand in known_attrs:
                        return [cand]
                # 动态 index：保守连接到 container 的所有展开子元素；若无展开则退回 container 本身
                if container in container_to_elems and container_to_elems[container]:
                    return list(container_to_elems[container])
                return [container]

            # Collect lines from forward() AND helper methods called from forward()
            # that themselves invoke known attrs (nn.Module submodules).
            # This handles patterns like MultiHeadAttention.forward() calling
            # self.forward_dense() which contains the actual module calls.
            _all_methods = info["methods"]
            _helper_method_ranges = []

            # Iter16: transitive helper-method discovery.
            # forward() -> helper1 -> helper2 -> ... where any node in the chain
            # invokes known attrs is included. This is required to support
            # dispatch patterns like
            #     def _compute_task_tower_outputs(self):
            #         if cond: return self._compute_task_tower_outputs_grouped()
            #         return self._compute_task_tower_outputs_old()
            # where the leaf method is what actually consumes setattr-registered
            # modules. Without transitivity those consumers are invisible and
            # Rule2 fires "缺少输出连线".
            _fwd_text = "\n".join(lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))])
            _method_text_cache = {"forward": _fwd_text}
            for _mname, _mrange in _all_methods.items():
                if _mname in ("forward", "__init__"):
                    continue
                _method_text_cache[_mname] = "\n".join(
                    lines[_mrange[0] - 1: min(_mrange[1], len(lines))])

            # AST cache for helper-method discovery: parse each method body once
            # and reuse for both attr-call detection and self-method-call lookup.
            _method_ast_cache = {}

            def _parse_method_ast(_text: str):
                """Parse method body text via ast.parse; returns the function-body
                AST list or None on failure.  Uses leading-whitespace stripping
                so that a top-level ``def`` parses correctly."""
                if not _text:
                    return None
                if _text in _method_ast_cache:
                    return _method_ast_cache[_text]
                stripped = textwrap.dedent(_text)
                try:
                    tree = ast.parse(stripped)
                except SyntaxError:
                    _method_ast_cache[_text] = None
                    return None
                # Find the FunctionDef body
                body = None
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = n.body
                        break
                if body is None:
                    body = tree.body
                _method_ast_cache[_text] = body
                return body

            def _method_has_attr_call(_mtext: str) -> bool:
                """AST-only: does this method body contain a ``self.<attr>(...)`` or
                ``self.<attr>[...]`` reference for any known attr?  Walks both
                ``ast.Call`` (with self-attr func) and ``ast.Subscript`` whose
                owner is ``self.<attr>``."""
                body = _parse_method_ast(_mtext)
                if body is None:
                    return False
                for top in body:
                    for sub in ast.walk(top):
                        # self.<attr>(...)  : Call → Attribute(value=Name('self'))
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if (
                                isinstance(f, ast.Attribute)
                                and isinstance(f.value, ast.Name)
                                and f.value.id == "self"
                                and f.attr in attrs
                            ):
                                return True
                        # self.<attr>[...]  : Subscript whose value is Attribute(self, attr)
                        if isinstance(sub, ast.Subscript):
                            owner = sub.value
                            if (
                                isinstance(owner, ast.Attribute)
                                and isinstance(owner.value, ast.Name)
                                and owner.value.id == "self"
                                and owner.attr in attrs
                            ):
                                return True
                return False

            def _method_calls_self_method(_text: str, _target_name: str) -> bool:
                """AST-only: does this method body contain a ``self.<_target_name>(...)`` call?"""
                body = _parse_method_ast(_text)
                if body is None:
                    return False
                for top in body:
                    for sub in ast.walk(top):
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if (
                                isinstance(f, ast.Attribute)
                                and isinstance(f.value, ast.Name)
                                and f.value.id == "self"
                                and f.attr == _target_name
                            ):
                                return True
                return False

            _reachable_helpers: "set[str]" = set()
            _seen_methods: "set[str]" = {"forward"}
            _frontier = ["forward"]
            while _frontier:
                _next = []
                for _src in _frontier:
                    _src_text = _method_text_cache.get(_src, "")
                    if not _src_text:
                        continue
                    for _mname in _all_methods:
                        if _mname in _seen_methods or _mname in ("__init__",):
                            continue
                        if not _method_calls_self_method(_src_text, _mname):
                            continue
                        _seen_methods.add(_mname)
                        _next.append(_mname)
                        # Include the helper if it (transitively) contains attr calls.
                        if _method_has_attr_call(_method_text_cache.get(_mname, "")):
                            _reachable_helpers.add(_mname)
                _frontier = _next

            # Second pass: also include intermediate dispatchers that don't
            # directly call attrs but route into a helper that does. Without
            # this, var/loop-var producers set inside the dispatcher are lost.
            # Conservative: include ANY transitively reached helper.
            for _mname in _seen_methods - {"forward"}:
                _reachable_helpers.add(_mname)

            for _mname in sorted(_reachable_helpers):
                _mrange = _all_methods.get(_mname)
                if _mrange is not None:
                    _helper_method_ranges.append(_mrange)

            fwd_lines = lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))]
            logical_lines = _join_logical_lines(fwd_lines, fwd_range[0])
            # Append helper method lines after forward lines
            for _hrange in _helper_method_ranges:
                _hlines = lines[_hrange[0] - 1: min(_hrange[1], len(lines))]
                logical_lines.extend(_join_logical_lines(_hlines, _hrange[0]))

            _ast_stmt_info_by_line = {}
            _ast_var_env = {}
            _ast_alias_env = {}
            _ast_dict_slot_env = {}
            _ast_calls_by_line = {}
            _ast_fe_dd = _get_ast_frontend(fname)
            if _ast_fe_dd:
                _ast_method_names = ["forward"] + sorted(_reachable_helpers)
                for _mn in _ast_method_names:
                    for _stmt_info in (_ast_fe_dd._get_stmt_infos(cname, _mn) or []):
                        _ln = _stmt_info.get("line")
                        if _ln is not None and _ln not in _ast_stmt_info_by_line:
                            _ast_stmt_info_by_line[_ln] = _stmt_info
                    for _call_info in (_ast_fe_dd.get_module_calls(cname, _mn) or []):
                        _ln = _call_info.get("line")
                        if _ln is None:
                            continue
                        _ast_calls_by_line.setdefault(_ln, []).append(_call_info)
                    _mn_var_env = _ast_fe_dd.build_var_env(cname, _mn)
                    for _var_name, _chain in (_mn_var_env or {}).items():
                        if not _chain:
                            continue
                        _ast_var_env.setdefault(_var_name, []).extend(list(_chain))
                    _ast_alias_env.update(_ast_fe_dd.build_alias_env(cname, _mn))
                    _ast_dict_slot_env.update(_ast_fe_dd.build_dict_slot_env(cname, _mn))
                for _ln, _calls in _ast_calls_by_line.items():
                    _calls.sort(key=lambda item: (item.get("col") is None, item.get("col") or 0, item.get("attr") or item.get("name") or ""))

            # --- Call-site splitting pre-pass ---
            # Strategy: only split attrs that would create cycles if unsplit.
            # 1. Do a quick pre-scan to count calls per attr within each method
            # 2. Only mark attrs for splitting if they are called >1 in ANY single method
            # 3. After edge building, validate no cycles; if cycles remain, warn.
            #
            # Additionally, to avoid splitting attrs in mutually exclusive branches
            # (if/else), we only split attrs that actually cause cycles. We detect
            # this by checking if an attr is both a source AND a target in the edge set.
            _method_ranges_for_counting = [(fwd_range[0], fwd_range[1])]
            for _hr in _helper_method_ranges:
                _method_ranges_for_counting.append((_hr[0], _hr[1]))

            _attr_max_calls_in_single_method = defaultdict(int)
            for _mr_start, _mr_end in _method_ranges_for_counting:
                _method_lines = lines[_mr_start - 1: min(_mr_end, len(lines))]
                _method_text = "\n".join(_method_lines)
                _method_call_count = defaultdict(int)

                # AST-only: walk every Call/Subscript/For node in the method body.
                _body = _parse_method_ast(_method_text)
                if not _body:
                    # If method body fails to parse, skip; do NOT fall back to regex.
                    for _ca, _cnt in _method_call_count.items():
                        if _cnt > _attr_max_calls_in_single_method[_ca]:
                            _attr_max_calls_in_single_method[_ca] = _cnt
                    continue

                def _is_self_attr(node, attr_name=None):
                    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"):
                        return False
                    return attr_name is None or node.attr == attr_name

                # ``for`` loops over self.<attr> introduce a loop variable that
                # binds to a Module-instance taken from self.<attr>.  We track
                # only existence of these loop vars (the call-counting prepass
                # below does not need to know loop var names because it counts
                # syntactic call sites, not run-time invocations).
                for top in _body:
                    for sub in ast.walk(top):
                        if isinstance(sub, ast.Call):
                            _func = sub.func
                            # 1) self.<attr>(...)
                            if _is_self_attr(_func) and _func.attr in known_attrs:
                                _method_call_count[_func.attr] += 1
                                continue
                            # 2) getattr(self, '<lit>')(...)
                            if (
                                isinstance(_func, ast.Call)
                                and isinstance(_func.func, ast.Name)
                                and _func.func.id == "getattr"
                                and len(_func.args) >= 2
                                and isinstance(_func.args[0], ast.Name)
                                and _func.args[0].id == "self"
                                and isinstance(_func.args[1], ast.Constant)
                                and isinstance(_func.args[1].value, str)
                                and _func.args[1].value in known_attrs
                            ):
                                _method_call_count[_func.args[1].value] += 1
                                continue
                            # 3) self.<container>[<idx>](...)
                            if (
                                isinstance(_func, ast.Subscript)
                                and isinstance(_func.value, ast.Attribute)
                                and isinstance(_func.value.value, ast.Name)
                                and _func.value.value.id == "self"
                            ):
                                _cont = _func.value.attr
                                if _cont not in known_attrs:
                                    continue
                                idx_node = _func.slice
                                if isinstance(idx_node, ast.Index):  # py<3.9
                                    idx_node = idx_node.value
                                # Literal index?
                                _norm = None
                                if isinstance(idx_node, ast.Constant):
                                    if isinstance(idx_node.value, int):
                                        _norm = str(idx_node.value)
                                    elif isinstance(idx_node.value, str):
                                        _norm = "'" + idx_node.value.replace("'", "\\'") + "'"
                                if _norm is not None:
                                    _cand = f"{_cont}[{_norm}]"
                                    if _cand in known_attrs:
                                        _method_call_count[_cand] += 1
                                    else:
                                        _wild = f"{_cont}[*]"
                                        if _wild in known_attrs:
                                            _method_call_count[_wild] += 1
                                        else:
                                            _method_call_count[_cont] += 1
                                else:
                                    _wild = f"{_cont}[*]"
                                    if _wild in known_attrs:
                                        _method_call_count[_wild] += 1
                                    elif _cont in container_to_elems and container_to_elems[_cont]:
                                        _method_call_count[_cont + "[*]"] += 1
                                    else:
                                        _method_call_count[_cont] += 1
                # Track the max across all methods
                for _ca, _cnt in _method_call_count.items():
                    if _cnt > _attr_max_calls_in_single_method[_ca]:
                        _attr_max_calls_in_single_method[_ca] = _cnt

            # Only attrs called >1 in a single method are CANDIDATES for splitting.
            # We only split attrs that actually participate in cycles (detected after
            # a trial edge-build without splitting). This avoids unnecessary splits
            # for attrs in mutually exclusive if/else branches.
            _multi_call_candidates = {a for a, cnt in _attr_max_calls_in_single_method.items() if cnt > 1}

            # split_node_map populated post-hoc if cycles are detected
            split_node_map = {}

            for phys_lineno, line in logical_lines:
                # v7 修复：剥离行内注释（避免注释中的非平衡括号/字符干扰解析）
                line = _strip_inline_comment(line).strip()
                stripped = line
                if not stripped or stripped.startswith('#'):
                    continue

                _ast_stmt = _ast_stmt_info_by_line.get(phys_lineno)

                def _rhs_last_called_attr(expr: str, expr_node=None):
                    """近似判断 expr 的“输出 producer”：优先取 AST 顶层调用 attr。

                    仅用于 dict 槽位赋值/字面量场景，避免把“输入 producer”误当成“输出 producer”。
                    """
                    def _resolve_call_attrs(_call_node):
                        if not isinstance(_call_node, ast.Call):
                            return []
                        _func = _call_node.func
                        _direct_attr = _ast_fe_dd._extract_called_self_attr(_func) if _ast_fe_dd else None
                        if _direct_attr:
                            if _direct_attr in known_attrs:
                                return [_direct_attr]
                            if "[" in _direct_attr:
                                _base, _idx = _direct_attr.split("[", 1)
                                _idx = _idx.rstrip("]")
                                return [a for a in _resolve_indexed_attr(_base, _idx) if a in known_attrs]
                        if isinstance(_func, ast.Call) and isinstance(_func.func, ast.Name) and _func.func.id == "getattr":
                            if len(_func.args) >= 2 and isinstance(_func.args[0], ast.Name) and _func.args[0].id == "self":
                                _name_node = _func.args[1]
                                if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                    return [_name_node.value] if _name_node.value in known_attrs else []
                                _name_arg = _ast_fe_dd._node_to_text(_name_node).strip() if _ast_fe_dd else ""
                                if _name_arg in _loop_var_to_real_attrs:
                                    return [a for a in _loop_var_to_real_attrs[_name_arg] if a in known_attrs]
                                if dynamic_setattr_attrs:
                                    return [a for a in dynamic_setattr_attrs if a in known_attrs]
                        if isinstance(_func, ast.Name):
                            _name = _func.id
                            if _name in var_producers:
                                _ps, _pl = var_producers[_name]
                                return [a for a in _ps if a in known_attrs]
                        return []

                    _ast_attrs = []
                    if expr_node is None and expr:
                        try:
                            _parsed = ast.parse(expr, mode="eval")
                            expr_node = _parsed.body
                        except Exception:
                            expr_node = None
                    if isinstance(expr_node, ast.Call):
                        _ast_attrs = _resolve_call_attrs(expr_node)
                        if _ast_attrs:
                            if len(_ast_attrs) > 1:
                                _tail = None
                                for _cont, _elems in container_to_elems.items():
                                    for _ea in reversed(_elems):
                                        if _ea in _ast_attrs:
                                            _tail = _ea
                                            break
                                    if _tail:
                                        break
                                return _tail or _ast_attrs[-1]
                            return _ast_attrs[-1]

                    # AST is the authoritative path for ``_rhs_last_called_attr``.
                    # If the expression failed to parse or the AST root is not a
                    # Call, we cannot reliably identify the producer; return None
                    # rather than fall back to regex (per AST-only directive).
                    return None

                # ------------------------------
                # dict 数据流：初始化 / 字面量 / 槽位写入  (AST-only)
                # ------------------------------
                # Parse the joined logical line as a standalone Python module.
                # Failure → skip the dict-flow scan for this line (the regex
                # path used to gracefully no-op too).
                _line_mod = None
                try:
                    _line_mod = ast.parse(line)
                except SyntaxError:
                    _line_mod = None
                _line_stmt = (_line_mod.body[0]
                              if _line_mod and len(_line_mod.body) == 1
                              else None)

                # Pattern: ``var = {}`` or ``var = dict()``  →  initialise empty dict slot.
                if (
                    isinstance(_line_stmt, ast.Assign)
                    and len(_line_stmt.targets) == 1
                    and isinstance(_line_stmt.targets[0], ast.Name)
                ):
                    _val = _line_stmt.value
                    _is_empty_dict = (
                        (isinstance(_val, ast.Dict) and not _val.keys and not _val.values)
                        or (
                            isinstance(_val, ast.Call)
                            and isinstance(_val.func, ast.Name)
                            and _val.func.id == "dict"
                            and not _val.args
                            and not _val.keywords
                        )
                    )
                    if _is_empty_dict:
                        dname = _line_stmt.targets[0].id
                        dict_slots[dname] = {}
                        shared_dict_slots[(fname, dname)] = dict_slots[dname]
                        continue

                # Pattern: ``var = { 'k': expr, ... }``  →  literal dict assignment.
                if (
                    isinstance(_line_stmt, ast.Assign)
                    and len(_line_stmt.targets) == 1
                    and isinstance(_line_stmt.targets[0], ast.Name)
                    and isinstance(_line_stmt.value, ast.Dict)
                    and _line_stmt.value.keys
                ):
                    dname = _line_stmt.targets[0].id
                    for _k_node, _v_node in zip(_line_stmt.value.keys, _line_stmt.value.values):
                        if _k_node is None or _v_node is None:
                            continue
                        # key must be a literal string
                        if not (isinstance(_k_node, ast.Constant) and isinstance(_k_node.value, str)):
                            continue
                        k_lit = _k_node.value
                        try:
                            v_expr = ast.unparse(_v_node)
                        except Exception:
                            v_expr = ""
                        last_attr = _rhs_last_called_attr(v_expr, expr_node=_v_node)
                        if last_attr is not None:
                            dict_slots[dname][k_lit] = ({last_attr}, phys_lineno, f"self.{last_attr}(...)")
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            continue
                        # module ref: {'k': self.xxx}
                        if (
                            isinstance(_v_node, ast.Attribute)
                            and isinstance(_v_node.value, ast.Name)
                            and _v_node.value.id == "self"
                            and _v_node.attr in known_attrs
                        ):
                            a = _v_node.attr
                            dict_slots[dname][k_lit] = ({a}, phys_lineno, f"self.{a}")
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            continue
                        ps, pl, pv = _collect_expr_producers(v_expr, var_producers, dict_slots, fname)
                        if ps:
                            dict_slots[dname][k_lit] = (ps, pl if pl is not None else phys_lineno, pv or v_expr)
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    # value 里可能包含 self.xxx()，这里不 continue

                _ast_dict_write = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "dict_write" else None
                _ast_subscript_assign = None
                if _ast_dict_write:
                    dname = (_ast_dict_write.get("dict_var") or "").strip()
                    slot_key = _ast_dict_write.get("dict_key")
                    slot_key = slot_key if slot_key is not None else '*'
                    rhs = (_ast_dict_write.get("rhs_text") or "").strip()
                    key_expr = repr(slot_key) if slot_key != '*' else '*'
                    _v_rhs_node = _ast_dict_write.get("rhs_node")
                else:
                    # AST-only fallback: ``dname[key_expr] = rhs`` form.
                    if (
                        isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Subscript)
                        and isinstance(_line_stmt.targets[0].value, ast.Name)
                    ):
                        _ast_subscript_assign = _line_stmt
                        dname = _line_stmt.targets[0].value.id
                        _slice = _line_stmt.targets[0].slice
                        try:
                            key_expr = ast.unparse(_slice)
                        except Exception:
                            key_expr = ""
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                        if isinstance(_slice, ast.Constant) and isinstance(_slice.value, str):
                            slot_key = _slice.value
                        else:
                            slot_key = '*'
                        _v_rhs_node = _line_stmt.value
                if _ast_dict_write or _ast_subscript_assign is not None:
                    _ast_slot = _ast_dict_slot_env.get((dname, slot_key))
                    if _ast_slot:
                        _prod_attr, _prod_line = _ast_slot
                        if _prod_attr:
                            dict_slots[dname][slot_key] = ({_prod_attr}, _prod_line if _prod_line is not None else phys_lineno, f"self.{_prod_attr}(...)" )
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    else:
                        last_attr = _rhs_last_called_attr(rhs, expr_node=_v_rhs_node)
                        if last_attr is not None:
                            dict_slots[dname][slot_key] = ({last_attr}, phys_lineno, f"self.{last_attr}(...)" )
                            shared_dict_slots[(fname, dname)] = dict_slots[dname]
                        else:
                            # module ref: d['k'] = self.xxx  (AST check)
                            if (
                                isinstance(_v_rhs_node, ast.Attribute)
                                and isinstance(_v_rhs_node.value, ast.Name)
                                and _v_rhs_node.value.id == "self"
                                and _v_rhs_node.attr in known_attrs
                            ):
                                a = _v_rhs_node.attr
                                dict_slots[dname][slot_key] = ({a}, phys_lineno, f"self.{a}")
                                shared_dict_slots[(fname, dname)] = dict_slots[dname]
                            else:
                                ps, pl, pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                                if ps:
                                    dict_slots[dname][slot_key] = (ps, pl if pl is not None else phys_lineno, pv or rhs)
                                    shared_dict_slots[(fname, dname)] = dict_slots[dname]
                    # RHS 可能包含 self.xxx()，这里不 continue

                # Detect loop iteration over ModuleList/ModuleDict values:
                #   for x in self.attr:
                #   for _, x in enumerate(self.attr):
                #   for k, v in self.dict.items():
                #   for v in self.dict.values():
                # AST-only: the joined logical line by itself is not a complete
                # statement (no body), so we wrap it as ``<for-line>\n    pass``
                # before parsing.
                loop_var = None
                loop_attr = None
                loop_items = None
                _for_stmt = None
                _stripped_line = line.lstrip()
                if _stripped_line.startswith("for ") or _stripped_line.startswith("for\t"):
                    try:
                        _for_mod = ast.parse(_stripped_line.rstrip() + "\n    pass\n")
                        if (
                            _for_mod.body
                            and isinstance(_for_mod.body[0], ast.For)
                        ):
                            _for_stmt = _for_mod.body[0]
                    except SyntaxError:
                        _for_stmt = None
                if _for_stmt is not None:
                    _it = _for_stmt.iter
                    _tgt = _for_stmt.target
                    # Determine the loop variable name (Name or Tuple).
                    _loop_var_candidate = None
                    if isinstance(_tgt, ast.Name):
                        _loop_var_candidate = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and _tgt.elts:
                        # for k, v in ...  -> the SECOND element (when 2-tuple)
                        # for i, x in enumerate(...) -> SECOND element
                        # The original regex picked group(1)=last var of any
                        # leading "<x>," prefix, then last positional name.
                        _last = _tgt.elts[-1]
                        if isinstance(_last, ast.Name):
                            _loop_var_candidate = _last.id
                    if _loop_var_candidate is not None:
                        # Variant 1: ``for x in self.<attr>:`` (also matches
                        # ``for _, x in enumerate(self.<attr>):``)
                        _attr_node = None
                        if (
                            isinstance(_it, ast.Attribute)
                            and isinstance(_it.value, ast.Name)
                            and _it.value.id == "self"
                        ):
                            _attr_node = _it
                        elif (
                            isinstance(_it, ast.Call)
                            and isinstance(_it.func, ast.Name)
                            and _it.func.id == "enumerate"
                            and _it.args
                            and isinstance(_it.args[0], ast.Attribute)
                            and isinstance(_it.args[0].value, ast.Name)
                            and _it.args[0].value.id == "self"
                        ):
                            _attr_node = _it.args[0]
                        if _attr_node is not None:
                            loop_var = _loop_var_candidate
                            loop_attr = _attr_node.attr
                        # Variant 2: ``for k, v in self.<attr>.items():`` /
                        #            ``for v in self.<attr>.values():``
                        if (
                            loop_attr is None
                            and isinstance(_it, ast.Call)
                            and isinstance(_it.func, ast.Attribute)
                            and _it.func.attr in ("items", "values")
                            and isinstance(_it.func.value, ast.Attribute)
                            and isinstance(_it.func.value.value, ast.Name)
                            and _it.func.value.value.id == "self"
                            and not _it.args
                            and not _it.keywords
                        ):
                            loop_var = _loop_var_candidate
                            loop_attr = _it.func.value.attr
                if loop_var and loop_attr and loop_attr in known_attrs:
                    # 若该容器已展开，则 loop_var 视为可能调用任一 elem；并补一条 elem 顺序链
                    if loop_attr in container_to_elems and container_to_elems[loop_attr]:
                        elems = container_to_elems[loop_attr]
                        var_producers[loop_var] = (set(elems), phys_lineno)
                        _record_lineage(loop_var, [], phys_lineno)
                        # 顺序链：e0 -> e1 -> ...（for 循环隐含的顺序数据依赖）
                        if len(elems) > 1:
                            for ii in range(len(elems) - 1):
                                edges.add((elems[ii], elems[ii + 1]))
                                edge_locs.setdefault((elems[ii], elems[ii + 1]), {
                                    "file": fname,
                                    "var": f"{loop_var} (sequential iteration over self.{loop_attr})",
                                    "from_line": phys_lineno,
                                    "to_line": phys_lineno,
                                })
                    else:
                        var_producers[loop_var] = ({loop_attr}, phys_lineno)
                        _record_lineage(loop_var, [], phys_lineno)
                    continue

                # Iter11: detect `for name in self.xxx_names:` where xxx_names is a
                # known literal string list (NOT a Module attr). The loop var carries
                # an attribute *name* (string). Real per-element attrs registered via
                # setattr expansion are looked up later via `getattr(self, name)`.
                if loop_var and loop_attr and loop_attr in _str_list_attrs_for_class:
                    real_items = _str_list_attrs_for_class[loop_attr]
                    # Iter15: always remember the raw string items for f-string
                    # alias name expansion, even if some items aren't Module attrs.
                    _loop_var_to_str_items_local[loop_var] = list(real_items)
                    real_module_attrs = [a for a in real_items if a in known_attrs]
                    if real_module_attrs:
                        _loop_var_to_real_attrs[loop_var] = real_module_attrs
                    continue

                # ------------------------------------------------------------
                # Iter15 tensor-alias: detect `setattr(self, <name_expr>, <tensor_var>)`
                # where the value is an existing tracked variable (NOT a class
                # constructor like ``Foo(...)``). This registers a tensor alias
                # mapping the (resolved) name string to the same producers as
                # the source variable. f-string templates referencing the
                # current loop variable get unrolled into per-iteration alias
                # names. Common pattern (cvr.py:206):
                #     for group_name in self.group_tower_names:
                #         group_tower = getattr(self, group_name)
                #         group_tower_out = group_tower(combine_out)
                #         setattr(self, f"{group_name}_out", group_tower_out)
                # We resolve `f"{group_name}_out"` against the literal string
                # list bound to `group_name` and register tensor aliases
                # `relation_layer_out`, `gift_layer_out`, ... → producer set
                # of `group_tower_out` (which is the union of GroupTower attrs).
                # ------------------------------------------------------------
                m_setattr_tensor = None
                _setattr_call = None
                _setattr_name_node = None
                _setattr_src_var = None
                # AST-only: detect ``setattr(self, <name_expr>, <Name>)``
                # statements where the value is a bare local Name.
                if (
                    _line_stmt is not None
                    and isinstance(_line_stmt, ast.Expr)
                    and isinstance(_line_stmt.value, ast.Call)
                    and isinstance(_line_stmt.value.func, ast.Name)
                    and _line_stmt.value.func.id == "setattr"
                    and len(_line_stmt.value.args) == 3
                    and isinstance(_line_stmt.value.args[0], ast.Name)
                    and _line_stmt.value.args[0].id == "self"
                    and isinstance(_line_stmt.value.args[2], ast.Name)
                ):
                    _setattr_call = _line_stmt.value
                    _setattr_name_node = _setattr_call.args[1]
                    _setattr_src_var = _setattr_call.args[2].id
                    m_setattr_tensor = True
                if m_setattr_tensor:
                    src_var = _setattr_src_var
                    # Only proceed if the value isn't a constructor call
                    # (already handled by the Module setattr scanner).
                    if src_var in var_producers:
                        src_producers, _src_pl = var_producers[src_var]
                        # Resolve the name expression to one or more concrete
                        # alias name strings.
                        # alias_name_to_producers: {alias_name: producer_set}
                        # Default: every alias maps to the full producer set of
                        # the source variable; but f-string templates that are
                        # bound 1:1 to a loop variable iterating over Module
                        # attrs allow per-iteration refinement (each alias maps
                        # to ONLY the corresponding Module).
                        alias_name_to_producers = {}
                        if (
                            isinstance(_setattr_name_node, ast.Constant)
                            and isinstance(_setattr_name_node.value, str)
                        ):
                            alias_name_to_producers[_setattr_name_node.value] = set(src_producers)
                        elif isinstance(_setattr_name_node, ast.JoinedStr):
                            # f-string template: extract referenced variable
                            # names from the FormattedValue parts and the
                            # literal-only template (with each FormattedValue
                            # collapsed to a single placeholder for sub).
                            ref_names = set()
                            for _part in _setattr_name_node.values:
                                if isinstance(_part, ast.FormattedValue):
                                    _vname = None
                                    if isinstance(_part.value, ast.Name):
                                        _vname = _part.value.id
                                    if _vname:
                                        ref_names.add(_vname)
                            resolvable = (
                                len(ref_names) == 1
                                and next(iter(ref_names)) in _loop_var_to_str_items_local
                            )
                            if resolvable:
                                lv = next(iter(ref_names))
                                items = _loop_var_to_str_items_local[lv]
                                # Per-iteration refinement: if every item
                                # is itself a known Module attr AND the
                                # source variable's producers are exactly
                                # the union of those items, bind each
                                # alias to the single corresponding item.
                                items_set = set(items)
                                can_refine = (
                                    all(it in known_attrs for it in items)
                                    and items_set & set(src_producers) == items_set & set(known_attrs)
                                )
                                for it in items:
                                    # Expand the JoinedStr node by substituting
                                    # the loop variable's value for every
                                    # FormattedValue and concatenating the
                                    # literal parts.
                                    _expanded_parts = []
                                    for _part in _setattr_name_node.values:
                                        if isinstance(_part, ast.Constant) and isinstance(_part.value, str):
                                            _expanded_parts.append(_part.value)
                                        elif isinstance(_part, ast.FormattedValue):
                                            # Refining only handles the simple
                                            # ``{loop_var}`` case (with optional
                                            # !s/!r/!a or :spec).  Skip format
                                            # specs and conversions and just
                                            # str() the value.
                                            _expanded_parts.append(str(it))
                                        else:
                                            _expanded_parts.append("")
                                    expanded = "".join(_expanded_parts)
                                    if not expanded:
                                        continue
                                    if can_refine and it in src_producers:
                                        alias_name_to_producers[expanded] = {it}
                                    else:
                                        alias_name_to_producers[expanded] = set(src_producers)
                        else:
                            # Bare Name expression: try to resolve via
                            # _loop_var_to_str_items_local.
                            if isinstance(_setattr_name_node, ast.Name):
                                _name_arg = _setattr_name_node.id
                                if _name_arg in _loop_var_to_str_items_local:
                                    for it in _loop_var_to_str_items_local[_name_arg]:
                                        alias_name_to_producers[it] = set(src_producers)
                        # Register the aliases (union with any existing).
                        # Use the source variable's production line as the
                        # alias production line, so downstream getattr resolves
                        # back to the original Module call site (e.g. line 205
                        # ``group_tower_out = group_tower(combine_out)``)
                        # rather than the setattr line.
                        for an, prods in alias_name_to_producers.items():
                            existing = tensor_alias_producers.get(
                                an, (set(), _src_pl))[0]
                            tensor_alias_producers[an] = (
                                set(existing) | set(prods), _src_pl)
                    # The setattr line itself does not produce or consume a
                    # tensor variable directly; fall through so other passes
                    # (Module setattr) still see it if relevant.

                # Detect module calls (AST-first, regex fallback during migration)
                called_attrs = []
                _ast_line_calls = list(_ast_calls_by_line.get(phys_lineno, [])) if _ast_calls_by_line else []
                if _ast_line_calls:
                    for _call in _ast_line_calls:
                        _kind = _call.get("kind")
                        _start = _call.get("col")
                        _end = _call.get("func_end_col")
                        if _start is None:
                            _start = 0
                        if _end is None:
                            _end = _start
                        if _kind in ("self", "getattr_literal"):
                            _attr = _call.get("attr")
                            if _attr in known_attrs:
                                called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "self_indexed":
                            _cont = _call.get("container_attr")
                            _idx_expr = (_call.get("index_expr") or "").strip()
                            if _cont in known_attrs:
                                for _attr in _resolve_indexed_attr(_cont, _idx_expr):
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "dict_index":
                            _dname = _call.get("dict_name")
                            _key_expr = (_call.get("index_expr") or "").strip()
                            dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                            if dslots:
                                _key_lit = _as_str_lit(_key_expr)
                                if _key_lit is not None and _key_lit in dslots:
                                    _ps, _pl, _pv = dslots[_key_lit]
                                else:
                                    _ps, _pl, _pv = _dict_union_slots(dslots)
                                for _attr in _ps:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "dict_get":
                            _dname = _call.get("dict_name")
                            _key_expr = (_call.get("index_expr") or "").strip()
                            dslots = dict_slots.get(_dname) or shared_dict_slots.get((fname, _dname))
                            if dslots:
                                _key_lit = _as_str_lit(_key_expr)
                                if _key_lit is not None and _key_lit in dslots:
                                    _ps, _pl, _pv = dslots[_key_lit]
                                else:
                                    _ps, _pl, _pv = _dict_union_slots(dslots)
                                for _attr in _ps:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "getattr_dynamic":
                            _name_arg = (_call.get("name_expr") or "").strip()
                            _expr_node = None
                            _call_node = _call.get("node")
                            if (
                                _ast_fe_dd
                                and isinstance(_call_node, ast.Call)
                                and _ast_fe_dd._is_self_getattr_call(_call_node.func)
                            ):
                                _expr_node = _call_node.func.args[1]
                            if _name_arg in _loop_var_to_real_attrs:
                                for _attr in _loop_var_to_real_attrs[_name_arg]:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))
                            elif _expr_node is not None:
                                _matched_attrs = [
                                    _attr for _attr in known_attrs
                                    if _ast_fe_dd._dynamic_attr_expr_matches(_expr_node, _attr)
                                ]
                                for _attr in sorted(_matched_attrs):
                                    called_attrs.append((_attr, _start, _end, _call.get("node")))
                            elif dynamic_setattr_attrs:
                                for _attr in dynamic_setattr_attrs:
                                    called_attrs.append((_attr, _start, _end, _call.get("node")))
                        elif _kind == "name":
                            _name = _call.get("name")
                            if _name and _name not in known_attrs and _name in var_producers:
                                _producers, _ploc = var_producers[_name]
                                for _attr in _producers:
                                    if _attr in known_attrs:
                                        called_attrs.append((_attr, _start, _end, _call.get("node")))

                if not called_attrs:
                    # AST path is authoritative for module-call detection.
                    # No regex fallback — if the AST scan didn't find any calls
                    # we treat this as a non-call line.  The downstream
                    # ``if not called_attrs:`` block below still handles
                    # variable-flow tracking through non-module ops.
                    pass

                if not called_attrs:
                    # No module calls - track variable flow through non-module ops
                    _ast_stmt = _ast_stmt_info_by_line.get(phys_lineno)
                    _ast_lhs = None
                    _ast_rhs = None
                    if _ast_stmt and _ast_stmt.get("kind") in ("assign", "augassign"):
                        _targets = [t for t in (_ast_stmt.get("targets") or []) if t and t != '_']
                        if len(_targets) == 1:
                            _ast_lhs = _targets[0]
                            _ast_rhs = (_ast_stmt.get("rhs_text") or "").strip()
                    if _ast_lhs is not None:
                        lhs = _ast_lhs
                        rhs = _ast_rhs
                    elif (
                        isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Name)
                    ):
                        # AST-only fallback for ``var = expr`` parsed from the
                        # joined logical line (used when _ast_stmt did not
                        # provide an upstream record).
                        lhs = _line_stmt.targets[0].id
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                    else:
                        lhs = None
                        rhs = None
                    # Resolve the rhs AST node for downstream attribute checks.
                    _rhs_ast = None
                    if _ast_stmt:
                        _rhs_ast = _ast_stmt.get("rhs_node")
                    if _rhs_ast is None and isinstance(_line_stmt, (ast.Assign, ast.AugAssign)):
                        _rhs_ast = _line_stmt.value
                    if lhs is not None and rhs is not None:

                        # Special case: ModuleDict/ModuleList module reference
                        #   x = self.container[key]
                        if (
                            isinstance(_rhs_ast, ast.Subscript)
                            and isinstance(_rhs_ast.value, ast.Attribute)
                            and isinstance(_rhs_ast.value.value, ast.Name)
                            and _rhs_ast.value.value.id == "self"
                        ):
                            cont = _rhs_ast.value.attr
                            try:
                                idx_expr = ast.unparse(_rhs_ast.slice)
                            except Exception:
                                idx_expr = ""
                            if cont in known_attrs:
                                resolved = _resolve_indexed_attr(cont, idx_expr)
                                # 这里 resolved 可能是多个 elem（动态 key/index），保守合并
                                var_producers[lhs] = (set([a for a in resolved if a in known_attrs]), phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue

                        # Special case: module reference assignment
                        #   x = getattr(self, 'attr')
                        #   x = getattr(self, name_var)
                        # This is common with setattr-registered modules:
                        #   group_tower = getattr(self, group_name)
                        #   group_tower_out = group_tower(tensor)
                        if (
                            isinstance(_rhs_ast, ast.Call)
                            and isinstance(_rhs_ast.func, ast.Name)
                            and _rhs_ast.func.id == "getattr"
                            and len(_rhs_ast.args) >= 2
                            and isinstance(_rhs_ast.args[0], ast.Name)
                            and _rhs_ast.args[0].id == "self"
                            and isinstance(_rhs_ast.args[1], ast.Constant)
                            and isinstance(_rhs_ast.args[1].value, str)
                        ):
                            a = _rhs_ast.args[1].value
                            if a in known_attrs:
                                var_producers[lhs] = ({a}, phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue
                            # Iter15 tensor-alias: literal name resolves to a
                            # previously-registered tensor alias. The lhs
                            # variable inherits the same producer set as the
                            # tensor stored under that alias, and crucially
                            # the *production line* (so Rule1c sees the real
                            # Module-call site, not the setattr/getattr line).
                            if a in tensor_alias_producers:
                                _alias_prods, _alias_pl = tensor_alias_producers[a]
                                if _alias_prods:
                                    var_producers[lhs] = (set(_alias_prods), _alias_pl)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                        if (
                            isinstance(_rhs_ast, ast.Call)
                            and isinstance(_rhs_ast.func, ast.Name)
                            and _rhs_ast.func.id == "getattr"
                            and len(_rhs_ast.args) >= 2
                            and isinstance(_rhs_ast.args[0], ast.Name)
                            and _rhs_ast.args[0].id == "self"
                            and not (
                                isinstance(_rhs_ast.args[1], ast.Constant)
                                and isinstance(_rhs_ast.args[1].value, str)
                            )
                        ):
                            try:
                                _name_arg = ast.unparse(_rhs_ast.args[1]).strip()
                            except Exception:
                                _name_arg = ""
                            _ast_rhs_node = _rhs_ast
                            # Iter11: prefer precise loop-var resolution to literal str list
                            if _name_arg in _loop_var_to_real_attrs:
                                _resolved = [a for a in _loop_var_to_real_attrs[_name_arg] if a in known_attrs]
                                if _resolved:
                                    var_producers[lhs] = (set(_resolved), phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            if (
                                _ast_fe_dd
                                and isinstance(_ast_rhs_node, ast.Call)
                                and _ast_fe_dd._is_self_getattr_call(_ast_rhs_node)
                            ):
                                _resolved = [
                                    a for a in known_attrs
                                    if _ast_fe_dd._dynamic_attr_expr_matches(_ast_rhs_node.args[1], a)
                                ]
                                if _resolved:
                                    var_producers[lhs] = (set(_resolved), phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            # Iter15 tensor-alias: dynamic name bound to a
                            # known literal string list — union the producers
                            # of every alias name that's been registered.
                            if _name_arg in _loop_var_to_str_items_local:
                                _items = _loop_var_to_str_items_local[_name_arg]
                                _alias_union = set()
                                _alias_max_line = None
                                for _it in _items:
                                    if _it in tensor_alias_producers:
                                        _ap, _apl = tensor_alias_producers[_it]
                                        _alias_union |= _ap
                                        if _apl is not None and (_alias_max_line is None or _apl > _alias_max_line):
                                            _alias_max_line = _apl
                                if _alias_union:
                                    var_producers[lhs] = (_alias_union, _alias_max_line or phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            # Iter15 tensor-alias: dynamic name expression that
                            # references an attribute on a non-tensor object,
                            # e.g. ``getattr(self, f"{cvr_task_conf.group_key}")``
                            # or ``getattr(self, group_key)`` where ``group_key``
                            # iterates over a dict whose keys are the same as
                            # the alias name set. Fall back to unioning ALL
                            # registered tensor aliases when no precise binding
                            # is available.
                            #
                            # Iter16: gate the union by f-string template prefix /
                            # suffix.  ``f"combine_tower_mha_{x.task_name}"`` has a
                            # literal prefix ``combine_tower_mha_`` that does NOT
                            # match any alias name (e.g. ``gift_layer_out``), so
                            # we MUST avoid the all-aliases union for that case.
                            # Without this gate, the producer of ``combine_tower``
                            # in ``_compute_task_tower_outputs_old`` is the union
                            # of all 5 GroupTowers, and ``combine_tower(group_tower_out_fp32)``
                            # creates a fully-connected dependency mesh among the
                            # 5 GroupTowers (Rule3 cycle).  Filter aliases by
                            # plausible-name match (literal prefix and suffix).
                            #
                            # Compatible aliases must:
                            #   - start with the literal prefix
                            #   - end with the literal suffix
                            # When the template is purely ``f"{x}"`` (no literal
                            # prefix/suffix), every alias is compatible (the old
                            # behaviour, which is still correct for genuine alias
                            # lookups like ``getattr(self, group_key)``).
                            _compat_aliases = None
                            if tensor_alias_producers:
                                _name_lit_prefix = ""
                                _name_lit_suffix = ""
                                # AST-only: derive prefix/suffix from the
                                # JoinedStr (f-string) AST node when the
                                # name expression is an f-string.  For pure
                                # ``f"literal"`` (a JoinedStr containing only a
                                # single Constant) we treat the entire literal
                                # as the prefix.  For an unparseable or non-
                                # f-string node, prefix/suffix stay empty.
                                _name_node_for_fstr = (
                                    _rhs_ast.args[1] if _rhs_ast is not None and len(_rhs_ast.args) >= 2 else None
                                )
                                if isinstance(_name_node_for_fstr, ast.JoinedStr):
                                    _values = _name_node_for_fstr.values
                                    if all(
                                        isinstance(_p, ast.Constant) and isinstance(_p.value, str)
                                        for _p in _values
                                    ):
                                        _name_lit_prefix = "".join(_p.value for _p in _values)
                                    else:
                                        _prefix_parts = []
                                        for _p in _values:
                                            if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                                                _prefix_parts.append(_p.value)
                                            else:
                                                break
                                        _suffix_parts = []
                                        for _p in reversed(_values):
                                            if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                                                _suffix_parts.append(_p.value)
                                            else:
                                                break
                                        _name_lit_prefix = "".join(_prefix_parts)
                                        _name_lit_suffix = "".join(reversed(_suffix_parts))
                                _compat_aliases = []
                                for _alias_name in tensor_alias_producers:
                                    if _name_lit_prefix and not _alias_name.startswith(_name_lit_prefix):
                                        continue
                                    if _name_lit_suffix and not _alias_name.endswith(_name_lit_suffix):
                                        continue
                                    _compat_aliases.append(_alias_name)
                            if _compat_aliases:
                                _alias_union = set()
                                _alias_max_line = None
                                for _alias_name in _compat_aliases:
                                    _prods, _pl = tensor_alias_producers[_alias_name]
                                    _alias_union |= _prods
                                    if _pl is not None and (_alias_max_line is None or _pl > _alias_max_line):
                                        _alias_max_line = _pl
                                if _alias_union:
                                    var_producers[lhs] = (_alias_union, _alias_max_line or phys_lineno)
                                    _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                    continue
                            if dynamic_setattr_attrs:
                                var_producers[lhs] = (set(dynamic_setattr_attrs), phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue

                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            # Iter15: when the rhs is a passthrough/transform
                            # of an existing tracked variable (no Module call
                            # on this line), keep the upstream production line
                            # so edge evidence anchors at the original Module
                            # call site rather than the intermediate line.
                            # Only use the upstream line if there's no direct
                            # ``self.<attr>(...)`` call on this rhs.
                            _has_self_call = False
                            if isinstance(_rhs_ast, ast.AST):
                                # AST-only check for ``self.<attr>(...)`` /
                                # ``self.<attr>[...]`` access or
                                # ``getattr(self, <expr>)(...)`` calls
                                # anywhere within the rhs.
                                for _sub in ast.walk(_rhs_ast):
                                    if (
                                        isinstance(_sub, ast.Attribute)
                                        and isinstance(_sub.value, ast.Name)
                                        and _sub.value.id == "self"
                                    ):
                                        _has_self_call = True
                                        break
                                    if (
                                        isinstance(_sub, ast.Call)
                                        and isinstance(_sub.func, ast.Name)
                                        and _sub.func.id == "getattr"
                                        and len(_sub.args) >= 2
                                        and isinstance(_sub.args[0], ast.Name)
                                        and _sub.args[0].id == "self"
                                    ):
                                        _has_self_call = True
                                        break
                            _line_for_lhs = phys_lineno
                            if (not _has_self_call) and _pl is not None:
                                _line_for_lhs = _pl
                            var_producers[lhs] = (rhs_producers, _line_for_lhs)
                            _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                        else:
                            # No tracked module-producer in rhs, but still record
                            # an intermediate variable assignment for lineage if
                            # any tracked var appears on rhs.  This captures
                            # transformations like `b = a + bias` so the front-end
                            # can show the full chain.
                            _parents = _vars_in(rhs)
                            if _parents:
                                _record_lineage(lhs, _parents, phys_lineno)
                    _ast_append = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "append" else None
                    if _ast_append:
                        list_var = (_ast_append.get("target_var") or "").strip()
                        rhs = (_ast_append.get("rhs_text") or "").strip()
                    elif (
                        isinstance(_line_stmt, ast.Expr)
                        and isinstance(_line_stmt.value, ast.Call)
                        and isinstance(_line_stmt.value.func, ast.Attribute)
                        and _line_stmt.value.func.attr == "append"
                        and isinstance(_line_stmt.value.func.value, ast.Name)
                        and len(_line_stmt.value.args) == 1
                    ):
                        # AST-only fallback: ``var.append(rhs)``
                        list_var = _line_stmt.value.func.value.id
                        try:
                            rhs = ast.unparse(_line_stmt.value.args[0])
                        except Exception:
                            rhs = ""
                    else:
                        list_var = None
                        rhs = None
                    if list_var and rhs is not None:
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            existing = var_producers.get(list_var, (set(), phys_lineno))[0]
                            var_producers[list_var] = (existing | rhs_producers, phys_lineno)
                            _record_lineage(list_var, _vars_in(rhs), phys_lineno)
                    _ast_aug = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "augassign" else None
                    if _ast_aug:
                        _aug_targets = [t for t in (_ast_aug.get("targets") or []) if t and t != '_']
                        list_var = _aug_targets[0] if len(_aug_targets) == 1 else None
                        rhs = (_ast_aug.get("rhs_text") or "").strip()
                    elif (
                        isinstance(_line_stmt, ast.AugAssign)
                        and isinstance(_line_stmt.op, ast.Add)
                        and isinstance(_line_stmt.target, ast.Name)
                    ):
                        # AST-only fallback: ``var += rhs``
                        list_var = _line_stmt.target.id
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                    else:
                        list_var = None
                        rhs = None
                    if list_var and rhs is not None:
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            existing = var_producers.get(list_var, (set(), phys_lineno))[0]
                            var_producers[list_var] = (existing | rhs_producers, phys_lineno)
                            _record_lineage(list_var, [list_var] + _vars_in(rhs), phys_lineno)
                    # Handle tuple unpacking: (a, b) = ... or a, b = ...
                    _ast_tuple_targets = []
                    if _ast_stmt and _ast_stmt.get("kind") == "assign":
                        _targets = [v for v in (_ast_stmt.get("targets") or []) if v and v != '_']
                        if len(_targets) > 1:
                            _ast_tuple_targets = list(_targets)
                    # AST-only fallback: detect ``(a, b) = expr`` / ``a, b = expr``
                    _line_tuple_targets = []
                    _line_tuple_rhs = None
                    if (
                        not _ast_tuple_targets
                        and isinstance(_line_stmt, ast.Assign)
                        and len(_line_stmt.targets) == 1
                        and isinstance(_line_stmt.targets[0], ast.Tuple)
                    ):
                        _line_tuple_targets = [
                            elt.id for elt in _line_stmt.targets[0].elts
                            if isinstance(elt, ast.Name)
                        ]
                        if len(_line_tuple_targets) >= 2:
                            try:
                                _line_tuple_rhs = ast.unparse(_line_stmt.value)
                            except Exception:
                                _line_tuple_rhs = ""
                        else:
                            _line_tuple_targets = []
                    if _ast_tuple_targets or _line_tuple_targets:
                        lhs_vars = _ast_tuple_targets or _line_tuple_targets
                        if _ast_tuple_targets:
                            rhs = (_ast_stmt.get("rhs_text") or "")
                        else:
                            rhs = _line_tuple_rhs or ""
                        rhs_producers, _pl, _pv = _collect_expr_producers(rhs, var_producers, dict_slots, fname)
                        if rhs_producers:
                            for v in lhs_vars:
                                v = v.strip()
                                if v and v != '_':
                                    _line_for_v = phys_lineno
                                    _ast_var_entry = None
                                    if _ast_fe_dd is not None:
                                        _ast_var_entry = _ast_fe_dd._lookup_var_env_entry(_ast_var_env, v, phys_lineno)
                                    elif _ast_var_env.get(v):
                                        _chain = _ast_var_env.get(v) or []
                                        if not isinstance(_chain, list):
                                            _chain = [_chain]
                                        for _item in _chain:
                                            if _item and len(_item) > 1 and _item[1] is not None and _item[1] <= phys_lineno:
                                                _ast_var_entry = _item
                                    if _ast_var_entry is not None and len(_ast_var_entry) > 1 and _ast_var_entry[1] is not None:
                                        _line_for_v = _ast_var_entry[1]
                                    var_producers[v] = (rhs_producers, _line_for_v)
                                    _record_lineage(v, _vars_in(rhs), phys_lineno)
                    continue

                if called_attrs:
                    _ast_append = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "append" else None
                    if _ast_append:
                        _list_var = (_ast_append.get("target_var") or "").strip()
                        _rhs_text = (_ast_append.get("rhs_text") or "").strip()
                        if _list_var:
                            _existing = var_producers.get(_list_var, (set(), phys_lineno))[0]
                            _call_producers = {
                                _attr for _attr, _call_start, _call_end, _call_node in called_attrs
                                if _attr in known_attrs
                            }
                            if _call_producers:
                                var_producers[_list_var] = (_existing | _call_producers, phys_lineno)
                                _record_lineage(_list_var, _vars_in(_rhs_text), phys_lineno)

                # For each module call, determine consumed variables
                for attr, call_start, call_end, call_node in called_attrs:
                    consumed_producers = set()
                    consumed_var_for_producer = {}  # producer_attr -> (var_name, prod_line)

                    if call_node is not None:
                        _arg_nodes = list(getattr(call_node, "args", []) or [])
                        _arg_nodes.extend([kw.value for kw in (getattr(call_node, "keywords", []) or []) if getattr(kw, "value", None) is not None])
                        for _arg_node in _arg_nodes:
                            _ps, _ev = _collect_ast_expr_consumers(_arg_node, attr)
                            if _ps:
                                consumed_producers.update(_ps)
                            for p, meta in (_ev or {}).items():
                                prev = consumed_var_for_producer.get(p)
                                if prev is None or (meta[1] is not None and (prev[1] is None or meta[1] > prev[1])):
                                    consumed_var_for_producer[p] = meta
                    # Note: a regex-based fallback path used to handle
                    # ``call_node is None`` here; that branch has been retired
                    # because every call now originates from the AST scan and
                    # always carries a real ``ast.Call`` node.

                    for producer in consumed_producers:
                        if producer != attr:
                            edges.add((producer, attr))
                            ev = consumed_var_for_producer.get(producer, ("?", phys_lineno))
                            # Iter11 varlineage: gather the assignment chain that
                            # carried `producer`'s output into the consumer line.
                            # Steps come from var_lineage[var] for the variable
                            # name we identified. We append the consumer step at
                            # the end so the front-end shows a complete history
                            # ending at the actual call site.
                            _carrier = ev[0]
                            _chain = []
                            if isinstance(_carrier, str) and _carrier in var_lineage:
                                _chain = list(var_lineage.get(_carrier) or [])
                            _consumer_text = _line_text(phys_lineno)
                            if _consumer_text:
                                _consumer_step = {"var": _carrier if isinstance(_carrier, str) else "",
                                                  "file": fname,
                                                  "line": phys_lineno,
                                                  "text": _consumer_text,
                                                  "role": "consumer"}
                                # Avoid duplicating if last chain entry is the
                                # consumer line already.
                                if not _chain or _chain[-1].get("line") != phys_lineno:
                                    _chain.append(_consumer_step)
                            # Store evidence: file, line where var was produced (defines from_attr's output)
                            # and line where it was consumed (this attr's call line)
                            edge_locs.setdefault((producer, attr), {
                                "file": fname,
                                "var": ev[0],
                                "from_line": ev[1],
                                "to_line": phys_lineno,
                                "lineage": _chain,
                            })

                # Handle LHS: what variables this line produces
                # Prefer AST targets / producer_attr when available, with regex retained as fallback.
                _ast_assign_like = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") in ("assign", "augassign") else None
                _ast_lhs_targets = []
                if _ast_assign_like:
                    _ast_lhs_targets = [v for v in (_ast_assign_like.get("targets") or []) if v and v != '_']
                _handled_lhs = False
                if _ast_lhs_targets and called_attrs:
                    _ast_producer_attr = (_ast_assign_like.get("producer_attr") or "").strip() if _ast_assign_like else ""
                    last_calls = set()
                    if _ast_producer_attr:
                        if _ast_producer_attr in known_attrs:
                            last_calls = {_ast_producer_attr}
                        elif "[" in _ast_producer_attr:
                            _base, _idx = _ast_producer_attr.split("[", 1)
                            _idx = _idx.rstrip("]")
                            last_calls = set(a for a in _resolve_indexed_attr(_base, _idx) if a in known_attrs)
                    if not last_calls:
                        last_s, last_e = called_attrs[-1][1], called_attrs[-1][2]
                        last_calls = {a for (a, s, e, _node) in called_attrs if s == last_s and e == last_e}
                        last_calls = last_calls or {called_attrs[-1][0]}
                    if len(last_calls) > 1:
                        tail = None
                        for cont, elems in container_to_elems.items():
                            for ea in reversed(elems):
                                if ea in last_calls:
                                    tail = ea
                                    break
                            if tail:
                                break
                        if tail:
                            last_calls = {tail}
                    for v_clean in _ast_lhs_targets:
                        var_producers[v_clean] = (set(last_calls), phys_lineno)
                        _record_lineage(v_clean, _vars_in(line), phys_lineno)
                    _handled_lhs = True
                if not _handled_lhs:
                    # AST-only LHS handling.
                    # Three target shapes are recognised:
                    #   1) ``var = getattr(self, 'attr')(...)``
                    #   2) ``var = self.attr(...)`` / ``var = self.attr[idx](...)``
                    #   3) ``var = <expr>`` / ``a, b = <expr>`` (general)
                    # In each case, the LHS targets and (for shapes 1/2) the
                    # producer attr come straight from the AST stmt.
                    _ast_assign_stmt = None
                    if isinstance(_line_stmt, ast.Assign):
                        _ast_assign_stmt = _line_stmt
                    _lhs_targets = []
                    _lhs_is_tuple = False
                    if _ast_assign_stmt and len(_ast_assign_stmt.targets) == 1:
                        _t0 = _ast_assign_stmt.targets[0]
                        if isinstance(_t0, ast.Name):
                            _lhs_targets = [_t0.id]
                        elif isinstance(_t0, ast.Tuple):
                            _lhs_targets = [
                                e.id for e in _t0.elts
                                if isinstance(e, ast.Name) and e.id != "_"
                            ]
                            _lhs_is_tuple = True

                    # Shape 1: ``... = getattr(self, 'attr')(...)``
                    _shape_handled = False
                    if (
                        _ast_assign_stmt is not None
                        and isinstance(_ast_assign_stmt.value, ast.Call)
                        and isinstance(_ast_assign_stmt.value.func, ast.Call)
                        and isinstance(_ast_assign_stmt.value.func.func, ast.Name)
                        and _ast_assign_stmt.value.func.func.id == "getattr"
                        and len(_ast_assign_stmt.value.func.args) >= 2
                        and isinstance(_ast_assign_stmt.value.func.args[0], ast.Name)
                        and _ast_assign_stmt.value.func.args[0].id == "self"
                        and isinstance(_ast_assign_stmt.value.func.args[1], ast.Constant)
                        and isinstance(_ast_assign_stmt.value.func.args[1].value, str)
                    ):
                        producing_attr = _ast_assign_stmt.value.func.args[1].value
                        if producing_attr in known_attrs:
                            for v_clean in _lhs_targets:
                                if v_clean and v_clean != "_" and v_clean.isidentifier():
                                    var_producers[v_clean] = ({producing_attr}, phys_lineno)
                                    _record_lineage(v_clean, _vars_in(line), phys_lineno)
                        _shape_handled = True
                        continue

                    # Shape 2: ``... = self.attr(...)`` or
                    #         ``... = self.attr[idx](...)``
                    if (
                        _ast_assign_stmt is not None
                        and isinstance(_ast_assign_stmt.value, ast.Call)
                    ):
                        _call_func = _ast_assign_stmt.value.func
                        _producing_attr_name = None
                        _producing_idx_expr = None
                        if (
                            isinstance(_call_func, ast.Attribute)
                            and isinstance(_call_func.value, ast.Name)
                            and _call_func.value.id == "self"
                        ):
                            _producing_attr_name = _call_func.attr
                        elif (
                            isinstance(_call_func, ast.Subscript)
                            and isinstance(_call_func.value, ast.Attribute)
                            and isinstance(_call_func.value.value, ast.Name)
                            and _call_func.value.value.id == "self"
                        ):
                            _producing_attr_name = _call_func.value.attr
                            try:
                                _producing_idx_expr = ast.unparse(_call_func.slice)
                            except Exception:
                                _producing_idx_expr = ""
                        if _producing_attr_name is not None:
                            if _producing_idx_expr is not None:
                                producing_attrs = _resolve_indexed_attr(
                                    _producing_attr_name, _producing_idx_expr)
                            else:
                                producing_attrs = [_producing_attr_name]
                            producing_attrs = [a for a in producing_attrs if a in known_attrs]
                            if producing_attrs:
                                for v_clean in _lhs_targets:
                                    if v_clean and v_clean != "_" and v_clean.isidentifier():
                                        var_producers[v_clean] = (set(producing_attrs), phys_lineno)
                                        _record_lineage(v_clean, _vars_in(line), phys_lineno)
                            _shape_handled = True

                    # Shape 3: general ``var = ...`` / ``a, b = ...``
                    # (only when shape 2 didn't match).  When ``called_attrs``
                    # is non-empty, the producer is the last-call attr.
                    if not _shape_handled and called_attrs:
                        last_s, last_e = called_attrs[-1][1], called_attrs[-1][2]
                        last_calls = {a for (a, s, e, _node) in called_attrs if s == last_s and e == last_e}
                        last_calls = last_calls or {called_attrs[-1][0]}
                        # 若 last_calls 是 container 的元素集合（例如 loop_var），取"链尾"作为更贴近真实的 producer
                        if len(last_calls) > 1:
                            tail = None
                            for cont, elems in container_to_elems.items():
                                for ea in reversed(elems):
                                    if ea in last_calls:
                                        tail = ea
                                        break
                                if tail:
                                    break
                            if tail:
                                last_calls = {tail}
                        if not _lhs_is_tuple and len(_lhs_targets) == 1:
                            lhs = _lhs_targets[0]
                            if lhs and lhs != "_" and lhs.isidentifier():
                                var_producers[lhs] = (set(last_calls), phys_lineno)
                                _record_lineage(lhs, _vars_in(line), phys_lineno)
                        elif _lhs_is_tuple and _lhs_targets:
                            for v in _lhs_targets:
                                if v and v != "_" and v.isidentifier():
                                    var_producers[v] = (set(last_calls), phys_lineno)
                                    _record_lineage(v, _vars_in(line), phys_lineno)

            # --- Cycle validation (Rule 3): DAG must be acyclic ---
            # Detect cycles using Kahn's algorithm and return cycle-involved nodes.
            def _detect_cycle_nodes(edge_set):
                if not edge_set:
                    return set()
                in_deg = defaultdict(int)
                adj = defaultdict(list)
                all_nodes = set()
                for (u, v) in edge_set:
                    adj[u].append(v)
                    in_deg[v] += 1
                    all_nodes.add(u)
                    all_nodes.add(v)
                queue = [n for n in all_nodes if in_deg[n] == 0]
                visited = 0
                while queue:
                    node = queue.pop()
                    visited += 1
                    for nb in adj[node]:
                        in_deg[nb] -= 1
                        if in_deg[nb] == 0:
                            queue.append(nb)
                if visited < len(all_nodes):
                    return {n for n in all_nodes if in_deg[n] > 0}
                return set()

            def _wildcard_base(attr_name):
                _m = re.match(r'^(\w+)(?:\[\*\])$', attr_name or '')
                if _m:
                    return _m.group(1)
                return attr_name

            _wildcard_bases = {
                _wildcard_base(a) for a in known_attrs
                if isinstance(a, str) and a.endswith('[*]')
            }
            if edges and _wildcard_bases:
                _drop_edges = set()
                _edges_by_base = defaultdict(list)
                for (u, v) in edges:
                    _edges_by_base[(_wildcard_base(u), _wildcard_base(v))].append((u, v))
                for (_ub, _vb), _uv_edges in list(_edges_by_base.items()):
                    if _ub == _vb:
                        continue
                    if _ub not in _wildcard_bases or _vb not in _wildcard_bases:
                        continue
                    _vu_edges = _edges_by_base.get((_vb, _ub), [])
                    if not _vu_edges:
                        continue
                    _best_uv = min(
                        _uv_edges,
                        key=lambda _e: (edge_locs.get(_e, {}) or {}).get('to_line') or (edge_locs.get(_e, {}) or {}).get('from_line') or 10**9,
                    )
                    _best_vu = min(
                        _vu_edges,
                        key=lambda _e: (edge_locs.get(_e, {}) or {}).get('to_line') or (edge_locs.get(_e, {}) or {}).get('from_line') or 10**9,
                    )
                    _uv_line = (edge_locs.get(_best_uv, {}) or {}).get('to_line') or (edge_locs.get(_best_uv, {}) or {}).get('from_line') or 10**9
                    _vu_line = (edge_locs.get(_best_vu, {}) or {}).get('to_line') or (edge_locs.get(_best_vu, {}) or {}).get('from_line') or 10**9
                    if _uv_line <= _vu_line:
                        _drop_edges.update(_vu_edges)
                    else:
                        _drop_edges.update(_uv_edges)
                if _drop_edges:
                    edges = {e for e in edges if e not in _drop_edges}
                    for _e in _drop_edges:
                        edge_locs.pop(_e, None)

            cycle_nodes = _detect_cycle_nodes(edges)
            if cycle_nodes and _multi_call_candidates:
                # Identify which multi-call candidates participate in cycles
                _attrs_to_split = _multi_call_candidates & cycle_nodes
                if _attrs_to_split:
                    # Build split_node_map for cycle-involved attrs
                    split_node_map = {}
                    for a in _attrs_to_split:
                        cnt = _attr_max_calls_in_single_method[a]
                        split_names = [f"{a}#{i}" for i in range(cnt)]
                        split_node_map[a] = split_names

                    # Determine occurrence order for each split attr by collecting
                    # all line numbers where it appears (as source or target) from edge evidence
                    _attr_line_numbers = defaultdict(set)
                    for (u, v), ev in edge_locs.items():
                        if not isinstance(ev, dict):
                            continue
                        if u in _attrs_to_split:
                            _attr_line_numbers[u].add(ev.get("from_line", 0))
                        if v in _attrs_to_split:
                            _attr_line_numbers[v].add(ev.get("to_line", 0))
                    _attr_sorted_lines = {a: sorted(lns) for a, lns in _attr_line_numbers.items()}

                    def _line_to_occurrence(attr, lineno):
                        sorted_lns = _attr_sorted_lines.get(attr, [])
                        if not sorted_lns:
                            return 0
                        cnt = _attr_max_calls_in_single_method[attr]
                        # Map line numbers to occurrence indices by position in sorted order
                        idx = 0
                        for i, ln in enumerate(sorted_lns):
                            if lineno >= ln:
                                idx = i
                        return min(idx, cnt - 1)

                    # Remap edges: split attrs get #N suffix based on line evidence
                    _new_edges = set()
                    _new_edge_locs = {}
                    for (u, v) in edges:
                        ev = edge_locs.get((u, v), {})
                        new_u = u
                        new_v = v
                        if u in _attrs_to_split and isinstance(ev, dict):
                            fl = ev.get("from_line", 0)
                            idx = _line_to_occurrence(u, fl)
                            new_u = f"{u}#{idx}"
                        if v in _attrs_to_split and isinstance(ev, dict):
                            tl = ev.get("to_line", 0)
                            idx = _line_to_occurrence(v, tl)
                            new_v = f"{v}#{idx}"
                        _new_edges.add((new_u, new_v))
                        _new_edge_locs[(new_u, new_v)] = ev
                    edges = _new_edges
                    edge_locs = _new_edge_locs

                    # Re-validate after splitting
                    cycle_nodes_2 = _detect_cycle_nodes(edges)
                    if cycle_nodes_2:
                        print(f"  [WARNING] Cycle still present in {cname} after splitting: {cycle_nodes_2}")
            elif cycle_nodes:
                print(f"  [WARNING] Cycle detected in class {cname}: nodes = {list(cycle_nodes)}")

            class_dep_edges[cname] = list(edges)
            class_edge_locs[cname] = edge_locs
            if split_node_map:
                class_split_info[cname] = dict(split_node_map)

    return class_dep_edges, class_edge_locs, class_split_info


def _find_class_for_line(fname, lineno, class_map):
    for (f, cname), info in class_map.items():
        if f == fname and info["start"] <= lineno <= info["end"]:
            for mname, (ms, me) in info["methods"].items():
                if ms <= lineno <= me:
                    return cname, mname
            return cname, None
    return None, None


def enrich_kernel_modules_with_source(events, gpu_info, src_info):
    """When source code is available, map top kernel HostModules to actual nn.Module class names
    by using stack_traces to find the deepest user-code class in the call chain."""
    if not gpu_info or not src_info:
        return
    class_map = src_info.get("class_map", {})
    if not class_map:
        return

    # Build kernel -> source class mapping using stack_traces
    kernel_source_class = defaultdict(lambda: defaultdict(float))
    for e in events:
        if e.get("cat") != "kernel":
            continue
        kname = e.get("name", "")
        dur = e.get("dur", 0)
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        if not traces:
            continue
        n_traces = max(1, len(traces))
        per_dur = dur / n_traces
        for trace in traces:
            # Find the most specific (innermost) user source class in the stack
            # Stack traces go from innermost (top) to outermost (bottom) typically,
            # but in PyTorch traces they go outermost first. We want the leaf user class.
            best_class = None
            for tl in trace.split("\n"):
                m = re.search(r'File "([^"]+)", line (\d+), in (\w+)', tl)
                if not m:
                    continue
                fpath, lineno_s, func = m.groups()
                if "site-packages" in fpath or "/usr/" in fpath:
                    continue
                fname = os.path.basename(fpath)
                lineno = int(lineno_s)
                cname, mname = _find_class_for_line(fname, lineno, class_map)
                if cname:
                    best_class = cname  # keep overwriting to get the last (leaf) match
            if best_class:
                kernel_source_class[kname][best_class] += per_dur

    # Enrich top_kernel_modules: replace runtime names with source class names where possible
    enriched = {}
    for kname, runtime_modules in gpu_info["top_kernel_modules"].items():
        source_classes = kernel_source_class.get(kname, {})
        if source_classes:
            sorted_src = sorted(source_classes.items(), key=lambda x: -x[1])
            enriched[kname] = sorted_src
        elif runtime_modules:
            enriched[kname] = runtime_modules
        else:
            enriched[kname] = []
    gpu_info["top_kernel_modules"] = enriched


# ---------------------------------------------------------------------------
# Kernel call stack analysis
# ---------------------------------------------------------------------------

def analyze_kernel_call_stacks(events, gpu_info):
    if not gpu_info:
        return {}
    kernel_stacks = defaultdict(lambda: {"stacks": [], "total_dur": 0.0, "count": 0})
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0)
        st = e.get("args", {}).get("stack", {})
        traces = st.get("stack_traces", [])
        kernel_stacks[name]["total_dur"] += dur
        kernel_stacks[name]["count"] += 1
        if traces and len(kernel_stacks[name]["stacks"]) < 3:
            kernel_stacks[name]["stacks"].append(traces[0])

    top_names = [kname for kname, _ in gpu_info["top_kernels"][:20]]
    result = {}
    for kname in top_names:
        info = kernel_stacks.get(kname)
        if info and info["stacks"]:
            result[kname] = {
                "stack": info["stacks"][0],
                "total_dur": info["total_dur"],
                "count": info["count"],
            }
    return result


# ---------------------------------------------------------------------------
# lagrange_torch source code reference
# ---------------------------------------------------------------------------

LAGRANGE_TORCH_REPO = "https://code.byted.org/lagrange/torch"

def extract_lagrange_refs(events, gpu_info, kernel_call_stacks):
    lagrange_kernels = []
    if not gpu_info:
        return lagrange_kernels
    for kname, info in gpu_info["top_kernels"]:
        is_lagrange = "lagrange" in kname.lower()
        stack_info = kernel_call_stacks.get(kname, {})
        stack_text = stack_info.get("stack", "")
        if is_lagrange or (stack_text and "lagrange" in stack_text.lower()):
            op_name = None
            if "lagrange_torch::" in kname or "lagrange::" in kname:
                m = re.search(r"lagrange(?:_torch)?::(\w+)", kname)
                if m:
                    op_name = m.group(1)
            lagrange_kernels.append({
                "kernel_name": kname,
                "dur": info["dur"],
                "count": info["count"],
                "op_name": op_name,
                "repo_url": LAGRANGE_TORCH_REPO,
                "search_url": f"{LAGRANGE_TORCH_REPO}/search?search={op_name}&type=code" if op_name else None,
            })
    return lagrange_kernels


# ---------------------------------------------------------------------------
# Per-thread timeline analysis
# ---------------------------------------------------------------------------

def analyze_per_thread_timeline(events, thread_info, meta):
    thread_names = meta["thread_names"]
    cpu_pid = thread_info["cpu_pid"]
    gpu_pids = thread_info["gpu_pids"]
    thread_roles = thread_info["thread_roles"]

    timelines = {}

    # Main thread timeline
    main_tid = thread_info["main_fb_thread"] or thread_info["step_thread"]
    if main_tid:
        main_events = []
        for e in events:
            if e.get("pid") == main_tid[0] and e.get("tid") == main_tid[1] and e.get("ph") == "X":
                if e.get("cat") in ("python_function", "cpu_op", "user_annotation") and e.get("dur", 0) > 500:
                    p_idx = e.get("args", {}).get("_P", -1)
                    is_top = p_idx == -1 or e.get("dur", 0) > 10000
                    main_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                        "is_top": is_top,
                    })
        main_events.sort(key=lambda x: x["ts"])
        top_events = sorted(main_events, key=lambda x: -x["dur"])[:20]
        tname = thread_names.get(main_tid, "main")
        timelines["main_thread"] = {
            "tid": main_tid,
            "name": tname,
            "total_events": len(main_events),
            "top_events": top_events,
        }

    # GPU ProfilerStep timeline (primary stream)
    primary_stream = thread_info["primary_gpu_stream"]
    if primary_stream:
        gpu_timeline_events = []
        for e in events:
            if e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1] and e.get("ph") == "X":
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
                    gpu_timeline_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        gpu_timeline_events.sort(key=lambda x: x["ts"])
        top_gpu_events = sorted(gpu_timeline_events, key=lambda x: -x["dur"])[:20]
        sname = thread_names.get(primary_stream, f"stream {primary_stream[1]}")
        timelines["gpu_primary_stream"] = {
            "tid": primary_stream,
            "name": sname,
            "total_events": len(gpu_timeline_events),
            "top_events": top_gpu_events,
        }

    # Other GPU streams with significant activity
    for (pid, tid), n_kernels in thread_info["gpu_stream_stats"].items():
        if (pid, tid) == primary_stream:
            continue
        if n_kernels < 10:
            continue
        stream_events = []
        for e in events:
            if e.get("pid") == pid and e.get("tid") == tid and e.get("ph") == "X":
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                    stream_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        if stream_events:
            stream_events.sort(key=lambda x: x["ts"])
            top_stream = sorted(stream_events, key=lambda x: -x["dur"])[:10]
            sname = thread_names.get((pid, tid), f"stream {tid}")
            timelines[f"gpu_stream_{tid}"] = {
                "tid": (pid, tid),
                "name": sname,
                "total_events": len(stream_events),
                "top_events": top_stream,
            }

    # Worker threads
    for (pid, tid), role_info in thread_roles.items():
        if role_info["role"] not in ("worker_thread", "autograd_thread"):
            continue
        worker_events = []
        for e in events:
            if e.get("pid") == pid and e.get("tid") == tid and e.get("ph") == "X":
                if e.get("dur", 0) > 500:
                    worker_events.append({
                        "name": e.get("name", ""),
                        "cat": e.get("cat", ""),
                        "ts": e.get("ts", 0),
                        "dur": e.get("dur", 0),
                    })
        if worker_events:
            worker_events.sort(key=lambda x: x["ts"])
            top_w = sorted(worker_events, key=lambda x: -x["dur"])[:10]
            tname = thread_names.get((pid, tid), f"thread {tid}")
            role = role_info["role"]
            timelines[f"{role}_{tid}"] = {
                "tid": (pid, tid),
                "name": tname,
                "role": role,
                "total_events": len(worker_events),
                "top_events": top_w,
            }

    return timelines


# ---------------------------------------------------------------------------
# Chrome tracing visualization
# ---------------------------------------------------------------------------

def generate_trace_screenshot(trace_file, output_dir):
    screenshot_path = os.path.join(output_dir, "trace_screenshot.png")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright not installed"

    abs_trace = os.path.abspath(trace_file)
    trace_size_mb = os.path.getsize(abs_trace) / (1024 * 1024)
    if trace_size_mb > 200:
        return None, f"trace file too large ({trace_size_mb:.0f}MB) for browser visualization"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto("https://ui.perfetto.dev/", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)
            page.set_input_files('input[type="file"]', abs_trace, timeout=10000)
            page.wait_for_timeout(8000)
            page.screenshot(path=screenshot_path, full_page=False)
            browser.close()
        if os.path.exists(screenshot_path):
            return screenshot_path, None
    except Exception as ex:
        return None, str(ex)
    return None, "screenshot failed"


# ---------------------------------------------------------------------------
# Phase 1: Build parent-child relationships
# ---------------------------------------------------------------------------

def build_parent_index(events):
    for i, event in enumerate(events):
        if "args" not in event:
            event["args"] = {}
        event["args"]["_IDX"] = i


def add_func_call_parent(events):
    thread_indices = defaultdict(list)
    for event in events:
        pid = event.get("pid", -1)
        tid = event.get("tid", -1)
        if pid != -1 and tid != -1 and event.get("ph") == "X":
            thread_indices[(pid, tid)].append(event["args"]["_IDX"])

    for key in thread_indices:
        thread_indices[key].sort(key=lambda i: events[i]["ts"])

    for _, indices in thread_indices.items():
        stack = [(-1, -1e50, 1e50)]
        for idx in indices:
            while stack[-1][0] != -1 and events[idx]["ts"] > stack[-1][2]:
                stack.pop()
            events[idx]["args"]["_P"] = stack[-1][0]
            stack.append(
                (idx, events[idx]["ts"], events[idx]["ts"] + events[idx].get("dur", 0))
            )
    return thread_indices


# ---------------------------------------------------------------------------
# Phase 2: Identify threads and their roles
# ---------------------------------------------------------------------------

def classify_threads(events, meta):
    thread_names = meta["thread_names"]
    process_labels = meta["process_labels"]

    cpu_pid = None
    gpu_pids = set()
    for pid, label in process_labels.items():
        if "CPU" in label:
            cpu_pid = pid
        elif "GPU" in label or "MLU" in label:
            gpu_pids.add(pid)

    # Collect per-thread event stats
    thread_stats = defaultdict(lambda: {"count": 0, "dur": 0.0, "cats": defaultdict(int), "has_modules": False})
    for e in events:
        if e.get("ph") != "X":
            continue
        key = (e.get("pid", -1), e.get("tid", -1))
        ts = thread_stats[key]
        ts["count"] += 1
        ts["dur"] += e.get("dur", 0)
        ts["cats"][e.get("cat", "")] += 1
        if e.get("cat") == "python_function" and e.get("name", "").startswith("nn.Module:"):
            ts["has_modules"] = True

    # Identify main forward/backward thread
    main_fb_thread = None
    for e in events:
        name = e.get("name", "")
        if "forward_backward" in name and e.get("cat") == "python_function":
            main_fb_thread = (e.get("pid"), e.get("tid"))
            break

    # Identify step thread (train_runner step)
    step_thread = None
    for e in events:
        name = e.get("name", "")
        if name.startswith("train_runner.py") and "step" in name and e.get("dur", 0) > 100000:
            step_thread = (e.get("pid"), e.get("tid"))
            break

    # Classify each CPU thread
    thread_roles = {}
    if cpu_pid is not None:
        for (pid, tid), stats in thread_stats.items():
            if pid != cpu_pid:
                continue
            tname = thread_names.get((pid, tid), "")
            role = "unknown"
            if (pid, tid) == main_fb_thread or (pid, tid) == step_thread:
                role = "main_thread"
            elif "pt_autograd" in tname:
                role = "autograd_thread"
            elif stats["count"] > 100 and stats["has_modules"]:
                role = "worker_thread"
            elif stats["count"] > 10:
                role = "worker_thread"
            else:
                role = "other"
            thread_roles[(pid, tid)] = {
                "name": tname,
                "role": role,
                "count": stats["count"],
                "dur": stats["dur"],
            }

    # Find the primary GPU stream (most kernels)
    gpu_stream_stats = {}
    for (pid, tid), stats in thread_stats.items():
        if pid not in gpu_pids:
            continue
        n_kernels = stats["cats"].get("kernel", 0) + stats["cats"].get("gpu_memcpy", 0) + stats["cats"].get("gpu_memset", 0)
        if n_kernels > 0:
            gpu_stream_stats[(pid, tid)] = n_kernels

    primary_gpu_stream = max(gpu_stream_stats, key=gpu_stream_stats.get) if gpu_stream_stats else None

    return {
        "cpu_pid": cpu_pid,
        "gpu_pids": gpu_pids,
        "main_fb_thread": main_fb_thread,
        "step_thread": step_thread,
        "thread_roles": thread_roles,
        "primary_gpu_stream": primary_gpu_stream,
        "gpu_stream_stats": gpu_stream_stats,
    }


# ---------------------------------------------------------------------------
# Phase 3: Training step decomposition
# ---------------------------------------------------------------------------

def analyze_step_decomposition(events, thread_info, profiler_steps=None):
    main_tid = thread_info["main_fb_thread"] or thread_info["step_thread"]
    if not main_tid:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Collect all step/fb/optimize events for multi-step averaging
    step_events = []
    fb_events = []
    optimize_events = []
    for e in events:
        if e.get("pid") != main_tid[0] or e.get("tid") != main_tid[1]:
            continue
        if e.get("cat") != "python_function":
            continue
        name = e.get("name", "")
        if name.startswith("train_runner.py") and "step" in name and e.get("dur", 0) > 100000:
            step_events.append(e)
        if "forward_backward" in name:
            fb_events.append(e)
        if name.startswith("train_runner.py") and "optimize" in name and e.get("dur", 0) > 1000:
            optimize_events.append(e)

    result = {}
    if step_events:
        result["step_dur"] = sum(e.get("dur", 0) for e in step_events) / max(len(step_events), 1)
    if fb_events:
        result["forward_backward_dur"] = sum(e.get("dur", 0) for e in fb_events) / max(len(fb_events), 1)
    if optimize_events:
        result["optimize_dur"] = sum(e.get("dur", 0) for e in optimize_events) / max(len(optimize_events), 1)

    # Compute other time (step - fb - optimize)
    if all(k in result for k in ["step_dur", "forward_backward_dur", "optimize_dur"]):
        result["other_dur"] = result["step_dur"] - result["forward_backward_dur"] - result["optimize_dur"]

    return result


# ---------------------------------------------------------------------------
# Phase 4: Module analysis (per-thread aware)
# ---------------------------------------------------------------------------

def find_module_parent(event, events):
    p_idx = event["args"].get("_P", -1)
    while 0 <= p_idx < len(events):
        parent = events[p_idx]
        pname = parent.get("name", "")
        if parent.get("cat") == "python_function" and pname.startswith("nn.Module:"):
            return pname, p_idx
        p_idx = parent.get("args", {}).get("_P", -1)
    return None, -1


def _clip_dur_to_step(event_ts, event_dur, step_start, step_end):
    ev_start = event_ts
    ev_end = event_ts + event_dur
    clipped_start = max(ev_start, step_start)
    clipped_end = min(ev_end, step_end)
    if clipped_end <= clipped_start:
        return 0.0
    return clipped_end - clipped_start


def _find_step_for_event(event_ts, profiler_steps):
    for s in profiler_steps:
        if s["ts"] <= event_ts < s["end"]:
            return s
    return None


def analyze_modules_by_thread(events, thread_info, step_dur_us, profiler_steps=None):
    cpu_pid = thread_info["cpu_pid"]
    thread_roles = thread_info["thread_roles"]
    num_steps = len(profiler_steps) if profiler_steps else 1

    per_thread_modules = defaultdict(list)
    all_modules = []

    for e in events:
        if e.get("cat") != "python_function" or not e.get("name", "").startswith("nn.Module:"):
            continue
        pid, tid = e.get("pid", -1), e.get("tid", -1)
        if pid != cpu_pid:
            continue
        key = (pid, tid)
        per_thread_modules[key].append(e)
        all_modules.append(e)

    # Classify modules by thread role
    main_thread_modules = []
    worker_modules = defaultdict(list)

    for (pid, tid), mods in per_thread_modules.items():
        role_info = thread_roles.get((pid, tid), {})
        role = role_info.get("role", "unknown")
        if role == "main_thread":
            main_thread_modules.extend(mods)
        elif role in ("worker_thread",):
            worker_modules[(pid, tid)].extend(mods)

    # Build global module tree from all modules
    # Duration is clipped to step boundaries and averaged across steps
    module_children = defaultdict(set)
    module_parent_map = {}
    module_durations = defaultdict(float)  # sum of clipped durations across all steps
    module_call_counts = defaultdict(int)
    module_thread_map = defaultdict(set)

    for m in all_modules:
        name = m.get("name", "")
        raw_dur = m.get("dur", 0)
        m_ts = m.get("ts", 0)

        # Clip to step boundary if profiler_steps provided
        if profiler_steps:
            step = _find_step_for_event(m_ts, profiler_steps)
            if step:
                clipped = _clip_dur_to_step(m_ts, raw_dur, step["ts"], step["end"])
            else:
                clipped = 0.0
        else:
            clipped = raw_dur

        module_durations[name] += clipped
        module_call_counts[name] += 1
        module_thread_map[name].add((m.get("pid"), m.get("tid")))
        parent_name, _ = find_module_parent(m, events)
        if parent_name:
            module_children[parent_name].add(name)
            module_parent_map[name] = parent_name
        else:
            module_parent_map[name] = "ROOT"
            module_children["ROOT"].add(name)

    # Average across steps
    if num_steps > 1:
        for name in module_durations:
            module_durations[name] /= num_steps

    depth_map = {}
    def get_depth(name):
        if name in depth_map:
            return depth_map[name]
        parent = module_parent_map.get(name, "ROOT")
        if parent == "ROOT":
            depth_map[name] = 0
        else:
            depth_map[name] = get_depth(parent) + 1
        return depth_map[name]

    for name in module_parent_map:
        get_depth(name)

    max_depth = max(depth_map.values()) if depth_map else 0

    module_exclusive = {}
    for name in module_durations:
        children_dur = sum(module_durations.get(c, 0) for c in module_children.get(name, set()))
        module_exclusive[name] = max(0, module_durations[name] - children_dur)

    # Thread info for each module
    module_thread_info = {}
    for name in module_durations:
        threads = module_thread_map[name]
        roles = set()
        for t in threads:
            ri = thread_roles.get(t, {})
            roles.add(ri.get("role", "unknown"))
        module_thread_info[name] = {"threads": threads, "roles": roles}

    return {
        "module_durations": dict(module_durations),
        "module_exclusive": module_exclusive,
        "module_call_counts": dict(module_call_counts),
        "module_children": {k: list(v) for k, v in module_children.items()},
        "module_parent_map": module_parent_map,
        "depth_map": depth_map,
        "max_depth": max_depth,
        "module_thread_info": module_thread_info,
        "main_thread_count": len(main_thread_modules),
        "worker_thread_count": sum(len(v) for v in worker_modules.values()),
        "num_steps": num_steps,
    }


# ---------------------------------------------------------------------------
# Phase 5: GPU timeline analysis (kernel hotspot + gaps + host mapping)
# ---------------------------------------------------------------------------

def analyze_gpu_timeline(events, thread_info, step_dur_us, profiler_steps=None):
    primary_stream = thread_info["primary_gpu_stream"]
    if not primary_stream:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Get GPU ProfilerStep events for boundary alignment
    gpu_profiler_steps = []
    for e in events:
        if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
            gpu_profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    gpu_profiler_steps.sort(key=lambda x: x["ts"])

    # Get GPU events on primary stream, sorted by time
    gpu_events = []
    for e in events:
        if e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]:
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                gpu_events.append(e)
    gpu_events.sort(key=lambda x: x.get("ts", 0))

    if not gpu_events:
        return None

    # Total active time on this stream (clip to GPU step boundaries, average)
    total_kernel_dur = 0.0
    for e in gpu_events:
        dur = e.get("dur", 0)
        if gpu_profiler_steps:
            step = _find_step_for_event(e.get("ts", 0), gpu_profiler_steps)
            if step:
                dur = _clip_dur_to_step(e["ts"], dur, step["ts"], step["end"])
            else:
                dur = 0.0
        total_kernel_dur += dur
    if num_steps > 1:
        total_kernel_dur /= num_steps

    # GPU Step duration (average)
    if gpu_profiler_steps:
        gpu_step_dur = sum(s["dur"] for s in gpu_profiler_steps) / len(gpu_profiler_steps)
    else:
        gpu_step_dur = 0
        for e in events:
            if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                    and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
                gpu_step_dur = e.get("dur", 0)
                break

    # Compute inter-kernel gaps (average over steps)
    gaps = []
    for i in range(1, len(gpu_events)):
        end_prev = gpu_events[i-1]["ts"] + gpu_events[i-1].get("dur", 0)
        start_curr = gpu_events[i]["ts"]
        gap = start_curr - end_prev
        if gap > 0:
            gaps.append({
                "gap_us": gap,
                "before_name": gpu_events[i-1].get("name", ""),
                "before_dur": gpu_events[i-1].get("dur", 0),
                "after_name": gpu_events[i].get("name", ""),
                "after_dur": gpu_events[i].get("dur", 0),
            })
    total_gap = sum(g["gap_us"] for g in gaps)
    if num_steps > 1:
        total_gap /= num_steps
    gaps.sort(key=lambda x: -x["gap_us"])

    # Kernel hotspot: aggregate by name (clip and average)
    kernel_agg = defaultdict(lambda: {"dur": 0.0, "count": 0})
    for e in gpu_events:
        dur = e.get("dur", 0)
        if gpu_profiler_steps:
            step = _find_step_for_event(e.get("ts", 0), gpu_profiler_steps)
            if step:
                dur = _clip_dur_to_step(e["ts"], dur, step["ts"], step["end"])
            else:
                dur = 0.0
        kernel_agg[e.get("name", "")]["dur"] += dur
        kernel_agg[e.get("name", "")]["count"] += 1
    if num_steps > 1:
        for kname in kernel_agg:
            kernel_agg[kname]["dur"] /= num_steps
            kernel_agg[kname]["count"] = kernel_agg[kname]["count"] // num_steps or kernel_agg[kname]["count"]
    top_kernels = sorted(kernel_agg.items(), key=lambda x: -x[1]["dur"])[:30]

    # Map top kernels to host modules via External id
    ext_id_to_cpu_event = {}
    for e in events:
        ext_id = e.get("args", {}).get("External id")
        if ext_id is not None and e.get("cat") == "cpu_op" and e.get("ph") == "X":
            ext_id_to_cpu_event[ext_id] = e

    kernel_host_map = defaultdict(lambda: defaultdict(float))
    for e in gpu_events:
        ext_id = e.get("args", {}).get("External id")
        host_op = ext_id_to_cpu_event.get(ext_id)
        if not host_op:
            continue
        # Trace up to nn.Module
        module_chain = []
        p = host_op["args"].get("_P", -1)
        while 0 <= p < len(events):
            parent = events[p]
            if parent.get("cat") == "python_function" and parent.get("name", "").startswith("nn.Module:"):
                module_chain.append(parent["name"].replace("nn.Module: ", ""))
            p = parent.get("args", {}).get("_P", -1)
        if module_chain:
            leaf_module = module_chain[0]  # nearest module
            kernel_host_map[e.get("name", "")][leaf_module] += e.get("dur", 0)

    # Build top kernel -> host module mapping
    top_kernel_modules = {}
    for kname, _ in top_kernels:
        module_durs = kernel_host_map.get(kname, {})
        if module_durs:
            sorted_modules = sorted(module_durs.items(), key=lambda x: -x[1])
            top_kernel_modules[kname] = sorted_modules
        else:
            top_kernel_modules[kname] = []

    return {
        "primary_stream": primary_stream,
        "gpu_step_dur": gpu_step_dur,
        "total_kernel_dur": total_kernel_dur,
        "total_gap": total_gap,
        "num_kernels": len(gpu_events) // num_steps if num_steps > 1 else len(gpu_events),
        "gpu_utilization": total_kernel_dur / gpu_step_dur * 100 if gpu_step_dur > 0 else 0,
        "top_gaps": gaps[:15],
        "top_kernels": top_kernels,
        "top_kernel_modules": top_kernel_modules,
        "num_gpu_steps": len(gpu_profiler_steps),
    }


# ---------------------------------------------------------------------------
# Phase 6: Device/Host overview
# ---------------------------------------------------------------------------

def analyze_device_host(events, step_dur_us, thread_info, profiler_steps=None):
    cpu_pid = thread_info["cpu_pid"]
    main_thread = thread_info["main_fb_thread"] or thread_info["step_thread"]
    num_steps = len(profiler_steps) if profiler_steps else 1

    host_total = 0.0
    device_total = 0.0
    device_breakdown = defaultdict(float)
    device_op_dur = defaultdict(float)
    host_op_dur = defaultdict(float)

    for e in events:
        cat = e.get("cat", "")
        dur = e.get("dur", 0)
        name = e.get("name", "")

        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            device_total += dur
            device_breakdown[cat] += dur
            device_op_dur[name] += dur

        if cat == "cpu_op":
            host_total += dur
            host_op_dur[name] += dur

    # Average across steps
    if num_steps > 1:
        host_total /= num_steps
        device_total /= num_steps
        for k in device_breakdown:
            device_breakdown[k] /= num_steps
        for k in device_op_dur:
            device_op_dur[k] /= num_steps
        for k in host_op_dur:
            host_op_dur[k] /= num_steps

    return {
        "host_total_us": host_total,
        "device_total_us": device_total,
        "device_breakdown": dict(device_breakdown),
        "step_dur_us": step_dur_us,
        "top_device_ops": sorted(device_op_dur.items(), key=lambda x: -x[1])[:20],
        "top_host_ops": sorted(host_op_dur.items(), key=lambda x: -x[1])[:20],
    }


# ---------------------------------------------------------------------------
# Phase 7: Worker thread analysis
# ---------------------------------------------------------------------------

def analyze_worker_threads(events, thread_info, meta):
    cpu_pid = thread_info["cpu_pid"]
    thread_roles = thread_info["thread_roles"]
    thread_names = meta["thread_names"]

    workers = []
    for (pid, tid), role_info in thread_roles.items():
        if role_info["role"] not in ("worker_thread", "autograd_thread"):
            continue
        # Get top-level events on this thread
        thread_events = [e for e in events
                         if e.get("pid") == pid and e.get("tid") == tid
                         and e.get("ph") == "X" and e.get("dur", 0) > 1000]
        thread_events.sort(key=lambda x: -x.get("dur", 0))

        # Find nn.Module events
        modules = [e for e in events
                   if e.get("pid") == pid and e.get("tid") == tid
                   and e.get("cat") == "python_function"
                   and e.get("name", "").startswith("nn.Module:")]
        module_names = sorted(set(e.get("name", "").replace("nn.Module: ", "") for e in modules))

        # Find top python_function calls (not nn.Module)
        top_funcs = []
        for e in thread_events[:10]:
            name = e.get("name", "")
            if e.get("cat") == "python_function" and not name.startswith("nn.Module:"):
                # Skip threading boilerplate
                if any(skip in name for skip in ["threading.py", "concurrent/futures"]):
                    continue
                top_funcs.append({"name": name, "dur": e.get("dur", 0)})

        workers.append({
            "pid": pid,
            "tid": tid,
            "name": thread_names.get((pid, tid), ""),
            "role": role_info["role"],
            "event_count": role_info["count"],
            "total_dur": role_info["dur"],
            "modules": module_names,
            "top_funcs": top_funcs[:5],
        })

    workers.sort(key=lambda x: -x["total_dur"])
    return workers


# ---------------------------------------------------------------------------
# Phase 8: Communication/Compute overlap analysis
# ---------------------------------------------------------------------------

def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _intervals_total(merged):
    return sum(e - s for s, e in merged)


def _intervals_overlap(a_merged, b_merged):
    total = 0.0
    j = 0
    for a_s, a_e in a_merged:
        while j < len(b_merged) and b_merged[j][1] <= a_s:
            j += 1
        k = j
        while k < len(b_merged) and b_merged[k][0] < a_e:
            ov_s = max(a_s, b_merged[k][0])
            ov_e = min(a_e, b_merged[k][1])
            if ov_e > ov_s:
                total += ov_e - ov_s
            k += 1
    return total


def analyze_comm_compute_overlap(events, thread_info, profiler_steps=None):
    primary_stream = thread_info["primary_gpu_stream"]
    gpu_pids = thread_info["gpu_pids"]
    if not primary_stream or not gpu_pids:
        return None

    num_steps = len(profiler_steps) if profiler_steps else 1

    # Collect GPU ProfilerStep boundaries on the primary stream
    gpu_profiler_steps = []
    for e in events:
        if (e.get("cat") == "gpu_user_annotation" and "ProfilerStep" in e.get("name", "")
                and e.get("pid") == primary_stream[0] and e.get("tid") == primary_stream[1]):
            gpu_profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    gpu_profiler_steps.sort(key=lambda x: x["ts"])

    # Determine time bounds per step: use CPU profiler_steps as the universal time window
    if profiler_steps:
        step_bounds = [(s["ts"], s["end"]) for s in profiler_steps]
    elif gpu_profiler_steps:
        step_bounds = [(s["ts"], s["end"]) for s in gpu_profiler_steps]
    else:
        return None

    # Classify GPU streams
    primary_key = (primary_stream[0], primary_stream[1])
    comm_stream_keys = set()
    memcpy_stream_keys = set()
    for (pid, tid), n in thread_info["gpu_stream_stats"].items():
        if (pid, tid) == primary_key:
            continue
        if pid not in gpu_pids:
            continue
        has_nccl = False
        has_memcpy = False
        for e in events:
            if e.get("pid") != pid or e.get("tid") != tid or e.get("ph") != "X":
                continue
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
                has_nccl = True
            if e.get("cat") == "gpu_memcpy":
                has_memcpy = True
            if has_nccl:
                break
        if has_nccl:
            comm_stream_keys.add((pid, tid))
        elif has_memcpy:
            memcpy_stream_keys.add((pid, tid))

    # Pre-collect relevant events once (avoid iterating all events per step)
    compute_events = []
    comm_events = []
    memcpy_events = []
    for e in events:
        if e.get("ph") != "X":
            continue
        pid, tid = e.get("pid", -1), e.get("tid", -1)
        cat = e.get("cat", "")
        if (pid, tid) == primary_key and cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            compute_events.append(e)
        elif (pid, tid) in comm_stream_keys and cat == "kernel":
            comm_events.append(e)
        elif (pid, tid) in memcpy_stream_keys and cat in ("gpu_memcpy", "gpu_memset"):
            memcpy_events.append(e)

    compute_events.sort(key=lambda e: e["ts"])
    comm_events.sort(key=lambda e: e["ts"])
    memcpy_events.sort(key=lambda e: e["ts"])

    # Per-step analysis
    per_step_results = []
    for step_start, step_end in step_bounds:
        def _clip_events_to_step(ev_list):
            ivs = []
            for e in ev_list:
                ts = e["ts"]
                if ts >= step_end:
                    break
                dur = e.get("dur", 0)
                cs = max(ts, step_start)
                ce = min(ts + dur, step_end)
                if ce > cs:
                    ivs.append((cs, ce))
            return ivs

        compute_ivs = _clip_events_to_step(compute_events)
        compute_merged = _merge_intervals(compute_ivs)

        comm_ivs = []
        comm_kernel_names = defaultdict(float)
        for e in comm_events:
            ts = e["ts"]
            if ts >= step_end:
                break
            dur = e.get("dur", 0)
            cs = max(ts, step_start)
            ce = min(ts + dur, step_end)
            if ce > cs:
                comm_ivs.append((cs, ce))
                comm_kernel_names[e.get("name", "")] += ce - cs
        comm_merged = _merge_intervals(comm_ivs)

        memcpy_ivs = _clip_events_to_step(memcpy_events)
        memcpy_merged = _merge_intervals(memcpy_ivs)

        compute_total = _intervals_total(compute_merged)
        comm_total = _intervals_total(comm_merged)
        memcpy_total = _intervals_total(memcpy_merged)

        comm_overlap = _intervals_overlap(comm_merged, compute_merged)
        memcpy_overlap = _intervals_overlap(memcpy_merged, compute_merged)

        comm_exposed = comm_total - comm_overlap
        memcpy_exposed = memcpy_total - memcpy_overlap

        per_step_results.append({
            "step_dur": step_end - step_start,
            "compute_total": compute_total,
            "comm_total": comm_total,
            "comm_overlap": comm_overlap,
            "comm_exposed": comm_exposed,
            "memcpy_total": memcpy_total,
            "memcpy_overlap": memcpy_overlap,
            "memcpy_exposed": memcpy_exposed,
            "comm_kernel_names": dict(comm_kernel_names),
        })

    # Average across steps
    n = max(len(per_step_results), 1)
    avg = {}
    for key in ("step_dur", "compute_total", "comm_total", "comm_overlap", "comm_exposed",
                "memcpy_total", "memcpy_overlap", "memcpy_exposed"):
        avg[key] = sum(r[key] for r in per_step_results) / n

    # Aggregate comm kernel names across all steps
    agg_comm_names = defaultdict(float)
    for r in per_step_results:
        for kn, dur in r["comm_kernel_names"].items():
            agg_comm_names[kn] += dur
    for kn in agg_comm_names:
        agg_comm_names[kn] /= n
    top_comm_kernels = sorted(agg_comm_names.items(), key=lambda x: -x[1])[:15]

    avg["comm_overlap_pct"] = avg["comm_overlap"] / avg["comm_total"] * 100 if avg["comm_total"] > 0 else 0
    avg["memcpy_overlap_pct"] = avg["memcpy_overlap"] / avg["memcpy_total"] * 100 if avg["memcpy_total"] > 0 else 0
    avg["comm_exposed_pct_of_step"] = avg["comm_exposed"] / avg["step_dur"] * 100 if avg["step_dur"] > 0 else 0
    avg["memcpy_exposed_pct_of_step"] = avg["memcpy_exposed"] / avg["step_dur"] * 100 if avg["step_dur"] > 0 else 0
    avg["num_steps"] = n
    avg["num_comm_streams"] = len(comm_stream_keys)
    avg["num_memcpy_streams"] = len(memcpy_stream_keys)
    avg["top_comm_kernels"] = top_comm_kernels

    return avg


# ===========================================================================
# Printing / Output
# ===========================================================================

def print_header(title):
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_trace_overview(meta, trace_type, step_dur_us, thread_info, step_decomp, num_steps=1):
    print_header("Trace 概览")
    print(f"  Trace 类型:       {'训练 (Training)' if trace_type == 'training' else '推理 (Inference)'}")
    print(f"  设备:             {meta.get('device_name', 'N/A')}")
    if meta.get("world_size", 1) > 1:
        print(f"  分布式:           rank {meta.get('rank', 0)} / world_size {meta.get('world_size', 1)}")
    print(f"  ProfilerStep 数:  {num_steps}")
    print(f"  平均 Step 耗时:   {format_duration(step_dur_us)} (所有数据已按 {num_steps} 步平均)")
    if step_decomp:
        if "forward_backward_dur" in step_decomp:
            print(f"  ├─ forward_backward: {format_duration(step_decomp['forward_backward_dur'])}  ({pct_str(step_decomp['forward_backward_dur'], step_decomp.get('step_dur', step_dur_us))})")
        if "optimize_dur" in step_decomp:
            print(f"  ├─ optimize:         {format_duration(step_decomp['optimize_dur'])}  ({pct_str(step_decomp['optimize_dur'], step_decomp.get('step_dur', step_dur_us))})")
        if "other_dur" in step_decomp:
            print(f"  └─ other:            {format_duration(step_decomp['other_dur'])}  ({pct_str(step_decomp['other_dur'], step_decomp.get('step_dur', step_dur_us))})")
    print()

    # Thread summary
    roles = thread_info["thread_roles"]
    main_count = sum(1 for r in roles.values() if r["role"] == "main_thread")
    worker_count = sum(1 for r in roles.values() if r["role"] == "worker_thread")
    autograd_count = sum(1 for r in roles.values() if r["role"] == "autograd_thread")
    print(f"  CPU 线程:  main={main_count}, worker={worker_count}, autograd={autograd_count}")
    gpu_streams = len(thread_info["gpu_stream_stats"])
    print(f"  GPU 流:    {gpu_streams} 个活跃流 (主流: stream {thread_info['primary_gpu_stream'][1] if thread_info['primary_gpu_stream'] else 'N/A'})")
    print()


def print_thread_overview(thread_info, meta, workers):
    print_header("线程分析")
    thread_names = meta["thread_names"]
    roles = thread_info["thread_roles"]

    # Main thread
    print("  ── 主线程 (forward_backward 执行线程) ──")
    for (pid, tid), info in sorted(roles.items(), key=lambda x: -x[1]["dur"]):
        if info["role"] != "main_thread":
            continue
        print(f"    [{pid},{tid}] {info['name']}  events={info['count']}, dur={format_duration(info['dur'])}")
    print()

    # Autograd thread
    print("  ── Autograd 线程 (反向传播引擎) ──")
    for (pid, tid), info in sorted(roles.items(), key=lambda x: -x[1]["dur"]):
        if info["role"] != "autograd_thread":
            continue
        print(f"    [{pid},{tid}] {info['name']}  events={info['count']}, dur={format_duration(info['dur'])}")
    print()

    # Worker threads
    print("  ── Worker 线程 (并发 plugin/pipeline) ──")
    for w in workers:
        if w["role"] != "worker_thread":
            continue
        mod_str = ", ".join(w["modules"][:5]) if w["modules"] else "无 Module"
        print(f"    [{w['pid']},{w['tid']}] {w['name']}")
        print(f"      events={w['event_count']}, dur={format_duration(w['total_dur'])}")
        print(f"      modules: {mod_str}")
        if w["top_funcs"]:
            for f in w["top_funcs"][:3]:
                print(f"      top_func: {f['name'][:80]} ({format_duration(f['dur'])})")
        print()


def print_module_tree(mod_info, parent="ROOT", indent=0, step_dur_us=1.0, thread_info_map=None):
    children = mod_info["module_children"].get(parent, [])
    children_sorted = sorted(children, key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)
    for name in children_sorted:
        inc_dur = mod_info["module_durations"].get(name, 0)
        pct = inc_dur / step_dur_us * 100 if step_dur_us > 0 else 0
        short = name.replace("nn.Module: ", "")
        prefix = "  " + "│  " * indent + "├─ "
        # Add thread role indicator
        thread_tag = ""
        if thread_info_map:
            ti = thread_info_map.get(name, {})
            roles = ti.get("roles", set())
            if "worker_thread" in roles and "main_thread" not in roles:
                thread_tag = " [worker]"
            elif "autograd_thread" in roles:
                thread_tag = " [autograd]"
        print(f"{prefix}{short}  [{format_duration(inc_dur)}, {pct:.1f}%]{thread_tag}")
        print_module_tree(mod_info, name, indent + 1, step_dur_us, thread_info_map)


def print_module_report(mod_info, step_dur_us, max_level=None):
    print_header("Module 耗时分析 (Host 侧, 每步平均)")

    print(f"  主线程 Module: {mod_info['main_thread_count']} 个事件")
    print(f"  Worker 线程 Module: {mod_info['worker_thread_count']} 个事件")
    print(f"  注意: Worker 线程 Module 与主线程并发执行，不一定构成阻塞")
    if mod_info.get("num_steps", 1) > 1:
        print(f"  数据已按 {mod_info['num_steps']} 步平均")
    print()

    depth_map = mod_info["depth_map"]
    max_depth = mod_info["max_depth"]
    if max_level is not None:
        max_depth = min(max_depth, max_level)

    for level in range(max_depth + 1):
        level_modules = [name for name, d in depth_map.items() if d == level]
        if not level_modules:
            continue
        level_modules.sort(key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)

        print(f"  ── Level {level} Modules ──")
        print(f"  {'Module Name':<45} {'Thread':>8} {'Inclusive':>12} {'Exclusive':>12} {'Incl%':>8} {'Excl%':>8} {'Calls':>6}")
        print(f"  {'─'*45} {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*6}")

        for name in level_modules:
            inc_dur = mod_info["module_durations"].get(name, 0)
            exc_dur = mod_info["module_exclusive"].get(name, 0)
            calls = mod_info["module_call_counts"].get(name, 0)
            inc_pct = (inc_dur / step_dur_us * 100) if step_dur_us > 0 else 0
            exc_pct = (exc_dur / step_dur_us * 100) if step_dur_us > 0 else 0
            short_name = name.replace("nn.Module: ", "")
            ti = mod_info["module_thread_info"].get(name, {})
            roles = ti.get("roles", set())
            if "worker_thread" in roles and "main_thread" not in roles:
                thread_tag = "worker"
            elif "main_thread" in roles:
                thread_tag = "main"
            elif "autograd_thread" in roles:
                thread_tag = "autograd"
            else:
                thread_tag = "other"
            print(
                f"  {short_name:<45} {thread_tag:>8} {format_duration(inc_dur):>12} {format_duration(exc_dur):>12} {inc_pct:>7.2f}% {exc_pct:>7.2f}% {calls:>6}"
            )
        print()


def print_gpu_timeline_report(gpu_info, step_dur_us):
    if not gpu_info:
        print("  GPU 时间线分析不可用")
        return

    print_header("GPU 时间线分析 (Device 侧, 主流 stream {})".format(gpu_info["primary_stream"][1]))

    n_gpu_steps = gpu_info.get("num_gpu_steps", 1)
    print(f"  GPU ProfilerStep 数:  {n_gpu_steps}")
    print(f"  平均 GPU Step 耗时:   {format_duration(gpu_info['gpu_step_dur'])}")
    print(f"  平均 Kernel 活跃时间: {format_duration(gpu_info['total_kernel_dur'])}  ({pct_str(gpu_info['total_kernel_dur'], gpu_info['gpu_step_dur'])})")
    print(f"  平均 Kernel 间隙时间: {format_duration(gpu_info['total_gap'])}  ({pct_str(gpu_info['total_gap'], gpu_info['gpu_step_dur'])})")
    print(f"  GPU 利用率:           {gpu_info['gpu_utilization']:.1f}%")
    print(f"  平均每步 Kernel 数:   {gpu_info['num_kernels']}")
    print()

    # Top kernels with host module mapping
    print("  ── Top 30 热点 Kernel (按总耗时) ──")
    print(f"  {'Kernel 名称':<65} {'耗时':>12} {'占比':>7} {'次数':>5}  Host Module")
    print(f"  {'─'*65} {'─'*12} {'─'*7} {'─'*5}  {'─'*30}")
    for kname, info in gpu_info["top_kernels"]:
        pct = info["dur"] / gpu_info["gpu_step_dur"] * 100 if gpu_info["gpu_step_dur"] > 0 else 0
        modules = gpu_info["top_kernel_modules"].get(kname, [])
        mod_str = modules[0][0] if modules else "N/A"
        print(f"  {kname[:65]:<65} {format_duration(info['dur']):>12} {pct:>6.2f}% {info['count']:>5}  {mod_str}")
    print()

    # Top kernel gaps
    print("  ── Top 15 Kernel 间隙 (GPU idle) ──")
    print(f"  {'间隙时长':>12}  {'前一个 Kernel':<45}  {'后一个 Kernel':<45}")
    print(f"  {'─'*12}  {'─'*45}  {'─'*45}")
    for g in gpu_info["top_gaps"]:
        print(f"  {format_duration(g['gap_us']):>12}  {g['before_name'][:45]:<45}  {g['after_name'][:45]:<45}")
    print()


def print_device_host_report(dh_info):
    print_header("Device/Host 耗时总览 (每步平均)")
    step_dur = dh_info["step_dur_us"]
    host = dh_info["host_total_us"]
    device = dh_info["device_total_us"]

    print(f"  平均 Step 耗时:          {format_duration(step_dur)}")
    print(f"  平均 Host (CPU) 耗时:    {format_duration(host)}  ({pct_str(host, step_dur)})")
    print(f"  平均 Device (GPU) 耗时:  {format_duration(device)}  ({pct_str(device, step_dur)})")
    print(f"  (Host/Device 可能超过 Step 耗时，因为多线程/多流并行)")
    print()

    print("  ── Device 耗时分解 ──")
    for cat, dur in sorted(dh_info["device_breakdown"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {format_duration(dur):>12}  ({pct_str(dur, step_dur)})")
    print()

    print("  ── Top 20 Device 算子 ──")
    print(f"  {'算子名称':<70} {'耗时':>12} {'占比':>8}")
    print(f"  {'─'*70} {'─'*12} {'─'*8}")
    for name, dur in dh_info["top_device_ops"]:
        print(f"  {name[:70]:<70} {format_duration(dur):>12} {pct_str(dur, step_dur):>8}")
    print()

    print("  ── Top 20 Host 算子 ──")
    print(f"  {'算子名称':<70} {'耗时':>12} {'占比':>8}")
    print(f"  {'─'*70} {'─'*12} {'─'*8}")
    for name, dur in dh_info["top_host_ops"]:
        print(f"  {name[:70]:<70} {format_duration(dur):>12} {pct_str(dur, step_dur):>8}")
    print()


def print_comm_compute_overlap(overlap_info, gpu_info):
    if not overlap_info:
        return
    print_header("通信/计算 Overlap 分析 (每步平均)")
    step_dur = overlap_info["step_dur"]
    print(f"  分析步数:           {overlap_info['num_steps']}")
    print(f"  通信流 (NCCL):      {overlap_info['num_comm_streams']} 条")
    print(f"  Memcpy 流:          {overlap_info['num_memcpy_streams']} 条")
    print()

    print("  ── 通信 (NCCL) vs 计算 ──")
    print(f"  通信总时间:         {format_duration(overlap_info['comm_total'])}")
    print(f"  与计算重叠时间:     {format_duration(overlap_info['comm_overlap'])}  ({overlap_info['comm_overlap_pct']:.1f}% 被重叠)")
    print(f"  暴露时间 (开销):    {format_duration(overlap_info['comm_exposed'])}  ({overlap_info['comm_exposed_pct_of_step']:.2f}% of step)")
    print()

    if overlap_info["memcpy_total"] > 0:
        print("  ── Memcpy vs 计算 ──")
        print(f"  Memcpy 总时间:      {format_duration(overlap_info['memcpy_total'])}")
        print(f"  与计算重叠时间:     {format_duration(overlap_info['memcpy_overlap'])}  ({overlap_info['memcpy_overlap_pct']:.1f}% 被重叠)")
        print(f"  暴露时间 (开销):    {format_duration(overlap_info['memcpy_exposed'])}  ({overlap_info['memcpy_exposed_pct_of_step']:.2f}% of step)")
        print()

    total_exposed = overlap_info["comm_exposed"] + overlap_info["memcpy_exposed"]
    total_exposed_pct = total_exposed / step_dur * 100 if step_dur > 0 else 0
    compute_total = overlap_info["compute_total"]
    print("  ── 综合 ──")
    print(f"  主流计算时间:       {format_duration(compute_total)}")
    print(f"  总暴露开销:         {format_duration(total_exposed)}  ({total_exposed_pct:.2f}% of step)")
    if gpu_info and gpu_info.get("gpu_step_dur", 0) > 0:
        effective_busy = compute_total + total_exposed
        print(f"  有效 GPU 忙碌时间:  {format_duration(effective_busy)}  (计算 + 暴露开销)")
    print()

    if overlap_info["top_comm_kernels"]:
        print("  ── Top 通信 Kernel (按耗时) ──")
        print(f"  {'Kernel 名称':<75} {'耗时':>12}")
        print(f"  {'─'*75} {'─'*12}")
        for kname, dur in overlap_info["top_comm_kernels"]:
            print(f"  {kname[:75]:<75} {format_duration(dur):>12}")
        print()


def save_comm_compute_overlap_md(overlap_info, gpu_info, L):
    if not overlap_info:
        return
    L.append("## 通信/计算 Overlap 分析\n")
    step_dur = overlap_info["step_dur"]

    L.append("### 通信 (NCCL) vs 计算\n")
    L.append("| 指标 | 数值 | 说明 |")
    L.append("|------|------|------|")
    L.append(f"| 通信总时间 | {format_duration(overlap_info['comm_total'])} | 所有 NCCL kernel 在通信流上的累计时间 |")
    L.append(f"| 与计算重叠 | {format_duration(overlap_info['comm_overlap'])} | 通信与主流计算同时进行的时间 ({overlap_info['comm_overlap_pct']:.1f}%) |")
    L.append(f"| **暴露时间** | **{format_duration(overlap_info['comm_exposed'])}** | **实际通信开销 ({overlap_info['comm_exposed_pct_of_step']:.2f}% of step)** |")
    L.append("")

    if overlap_info["memcpy_total"] > 0:
        L.append("### Memcpy vs 计算\n")
        L.append("| 指标 | 数值 | 说明 |")
        L.append("|------|------|------|")
        L.append(f"| Memcpy 总时间 | {format_duration(overlap_info['memcpy_total'])} | 非主流上的 memcpy 累计时间 |")
        L.append(f"| 与计算重叠 | {format_duration(overlap_info['memcpy_overlap'])} | ({overlap_info['memcpy_overlap_pct']:.1f}%) |")
        L.append(f"| **暴露时间** | **{format_duration(overlap_info['memcpy_exposed'])}** | **({overlap_info['memcpy_exposed_pct_of_step']:.2f}% of step)** |")
        L.append("")

    total_exposed = overlap_info["comm_exposed"] + overlap_info["memcpy_exposed"]
    total_exposed_pct = total_exposed / step_dur * 100 if step_dur > 0 else 0
    L.append("### 综合\n")
    L.append("| 指标 | 数值 |")
    L.append("|------|------|")
    L.append(f"| 主流计算时间 | {format_duration(overlap_info['compute_total'])} |")
    L.append(f"| **总暴露开销** | **{format_duration(total_exposed)} ({total_exposed_pct:.2f}% of step)** |")
    L.append("")

    if overlap_info["top_comm_kernels"]:
        L.append("### Top 通信 Kernel\n")
        L.append("| Kernel | 每步平均耗时 |")
        L.append("|--------|-------------|")
        for kname, dur in overlap_info["top_comm_kernels"]:
            L.append(f"| {kname[:80]} | {format_duration(dur)} |")
        L.append("")


# ===========================================================================
# Source code hotspot report
# ===========================================================================

def print_source_hotspot_report(src_info, source_files):
    if not src_info:
        return
    print_header("源码热点分析 (Class/Module 粒度, 正向/反向分离)")
    total = src_info["total_kernel_dur"]
    print(f"  Kernel 总耗时 (带 stack_traces): {format_duration(total)}")
    print()

    # Per-class with fwd/bwd
    print("  ── Class (nn.Module) 级别耗时 ──")
    print(f"  {'Class':<45} {'Forward':>12} {'Backward':>12} {'Total':>12} {'占比':>8}")
    print(f"  {'─'*45} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for ckey, phases in src_info["sorted_classes"][:20]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        print(f"  {ckey:<45} {format_duration(fwd):>12} {format_duration(bwd):>12} {format_duration(t):>12} {pct_str(t, total):>8}")
    print()

    # Per-class.method with fwd/bwd
    print("  ── Class.Method 级别耗时 ──")
    print(f"  {'Class.Method':<55} {'Forward':>12} {'Backward':>12} {'Total':>12} {'占比':>8}")
    print(f"  {'─'*55} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for cmkey, phases in src_info["sorted_methods"][:25]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        print(f"  {cmkey:<55} {format_duration(fwd):>12} {format_duration(bwd):>12} {format_duration(t):>12} {pct_str(t, total):>8}")
    print()

    # Top classes with annotated hot code
    for ckey, phases in src_info["sorted_classes"][:5]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        if t < 1000:
            continue
        hot_lines = src_info["class_hot_lines"].get(ckey, [])[:10]
        if not hot_lines:
            continue
        fname = ckey.split(":")[0]
        cname = ckey.split(":")[1]
        print(f"  ── {ckey} (fwd={format_duration(fwd)}, bwd={format_duration(bwd)}) ──")
        lines = source_files.get(fname, [])
        for lineno, dur_total, dur_fwd, dur_bwd, mname in hot_lines:
            content = lines[lineno - 1].rstrip()[:80] if 0 < lineno <= len(lines) else ""
            method_tag = f"[{mname}]" if mname else ""
            phase_tag = ""
            if dur_fwd > 0 and dur_bwd > 0:
                phase_tag = f"fwd={format_duration(dur_fwd)},bwd={format_duration(dur_bwd)}"
            elif dur_fwd > 0:
                phase_tag = f"fwd={format_duration(dur_fwd)}"
            else:
                phase_tag = f"bwd={format_duration(dur_bwd)}"
            print(f"    🔥 {format_duration(dur_total):>10} L{lineno:<5} {method_tag:<12} {content}")
        print()


def save_source_hotspot_markdown(src_info, source_files, L):
    if not src_info:
        return
    total = src_info["total_kernel_dur"]
    L.append("## 源码热点分析 (Class/Module 粒度)\n")
    L.append(f"Kernel 总耗时 (带 stack_traces): {format_duration(total)}\n")

    L.append("### Class (nn.Module) 级别耗时\n")
    L.append("| Class | Forward | Backward | Total | 占比 |")
    L.append("|-------|---------|----------|-------|------|")
    for ckey, phases in src_info["sorted_classes"][:20]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        L.append(f"| {ckey} | {format_duration(fwd)} | {format_duration(bwd)} | {format_duration(t)} | {pct_str(t, total)} |")
    L.append("")

    L.append("### Class.Method 级别耗时\n")
    L.append("| Class.Method | Forward | Backward | Total | 占比 |")
    L.append("|--------------|---------|----------|-------|------|")
    for cmkey, phases in src_info["sorted_methods"][:25]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        L.append(f"| {cmkey} | {format_duration(fwd)} | {format_duration(bwd)} | {format_duration(t)} | {pct_str(t, total)} |")
    L.append("")

    L.append("### 热点代码段\n")
    for ckey, phases in src_info["sorted_classes"][:5]:
        fwd, bwd = phases["fwd"], phases["bwd"]
        t = fwd + bwd
        if t < 1000:
            continue
        hot_lines = src_info["class_hot_lines"].get(ckey, [])[:10]
        if not hot_lines:
            continue
        fname = ckey.split(":")[0]
        L.append(f"**{ckey}** (fwd={format_duration(fwd)}, bwd={format_duration(bwd)})\n")
        L.append("```python")
        lines = source_files.get(fname, [])
        for lineno, dur_total, dur_fwd, dur_bwd, mname in hot_lines:
            content = lines[lineno - 1].rstrip()[:90] if 0 < lineno <= len(lines) else ""
            L.append(f"# 🔥 {format_duration(dur_total)} (L{lineno}) [{mname or ''}]")
            L.append(f"{content}")
        L.append("```\n")
    L.append("")


# ===========================================================================
# Per-thread timeline report
# ===========================================================================

def print_per_thread_timeline(timelines):
    print_header("多线程时间线分析 (按线程独立标注)")
    for tkey, tinfo in timelines.items():
        label = tkey.replace("_", " ").title()
        tid = tinfo["tid"]
        tname = tinfo.get("name", "")
        role = tinfo.get("role", tkey)
        print(f"  ── {label}: {tname} [pid={tid[0]}, tid={tid[1]}] ──")
        print(f"    事件总数: {tinfo['total_events']}")
        print(f"    {'事件名称':<65} {'类别':>12} {'耗时':>12}")
        print(f"    {'─'*65} {'─'*12} {'─'*12}")
        for ev in tinfo["top_events"]:
            print(f"    {ev['name'][:65]:<65} {ev['cat']:>12} {format_duration(ev['dur']):>12}")
        print()


def save_per_thread_timeline_md(timelines, L):
    L.append("## 多线程时间线分析\n")
    L.append("> 各线程独立分析，以主线程和 GPU ProfilerStep 为主。\n")
    for tkey, tinfo in timelines.items():
        label = tkey.replace("_", " ").title()
        tid = tinfo["tid"]
        tname = tinfo.get("name", "")
        L.append(f"### {label}: {tname} [pid={tid[0]}, tid={tid[1]}]\n")
        L.append(f"事件总数: {tinfo['total_events']}\n")
        L.append("| 事件名称 | 类别 | 耗时 |")
        L.append("|----------|------|------|")
        for ev in tinfo["top_events"]:
            L.append(f"| {ev['name'][:80]} | {ev['cat']} | {format_duration(ev['dur'])} |")
        L.append("")


# ===========================================================================
# Kernel call stack report
# ===========================================================================

def print_kernel_call_stacks(kernel_stacks, gpu_info):
    if not kernel_stacks:
        return
    print_header("Kernel 热点调用栈分析")
    for kname, info in list(kernel_stacks.items())[:10]:
        dur = info["total_dur"]
        pct = dur / gpu_info["gpu_step_dur"] * 100 if gpu_info.get("gpu_step_dur", 0) > 0 else 0
        print(f"  ── {kname[:80]} ({format_duration(dur)}, {pct:.2f}%, count={info['count']}) ──")
        stack = info["stack"]
        for line in stack.strip().split("\n")[:12]:
            print(f"    {line.strip()}")
        print()


def save_kernel_call_stacks_md(kernel_stacks, gpu_info, L):
    if not kernel_stacks:
        return
    L.append("## Kernel 热点调用栈\n")
    for kname, info in list(kernel_stacks.items())[:10]:
        dur = info["total_dur"]
        pct = dur / gpu_info["gpu_step_dur"] * 100 if gpu_info.get("gpu_step_dur", 0) > 0 else 0
        L.append(f"### {kname[:80]}\n")
        L.append(f"- 耗时: {format_duration(dur)} ({pct:.2f}%), 次数: {info['count']}\n")
        L.append("```")
        stack = info["stack"]
        for line in stack.strip().split("\n")[:15]:
            L.append(line.strip())
        L.append("```\n")


# ===========================================================================
# lagrange_torch reference report
# ===========================================================================

def print_lagrange_refs(lagrange_refs):
    if not lagrange_refs:
        return
    print_header("lagrange_torch 框架引用")
    print(f"  仓库地址: {LAGRANGE_TORCH_REPO}\n")
    for ref in lagrange_refs:
        print(f"  Kernel: {ref['kernel_name'][:80]}")
        print(f"    耗时: {format_duration(ref['dur'])}, 次数: {ref['count']}")
        if ref["op_name"]:
            print(f"    算子: {ref['op_name']}")
            if ref["search_url"]:
                print(f"    源码搜索: {ref['search_url']}")
        print()


def save_lagrange_refs_md(lagrange_refs, L):
    if not lagrange_refs:
        return
    L.append("## lagrange_torch 框架引用\n")
    L.append(f"仓库地址: [{LAGRANGE_TORCH_REPO}]({LAGRANGE_TORCH_REPO})\n")
    L.append("| Kernel | 耗时 | 次数 | 算子 | 源码链接 |")
    L.append("|--------|------|------|------|----------|")
    for ref in lagrange_refs:
        op = ref["op_name"] or "-"
        link = f"[搜索]({ref['search_url']})" if ref["search_url"] else "-"
        L.append(f"| {ref['kernel_name'][:70]} | {format_duration(ref['dur'])} | {ref['count']} | {op} | {link} |")
    L.append("")


# ===========================================================================
# Markdown report
# ===========================================================================

def save_markdown_report(meta, trace_type, step_dur_us, step_decomp, thread_info, workers,
                         mod_info, gpu_info, dh_info, output_path, max_level=None,
                         src_info=None, source_files=None, timelines=None,
                         kernel_stacks=None, lagrange_refs=None, screenshot_path=None,
                         num_steps=1, overlap_info=None):
    L = []
    L.append("# Torch Trace 训练分析报告\n")

    # Overview
    L.append("## 概览\n")
    L.append(f"- **Trace 类型**: {'训练' if trace_type == 'training' else '推理'}")
    L.append(f"- **设备**: {meta.get('device_name', 'N/A')}")
    if meta.get("world_size", 1) > 1:
        L.append(f"- **分布式**: rank {meta.get('rank', 0)} / world_size {meta.get('world_size', 1)}")
    L.append(f"- **ProfilerStep 数**: {num_steps}")
    L.append(f"- **平均 Step 耗时**: {format_duration(step_dur_us)} (所有数据已按 {num_steps} 步平均)")
    L.append("")

    if step_decomp:
        L.append("### Step 分解\n")
        L.append("| 阶段 | 耗时 | 占比 |")
        L.append("|------|------|------|")
        sdur = step_decomp.get("step_dur", step_dur_us)
        if "forward_backward_dur" in step_decomp:
            L.append(f"| forward_backward | {format_duration(step_decomp['forward_backward_dur'])} | {pct_str(step_decomp['forward_backward_dur'], sdur)} |")
        if "optimize_dur" in step_decomp:
            L.append(f"| optimize | {format_duration(step_decomp['optimize_dur'])} | {pct_str(step_decomp['optimize_dur'], sdur)} |")
        if "other_dur" in step_decomp:
            L.append(f"| other | {format_duration(step_decomp['other_dur'])} | {pct_str(step_decomp['other_dur'], sdur)} |")
        L.append("")

    # Thread overview
    L.append("## 线程分析\n")
    L.append("| 线程 | 角色 | 事件数 | 总耗时 | Modules |")
    L.append("|------|------|--------|--------|---------|")
    for w in workers:
        mod_str = ", ".join(w["modules"][:3]) if w["modules"] else "-"
        L.append(f"| {w['name']} | {w['role']} | {w['event_count']} | {format_duration(w['total_dur'])} | {mod_str} |")
    L.append("")

    # Module tree
    L.append("## Module 耗时分析\n")
    L.append("### Module 层级树\n")
    L.append("```")
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_module_tree(mod_info, "ROOT", 0, step_dur_us, mod_info.get("module_thread_info"))
    tree_str = buf.getvalue()
    sys.stdout = old_stdout
    L.append(tree_str)
    L.append("```\n")

    depth_map = mod_info["depth_map"]
    max_depth = mod_info["max_depth"]
    if max_level is not None:
        max_depth = min(max_depth, max_level)

    for level in range(max_depth + 1):
        level_modules = [n for n, d in depth_map.items() if d == level]
        if not level_modules:
            continue
        level_modules.sort(key=lambda n: mod_info["module_durations"].get(n, 0), reverse=True)
        L.append(f"### Level {level} Modules\n")
        L.append("| Module | Thread | Inclusive | Exclusive | Incl% | Excl% | Calls |")
        L.append("|--------|--------|----------|-----------|-------|-------|-------|")
        for name in level_modules:
            inc = mod_info["module_durations"].get(name, 0)
            exc = mod_info["module_exclusive"].get(name, 0)
            calls = mod_info["module_call_counts"].get(name, 0)
            short = name.replace("nn.Module: ", "")
            ti = mod_info["module_thread_info"].get(name, {})
            roles = ti.get("roles", set())
            tag = "worker" if ("worker_thread" in roles and "main_thread" not in roles) else "main" if "main_thread" in roles else "other"
            L.append(f"| {short} | {tag} | {format_duration(inc)} | {format_duration(exc)} | {pct_str(inc, step_dur_us)} | {pct_str(exc, step_dur_us)} | {calls} |")
        L.append("")

    # GPU timeline
    if gpu_info:
        L.append("## GPU 时间线分析\n")
        L.append(f"- **GPU 利用率**: {gpu_info['gpu_utilization']:.1f}%")
        L.append(f"- **Kernel 活跃时间**: {format_duration(gpu_info['total_kernel_dur'])}")
        L.append(f"- **Kernel 间隙时间**: {format_duration(gpu_info['total_gap'])}")
        L.append("")

        L.append("### Top 热点 Kernel\n")
        L.append("| Kernel | 耗时 | 占比 | 次数 | Host Module |")
        L.append("|--------|------|------|------|-------------|")
        for kname, info in gpu_info["top_kernels"]:
            pct = pct_str(info["dur"], gpu_info["gpu_step_dur"])
            modules = gpu_info["top_kernel_modules"].get(kname, [])
            mod_str = modules[0][0] if modules else "N/A"
            L.append(f"| {kname[:80]} | {format_duration(info['dur'])} | {pct} | {info['count']} | {mod_str} |")
        L.append("")

        L.append("### Top Kernel 间隙\n")
        L.append("| 间隙时长 | 前一个 Kernel | 后一个 Kernel |")
        L.append("|----------|---------------|---------------|")
        for g in gpu_info["top_gaps"][:15]:
            L.append(f"| {format_duration(g['gap_us'])} | {g['before_name'][:50]} | {g['after_name'][:50]} |")
        L.append("")

    # Device/Host
    L.append("## Device/Host 耗时总览\n")
    host = dh_info["host_total_us"]
    device = dh_info["device_total_us"]
    L.append("| 类别 | 耗时 | 占比 |")
    L.append("|------|------|------|")
    L.append(f"| Host (CPU) | {format_duration(host)} | {pct_str(host, step_dur_us)} |")
    L.append(f"| Device (GPU) | {format_duration(device)} | {pct_str(device, step_dur_us)} |")
    L.append("")

    L.append("### Device 耗时分解\n")
    L.append("| 类别 | 耗时 | 占比 |")
    L.append("|------|------|------|")
    for cat, dur in sorted(dh_info["device_breakdown"].items(), key=lambda x: -x[1]):
        L.append(f"| {cat} | {format_duration(dur)} | {pct_str(dur, step_dur_us)} |")
    L.append("")

    # Comm/compute overlap
    if overlap_info:
        save_comm_compute_overlap_md(overlap_info, gpu_info, L)

    # Source hotspot
    if src_info and source_files:
        save_source_hotspot_markdown(src_info, source_files, L)

    # Per-thread timeline
    if timelines:
        save_per_thread_timeline_md(timelines, L)

    # Kernel call stacks
    if kernel_stacks and gpu_info:
        save_kernel_call_stacks_md(kernel_stacks, gpu_info, L)

    # lagrange_torch references
    if lagrange_refs:
        save_lagrange_refs_md(lagrange_refs, L)

    # Screenshot
    if screenshot_path:
        rel_path = os.path.basename(screenshot_path)
        L.append("## Trace 可视化\n")
        L.append(f"![Trace Screenshot]({rel_path})\n")

    with open(output_path, "w") as f:
        f.write("\n".join(L))
    print(f"  Markdown 报告已保存到: {output_path}")


# ===========================================================================
# HTML Flowchart Generation (Source-Code First)
# ===========================================================================


def _resolve_str_list_iterative(
    expr: str,
    fname: str,
    cname: str,
    source_lines,
    loop_var_to_str_items: dict,
    class_str_list_attrs: dict,
    file_str_list_globals: dict,
    cond_branch_stack: list,
    conditional_mode: str = "default",
    _global_str_lists_unique: dict = None,
):
    """Phase B iterative multi-step string-list resolver.

    Resolves a dynamic key expression (used in ModuleDict subscript or setattr
    name_expr) to a concrete list[str] of keys via iterative derivation.

    Returns list[str] on success, None when resolution fails.
    """
    if _global_str_lists_unique is None:
        _global_str_lists_unique = {}
    if expr is None:
        return None
    e = expr.strip()
    if not e:
        return None

    # Step 1: direct loop variable lookup
    if e in loop_var_to_str_items:
        items = list(loop_var_to_str_items[e])
        # If the expression itself is a plain loop var, return its items
        return items

    # Parse the expression as AST
    try:
        _node = ast.parse(e, mode='eval').body
    except SyntaxError:
        return None

    # Step 2: f-string expansion
    # e.g. f"{name}_tower" with name ∈ ['relation', 'gift', 'stay']
    # → ['relation_tower', 'gift_tower', 'stay_tower']
    if isinstance(_node, ast.JoinedStr):
        return _expand_fstring_with_loop_vars(_node, loop_var_to_str_items)

    # Step 3: self.<attr> referencing a known class str-list attr
    if (isinstance(_node, ast.Attribute)
            and isinstance(_node.value, ast.Name)
            and _node.value.id == 'self'):
        items = (class_str_list_attrs.get(cname) or {}).get(_node.attr)
        if items:
            return list(items)

    # Step 4: UPPER_CASE file/global constant
    _const_name = None
    if isinstance(_node, ast.Name):
        _const_name = _node.id
    elif isinstance(_node, ast.Attribute):
        _const_name = _node.attr
    if (_const_name and _const_name.isupper() and _const_name[:1].isalpha()
            and all(c.isalnum() or c == '_' for c in _const_name)):
        file_slots = file_str_list_globals.get(fname, {})
        if _const_name in file_slots:
            return list(file_slots[_const_name])
        if _const_name in _global_str_lists_unique:
            return list(_global_str_lists_unique[_const_name])

    # Step 5: bare Name that is NOT a loop var and NOT UPPER_CASE —
    # indicates a runtime/parameter variable we cannot statically resolve.
    if isinstance(_node, ast.Name):
        # Check if it might be resolvable via dotted Cfg.KEYS (already handled)
        # or if it's truly opaque (e.g. cfg.dynamic_keys).
        return None

    # Step 6: dotted attribute that is NOT self.* (e.g. cfg.dynamic_keys)
    if isinstance(_node, ast.Attribute):
        return None

    return None


def _expand_fstring_with_loop_vars(joined_str_node, loop_var_to_str_items: dict):
    """Expand an f-string AST node by substituting loop variables.

    For each FormattedValue in the JoinedStr, if the referenced variable is a
    known loop variable with string items, produce the cartesian expansion.
    Only single-variable f-strings are supported (one FormattedValue referencing
    one loop var); multi-variable f-strings return None.

    Returns list[str] on success, None on failure.
    """
    # Collect parts: each is either a constant string or a variable reference
    parts = []  # list of (kind, data) where kind='const' data=str | kind='var' data=items_list
    for value in joined_str_node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(('const', value.value))
        elif isinstance(value, ast.FormattedValue):
            # Only support simple Name references
            inner = value.value
            if isinstance(inner, ast.Name):
                var_name = inner.id
                if var_name in loop_var_to_str_items:
                    parts.append(('var', list(loop_var_to_str_items[var_name])))
                else:
                    return None
            else:
                return None
        else:
            return None

    if not parts:
        return None

    # Check we have at least one variable part
    has_var = any(k == 'var' for k, _ in parts)
    if not has_var:
        # Pure constant f-string (no interpolation) — just concat
        result = ''.join(d for _, d in parts)
        return [result]

    # Cartesian expansion: for each variable part, iterate its items;
    # for constant parts, keep as-is.
    # Only support single variable dimension for now (avoid combinatorial explosion).
    var_parts = [(i, items) for i, (k, items) in enumerate(parts) if k == 'var']
    if len(var_parts) > 1:
        # Multiple independent variables in one f-string — check if they all
        # reference the same loop var (same items list).
        first_items = var_parts[0][1]
        if all(items == first_items for _, items in var_parts):
            # Same variable repeated — expand once
            pass
        else:
            return None

    # Single variable (or same variable repeated): expand
    items_list = var_parts[0][1]
    results = []
    for item in items_list:
        built = []
        for kind, data in parts:
            if kind == 'const':
                built.append(data)
            else:
                built.append(str(item))
        results.append(''.join(built))
    return results


def build_static_module_tree(source_files, preferred_root=None, conditional_mode="infer", runtime_overrides=None):
    """Build module hierarchy tree purely from source code.
    Args:
        source_files: dict of {filename: [lines]}
        preferred_root: optional class name to use as primary root if present in tree.
            Useful when timeline indicates a specific top-level wrapper (e.g. DDPAwemeLiveCVR
            implies AwemeLiveCVR is the actual model entry-point).
        conditional_mode: one of "infer" / "train" / "default".
            Iter13 Step1: inside ``__init__``, when we encounter::
                if is_training: self.attr = A(...)
                else:           self.attr = B(...)
            (or any equivalent variant whose condition references
            ``is_training`` / ``is_serving`` / ``training`` / ``serving``),
            we follow ONLY the matching branch:
              - ``infer`` → take the inference branch (default)
              - ``train`` → take the training branch
              - ``default`` → fall back to last-write-wins (legacy behaviour,
                kept for tests / backwards compatibility)
            The discovered ``class_conditional_attrs`` mapping is also stashed
            on every tree[cname] as ``conditional_attrs`` so the caller can
            decide whether to render train/infer tabs.
    Returns: tree dict, root list, class_map
    """
    ast_frontends = _build_ast_frontends(source_files)

    def _get_ast_frontend(fname):
        return ast_frontends.get(fname)

    class_map = _build_class_map(source_files, ast_frontends=ast_frontends)
    module_call_order, _ = _build_source_dependency_order(source_files, class_map, ast_frontends=ast_frontends)

    # Iter18: Build the AST module tree IN-PROCESS (no JSON intermediate).
    # The extractor runs once per build_static_module_tree() invocation and
    # produces a parallel "ground-truth" view of class_attrs / container
    # elements / dynamic setattrs that downstream consumers can consult when
    # the existing regex pass is uncertain.  The data is stashed on each
    # tree[cname] entry as ``_ast_*`` keys near the end of this function.
    _ast_extractor = _AST_ModuleTreeExtractor(source_files, ast_frontends=ast_frontends)

    # ------------------------------------------------------------------
    # 轻量常量表：用于 ModuleList/ModuleDict 展开时解析 range(N) / layers=CONST
    # 仅解析“文件级”形如 `NAME = 6` 的 int 常量。
    # AST-only:  parse each source file once and inspect top-level
    # ``ast.Assign`` statements where the target is an UPPER_SNAKE Name
    # and the value is a non-negative integer Constant.
    # ------------------------------------------------------------------
    file_int_consts = {}
    file_str_list_globals_raw = {}  # {fname: {VARNAME: raw_inner_string}}

    def _is_upper_snake_name(_s: str) -> bool:
        return bool(_s) and _s[0:1].isupper() and all(
            c.isupper() or c.isdigit() or c == "_" for c in _s
        )

    for _fname, _lines in source_files.items():
        int_consts = {}
        str_list_consts_raw = {}
        _fe = _get_ast_frontend(_fname)
        _file_tree = _fe.tree if _fe is not None else None
        if _file_tree is not None:
            for _node in _file_tree.body:
                if not isinstance(_node, ast.Assign) or len(_node.targets) != 1:
                    continue
                _t = _node.targets[0]
                if not (isinstance(_t, ast.Name) and _is_upper_snake_name(_t.id)):
                    continue
                _v = _node.value
                # int constant: ``NAME = 42``
                if isinstance(_v, ast.Constant) and isinstance(_v.value, int) and not isinstance(_v.value, bool):
                    if _v.value >= 0:
                        int_consts[_t.id] = _v.value
                # list literal: ``NAME = ['a', 'b']`` — store the unparse of
                # the inner element list so the existing
                # ``_parse_str_literal_list`` helper can consume it later.
                elif isinstance(_v, ast.List):
                    try:
                        _inner_text = ", ".join(ast.unparse(_e) for _e in _v.elts)
                    except Exception:
                        _inner_text = ""
                    str_list_consts_raw[_t.id] = _inner_text
        file_int_consts[_fname] = int_consts
        file_str_list_globals_raw[_fname] = str_list_consts_raw

    # 全局常量：同名常量在多个文件出现时，若值唯一则可跨文件解析
    global_int_const_values = defaultdict(set)
    for _fname, _cs in file_int_consts.items():
        for _k, _v in _cs.items():
            global_int_const_values[_k].add(_v)

    # Iter13 Step2: file-level string-literal list globals, e.g.
    #   COMMON_KEYS = ['a', 'b']
    # Used to resolve ModuleDict keys driven by such constants.
    # NOTE: ``file_str_list_globals_raw`` was populated above (AST-driven).
    # Realised lookup: parsed lists, populated below once
    # ``_parse_str_literal_list`` is available.  We keep it as a placeholder dict
    # and fill it lazily in the per-class scanning loop.
    file_str_list_globals = {}  # {fname: {VARNAME: [str, ...]}}

    # ctor_kw_int_args[ClassName][kw] = {1,2,...}
    ctor_kw_int_args = defaultdict(lambda: defaultdict(set))

    # Iter17: ctor_kw_list_lens[ClassName][kw] = {len1, len2, ...}
    # Records the LENGTH (int) of list literals passed as kw= at construction
    # sites of ClassName.  Used to expand `for i, x in enumerate(<param>):`
    # inside ClassName.__init__ when <param> is bound to a kw with known
    # list-length.
    ctor_kw_list_lens = defaultdict(lambda: defaultdict(set))

    # Iter17: per-instance ctor kw list lengths.
    # instance_kw_list_lens[(parent_cname, attr_name)][kw] = N
    # Used in build_dag_recursive to PRUNE per-instance children counts when
    # the class internally expands containers via `for ... in enumerate(<kw>)`.
    instance_kw_list_lens = defaultdict(dict)

    def _eval_expr_node(expr_text: str):
        expr_text = (expr_text or '').strip()
        if not expr_text:
            return None
        try:
            return ast.parse(expr_text, mode="eval").body
        except Exception:
            return None

    def _resolver_scope(fname: str, cname: str = None, mname: str = None,
                        parent_cls: str = None, parent_attr: str = None):
        return Scope(file=fname, cls=cname, method=mname,
                     parent_cls=parent_cls, parent_attr=parent_attr)

    def _eval_int_node(expr_node, fname: str, cname: str = None, mname: str = None,
                       parent_cls: str = None, parent_attr: str = None):
        if expr_node is None or _new_eval_resolver is None:
            return None
        try:
            iv = _new_eval_resolver.eval_int(
                expr_node,
                _resolver_scope(fname, cname, mname, parent_cls, parent_attr),
            )
        except Exception:
            return None
        return iv.value if iv is not None else None

    def _eval_list_len_node(expr_node, fname: str, cname: str = None, mname: str = None,
                            parent_cls: str = None, parent_attr: str = None):
        if expr_node is None or _new_eval_resolver is None:
            return None
        try:
            return _new_eval_resolver.eval_list_len(
                expr_node,
                _resolver_scope(fname, cname, mname, parent_cls, parent_attr),
            )
        except Exception:
            return None

    # Build inheritance info - detect which classes extend nn.Module (or other user classes)
    nn_module_classes = set()
    for fname, lines in source_files.items():
        _fe = _get_ast_frontend(fname)
        if _fe:
            for _ast_cname in _fe.class_registry.keys():
                if _fe.is_nn_module(_ast_cname):
                    nn_module_classes.add(_ast_cname)
        _file_tree_for_classes = _fe.tree if _fe is not None else None
        if _file_tree_for_classes is not None:
            for _node in ast.walk(_file_tree_for_classes):
                if not isinstance(_node, ast.ClassDef) or not _node.bases:
                    continue
                cname = _node.name
                for _base_node in _node.bases:
                    try:
                        _b = ast.unparse(_base_node)
                    except Exception:
                        _b = ""
                    b_short = _b.split(".")[-1] if _b else ""
                    if b_short in ("Module", "nn.Module") or b_short in nn_module_classes:
                        nn_module_classes.add(cname)
                        break
    # Also add classes found in class_map that have forward() method
    for (fname, cname), info in class_map.items():
        if "forward" in info["methods"]:
            nn_module_classes.add(cname)

    # ------------------------------------------------------------------
    # PR3 — build the ConstantTable + ConstantResolver once and use it as the
    # only evaluator path.
    # ------------------------------------------------------------------
    _new_eval_table = None
    _new_eval_resolver = None
    _new_eval_diagnostics = []
    _new_eval_ast_frontends = {}
    for _fname in source_files.keys():
        _fe = _get_ast_frontend(_fname)
        if _fe is not None:
            _new_eval_ast_frontends[_fname] = _fe
    _new_eval_table = ConstantTable.build_all(
        source_files=source_files,
        ast_frontends=_new_eval_ast_frontends,
        nn_module_classes=set(nn_module_classes),
    )
    _new_eval_resolver = ConstantResolver(
        _new_eval_table,
        runtime_overrides=runtime_overrides,
        diagnostics=_new_eval_diagnostics,
    )


    # Per-class attr->class mapping. We scan ALL methods (not just __init__) because
    # some codebases lazily build child sub-modules in helper methods (e.g.
    # ``self.revenue_perceiver_module = SeqPerceiver(...)`` inside ``gen_lsf_fc``).
    # We also capture ``setattr(self, name, ClassName(...))`` patterns where the
    # attribute name comes from a runtime variable — such usages cannot be resolved
    # to a literal attr name, so we synthesise one from the class to keep the child
    # connected (group_<ClassName>).
    class_attrs = {}  # {class_name: {attr_name: class_ref}}
    torch_native_module_classes = set()  # {"nn.LayerNorm", "nn.Linear", ...}
    # Per-(class, attr) source location: {(cname, attr_name): (fname, lineno)}
    attr_def_loc = {}
    # Track attrs that are LG input sources (LG.feature_column/dense_feature/get_sample_rate/get_bias)
    input_source_attrs = defaultdict(set)  # {class_name: {attr_name, ...}}
    # Iter11: literal string-list attrs, e.g. self.group_tower_names = ['relation_layer', ...]
    # {class_name: {attr_name: [str, str, ...]}}
    # Used to drive dynamic setattr/getattr expansion in for-loops over literal str lists.
    class_str_list_attrs = defaultdict(dict)
    # Iter11: track which attrs were registered via dynamic setattr (non-literal name,
    # not driven by a known str-list loop). DAG node name uses just the class name
    # (e.g. `MultiHeadAttention`); the per-class set lets the data-flow analyzer know
    # to wildcard-route dynamic getattr calls to these attrs.
    dynamic_attrs_per_class = defaultdict(set)  # {class_name: {attr_name, ...}}

    # Iter13 Step1: per-class conditional attr mapping discovered while
    # scanning ``__init__`` for ``if is_training: ...; else: ...`` patterns.
    # Structure:
    #   class_conditional_attrs[cname][attr_name] = {
    #       "train_class": Cls or None,
    #       "infer_class": Cls or None,
    #       "train_loc":  (file, lineno) or None,
    #       "infer_loc":  (file, lineno) or None,
    #   }
    # Independent of ``conditional_mode`` — we always populate this so that
    # the caller can decide whether to render a train / infer tab.
    class_conditional_attrs = defaultdict(dict)

    # Iter13 Step2: unresolved ModuleDict/ModuleList expansions.
    # Each entry: {"class": cname, "attr": attr_name, "expr": raw_expr,
    #              "reason": str, "file": fname, "line": lineno}
    # Surfaced in stdout as `[ERROR] Cannot statically enumerate ...` and
    # also stashed on the tree (tree[cname]["unresolved_moduledict"]) for
    # downstream summary printing in generate_html_flowchart().
    unresolved_moduledict = []

    def _cond_branch_polarity(cond_text: str):
        """Iter13 Step1: classify whether a Python ``if <cond>:`` header
        targets the **training** branch or the **inference** branch.

        Returns one of:
          - "train"   → cond evaluates True only when training
          - "infer"   → cond evaluates True only when serving/inference
          - None      → cond does not reference is_training / is_serving keywords

        Heuristic only — we don't try to fully evaluate complex Python
        expressions.  We look for whole-word occurrences of the four
        canonical names (case-insensitive, including ``self.is_training``
        / ``common.is_training`` style attribute access) and figure out
        the polarity from the surrounding ``not``.
        """
        if not cond_text:
            return None
        text = cond_text.strip()
        # Strip a trailing colon if any (defensive — _cond_if_re keeps it out).
        text = text.rstrip(':').strip()
        # AST-only: parse the cond as a Python expression, then walk Name/
        # Attribute nodes to collect tokens; track UnaryOp(Not, ...) clauses
        # to detect polarity flips around the canonical names.
        try:
            _cond_node = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None
        train_words = ("is_training", "training")
        serve_words = ("is_serving", "serving")

        def _leaf_token(_node):
            """Render the rightmost identifier of a Name/Attribute chain
            (lowercased). Returns None for anything else."""
            if isinstance(_node, ast.Name):
                return _node.id.lower()
            if isinstance(_node, ast.Attribute):
                return _node.attr.lower()
            return None

        # Collect every leaf token anywhere in the expression so we can
        # decide whether a train / serve canonical name appears at all.
        tokens = []
        for _sub in ast.walk(_cond_node):
            if isinstance(_sub, (ast.Name, ast.Attribute)):
                _tok = _leaf_token(_sub)
                if _tok:
                    tokens.append(_tok)
        has_train = any(t in train_words for t in tokens)
        has_serve = any(t in serve_words for t in tokens)
        if not (has_train or has_serve):
            return None
        # Detect a UnaryOp(Not, <name/attr>) whose leaf token is a canonical
        # train/serve word — that flips polarity.
        flipped = False
        for _sub in ast.walk(_cond_node):
            if isinstance(_sub, ast.UnaryOp) and isinstance(_sub.op, ast.Not):
                _tail = _leaf_token(_sub.operand)
                if _tail and (_tail in train_words or _tail in serve_words):
                    flipped = True
                    break
        # Compute base polarity from which keyword family appeared.
        if has_train and not has_serve:
            base = "train"
        elif has_serve and not has_train:
            base = "infer"
        else:
            # Mixed expression like ``is_training and is_serving`` is nonsense; skip.
            return None
        if flipped:
            return "infer" if base == "train" else "train"
        return base

    # ModuleList/ModuleDict 子模块展开：
    # - container_elems[cname][container_attr] = ["layers[0]", "layers[1]", ...] / ["loss_fns['ctr0']", ...]
    # - 这些 elem_attr 会额外写入 attrs 中，保证 DAG 里出现真实子模块节点。
    container_elems = defaultdict(lambda: defaultdict(list))
    # Iter14 container-group: record the *kind* (declared container class) of
    # each container attr so the DAG renderer can wrap expanded children in a
    # synthetic container parent group labelled e.g. ``ModuleDict[name][N]`` or
    # ``ModuleList[blocks][4]``. Keys: (cname, container_attr) -> kind string,
    # one of {"ModuleDict", "ModuleList", "dict", "list"}.
    container_kinds = {}

    def _record_container_kind(cname: str, container_attr: str, kind: str):
        """Record (or upgrade) the kind for cname.container_attr.

        Once recorded as a Module-typed container ("ModuleDict"/"ModuleList"),
        a later assignment (e.g. ``self.xx[k] = SomeModule(...)``) must not
        downgrade the kind to ``dict``/``list``.
        """
        key = (cname, container_attr)
        existing = container_kinds.get(key)
        priority = {"ModuleDict": 4, "ModuleList": 3, "dict": 2, "list": 1}
        if existing is None or priority.get(kind, 0) > priority.get(existing, 0):
            container_kinds[key] = kind

    def _normalize_str_key(expr: str):
        expr = (expr or '').strip()
        if not expr:
            return None
        try:
            node = ast.parse(expr, mode='eval').body
        except SyntaxError:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _elem_attr_list(container: str, idx: int) -> str:
        return f"{container}[{idx}]"

    def _elem_attr_dict(container: str, key: str) -> str:
        # 统一使用单引号，便于稳定匹配
        key = key.replace("'", "\\'")
        return f"{container}['{key}']"

    def _ensure_elem(container: str, cname: str, elem_attr: str, cls_ref: str, fname: str, lineno: int, attrs: dict):
        if elem_attr not in attrs:
            attrs[elem_attr] = cls_ref
            attr_def_loc.setdefault((cname, elem_attr), (fname, lineno))
        if elem_attr not in container_elems[cname][container]:
            container_elems[cname][container].append(elem_attr)

    def _split_top_level_commas(s: str):
        """把 `a(b, c), d, {'k': e}` 按顶层逗号切分。"""
        parts = []
        buf = []
        depth = 0
        in_str = None
        esc = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                p = ''.join(buf).strip()
                if p:
                    parts.append(p)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)
        return parts

    # Iter11: parse a literal list of string literals.
    # e.g. "['relation_layer', 'gift_layer', \"stay_layer\"]"
    # Returns list of strings, or None if any non-string-literal element exists.
    def _parse_str_literal_list(list_inner: str):
        try:
            node = ast.parse('[' + list_inner + ']', mode='eval').body
        except SyntaxError:
            return None
        if not isinstance(node, ast.List):
            return None
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                result.append(elt.value)
            else:
                return None
        return result if result else None

    # Iter13 Step2: realise the str-list globals lookup now that
    # ``_parse_str_literal_list`` is available.
    for _fname, _slots in file_str_list_globals_raw.items():
        parsed = {}
        for _name, _raw in _slots.items():
            _items = _parse_str_literal_list(_raw)
            if _items:
                parsed[_name] = _items
        file_str_list_globals[_fname] = parsed
    # Cross-file fallback: when a global str-list is unique by name, allow
    # lookup without specifying the file.
    _global_str_lists_unique = {}
    _name_to_lists = defaultdict(set)
    for _fname, _slots in file_str_list_globals.items():
        for _name, _items in _slots.items():
            _name_to_lists[_name].add(tuple(_items))
    for _name, _lists in _name_to_lists.items():
        if len(_lists) == 1:
            _global_str_lists_unique[_name] = list(next(iter(_lists)))

    _str_list_resolve_cache = {}

    def _resolve_str_list(expr: str, fname: str, cname: str,
                          loop_var_to_str_items: dict):
        """Iter13 Step2 + Phase B: iterative multi-step string-list resolver.
        Delegates to the module-level _resolve_str_list_iterative which handles:
          1. direct loop variable lookup
          2. f-string expansion with loop variable substitution
          3. self.<attr> referring to a known class str-list attr
          4. file-level / global UPPER_CASE str-list constant
          5. bare Name / dotted attribute → None (unresolvable)
        """
        _loop_key = tuple(
            sorted((str(k), tuple(v or ())) for k, v in (loop_var_to_str_items or {}).items())
        )
        _cache_key = (expr, fname, cname, _loop_key, conditional_mode)
        if _cache_key in _str_list_resolve_cache:
            _cached_keys = _str_list_resolve_cache[_cache_key]
            return list(_cached_keys) if _cached_keys is not None else None
        _resolved = _resolve_str_list_iterative(
            expr=expr,
            fname=fname,
            cname=cname,
            source_lines=source_files.get(fname, []),
            loop_var_to_str_items=loop_var_to_str_items,
            class_str_list_attrs=class_str_list_attrs,
            file_str_list_globals=file_str_list_globals,
            cond_branch_stack=[],
            conditional_mode=conditional_mode,
            _global_str_lists_unique=_global_str_lists_unique,
        )
        _str_list_resolve_cache[_cache_key] = tuple(_resolved) if _resolved is not None else None
        return list(_resolved) if _resolved is not None else None

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        _fe = _get_ast_frontend(fname)
        attrs = {}

        # Scan every method body for module assignments
        for mname, (ms, me) in info["methods"].items():
            _ast_method_range = _fe.get_method_lines(cname, mname) if _fe else None
            if _ast_method_range:
                ms, me = _ast_method_range
            method_lines = lines[ms - 1: min(me, len(lines))]
            # Track absolute line numbers per joined logical line by storing the
            # index of the first physical line that contributed to it.
            logical_lines = []
            buf = ""
            buf_start_lineno = None
            open_count = 0
            for offset, line in enumerate(method_lines):
                phys_lineno = ms + offset
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    if not buf:
                        logical_lines.append((phys_lineno, line))
                    continue
                if not buf:
                    buf_start_lineno = phys_lineno
                stripped_nc = _strip_inline_comment(stripped).strip()
                if not stripped_nc:
                    # 这一行去掉注释后为空
                    continue
                buf += (" " if buf else "") + stripped_nc
                open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
                open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
                if open_count <= 0:
                    logical_lines.append((buf_start_lineno, buf))
                    buf = ""
                    buf_start_lineno = None
                    open_count = 0
            if buf:
                logical_lines.append((buf_start_lineno, buf))

            _ast_init_assignments_by_line = {}
            if mname == "__init__" and _fe:
                for _ast_item in (_fe.get_init_assignments_ast(cname) or []):
                    _lineno = _ast_item.get("lineno")
                    if _lineno is not None:
                        _ast_init_assignments_by_line[_lineno] = _ast_item

            # Iter11: track loop variables that iterate over a literal string list,
            # so `for name in self.xxx_names: setattr(self, name, Cls(...))` expands
            # the setattr into real per-element attrs ['relation_layer', 'gift_layer', ...].
            # Scope: simple per-method, no nested-loop complexity. Reset per method.
            _loop_var_to_str_items = {}  # {loop_var_name: [str_item, ...]}
            _loop_indent_stack = []  # [(indent, loop_var)] tracking active loops

            # Iter13 Step1: track if/else conditional branches whose header
            # references is_training / is_serving / training / serving.
            # Stack entries: (indent_of_if_header, current_branch_polarity)
            #   current_branch_polarity ∈ {"train", "infer"}.
            # On ``else:`` at the SAME indent as the matching ``if``, we flip
            # the top-of-stack polarity. On dedent past the if-header indent,
            # we pop. Only top-level (single-frame) conditionals are honoured —
            # nested is_training/is_serving conditionals stack as expected.
            _cond_branch_stack = []  # [(if_indent, branch_polarity)]

            # Iter16: per-method local-variable constructor table.
            # Tracks `var = ClassName(...)` where ClassName is a known
            # nn.Module class. Used to resolve the two-step setattr pattern:
            #     combine_tower = DenseTower(...)
            #     setattr(self, f"combine_tower_mha_{name}", combine_tower)
            # so that `setattr(self, name_expr, var)` can be treated as an
            # inline `setattr(self, name_expr, ClassName(...))`.
            #   {var_name: (cls_ref, fname, ctor_lineno)}
            _local_var_ctor: dict = {}

            # Phase B: track local variable → string list assignments for
            # iterative multi-step resolution. Handles patterns like:
            #     ks0 = ['a', 'b', 'c']
            #     ks1 = ks0
            #     keys = ks1
            #     for k in keys: ...
            # {var_name: list[str]}
            _local_var_str_list: dict = {}

            def _self_attr_from_target(_target):
                return _fe._extract_self_attr_name(_target) if _fe else None

            def _node_to_text(_node):
                return _fe._local_node_to_text(_node) if _fe else ""

            for phys_lineno, line in logical_lines:
                # Iter11: track indent of this logical line (use raw source line)
                _raw_line_for_indent = lines[phys_lineno - 1] if 0 < phys_lineno <= len(lines) else ""
                _cur_indent = len(_raw_line_for_indent) - len(_raw_line_for_indent.lstrip())
                # Iter12-final-fix: skip pure comment lines.  `_join_logical_lines`
                # preserves blank/comment lines (with the leading ``#``) when no
                # multi-line buffer is in flight, so without this guard a
                # commented-out ``# self.dense_towers.append(DenseTower(`` would
                # be matched by the AST setattr/append helpers below and pollute
                # attr_def_loc / attr enumeration with attrs that have no real
                # call site.
                if line.lstrip().startswith('#'):
                    continue
                _local_stmt = _fe.parse_local_stmt(line) if _fe else None
                # Pop loops whose indent is >= current indent (we left their body)
                while _loop_indent_stack and _cur_indent <= _loop_indent_stack[-1][0]:
                    _, _popped_var = _loop_indent_stack.pop()
                    _loop_var_to_str_items.pop(_popped_var, None)

                # Iter13 Step1: pop conditional frames whose if-header indent is
                # >= current indent (we left their body) — but only when the
                # current line is NOT the matching ``else:`` (which lives at the
                # same indent as the if header).
                # Detect ``if/elif/else:`` headers via AST instead of regex.
                # Strategy: try parsing ``<line>\n    pass`` so a header line
                # becomes a syntactically complete If statement.
                _hdr_node = None
                _stripped_line = line.rstrip()
                if _stripped_line.endswith(':'):
                    _stripped_left = _stripped_line.lstrip()
                    if _stripped_left.startswith(('if ', 'elif ', 'else:', 'if(', 'elif(')) \
                            or _stripped_left == 'else:':
                        try:
                            _hdr_mod = ast.parse(_stripped_line + "\n    pass")
                        except SyntaxError:
                            _hdr_mod = None
                        if _hdr_mod and len(_hdr_mod.body) == 1 \
                                and isinstance(_hdr_mod.body[0], ast.If):
                            _hdr_node = _hdr_mod.body[0]
                _is_else_line = (_hdr_node is not None
                                 and _stripped_line.lstrip().startswith('else'))
                _is_elif_line = (_hdr_node is not None
                                 and _stripped_line.lstrip().startswith('elif'))
                _is_if_line = (_hdr_node is not None and not _is_else_line and not _is_elif_line)
                while _cond_branch_stack:
                    _if_ind, _polarity = _cond_branch_stack[-1]
                    if _cur_indent < _if_ind:
                        _cond_branch_stack.pop()
                    elif _cur_indent == _if_ind and (_is_else_line or _is_elif_line):
                        # ``else:`` at the matching indent → flip polarity.
                        _cond_branch_stack[-1] = (_if_ind,
                                                  "infer" if _polarity == "train" else "train")
                        break
                    elif _cur_indent == _if_ind:
                        # Iter14 fix: same indent as the ``if`` header but the
                        # current line is *not* an ``else``/``elif`` — the
                        # conditional block has ended and execution returned to
                        # the surrounding scope. Pop the frame so subsequent
                        # ``self.xxx = ...`` lines are treated as unconditional
                        # (shared between train/infer). Without this, e.g.
                        # CTRModule.__init__'s ``if not is_training: self.df=...``
                        # line was leaving the stack populated, causing every
                        # later ``self.ue = UniversalEmbedding()`` etc. to be
                        # filtered out in train mode.
                        _cond_branch_stack.pop()
                    else:
                        break

                # Detect new ``if <cond>:`` whose cond polarity is train/infer.
                # We treat a fresh ``if`` (not elif) as opening a new frame.
                if _is_if_line and _hdr_node is not None:
                    try:
                        _cond_text = ast.unparse(_hdr_node.test)
                    except Exception:
                        _cond_text = ""
                    _pol = _cond_branch_polarity(_cond_text)
                    if _pol is not None:
                        _cond_branch_stack.append((_cur_indent, _pol))

                # Compute the active branch polarity (top of stack) — None when
                # we are not inside any train/infer conditional.
                _active_branch = _cond_branch_stack[-1][1] if _cond_branch_stack else None

                # Iter11: detect literal string-list assignments:
                #   self.xxx_names = ['a', 'b', ...]
                # Tracked separately from `attrs` because the elements are not nn.Modules
                # by themselves; they are *names* used to drive setattr/getattr.
                if (isinstance(_local_stmt, ast.Assign)
                        and len(_local_stmt.targets) == 1
                        and isinstance(_local_stmt.value, ast.List)):
                    _slt_attr = _self_attr_from_target(_local_stmt.targets[0])
                    if _slt_attr:
                        _slt_items = []
                        _slt_ok = True
                        for _e in _local_stmt.value.elts:
                            if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                                _slt_items.append(_e.value)
                            else:
                                _slt_ok = False
                                break
                        if _slt_ok and _slt_items:
                            class_str_list_attrs[cname][_slt_attr] = list(_slt_items)

                # Phase B: track local variable string-list assignments
                # Patterns: var = ['a', 'b'] | var = other_known_var | var = self.attr
                if (isinstance(_local_stmt, ast.Assign)
                        and len(_local_stmt.targets) == 1
                        and isinstance(_local_stmt.targets[0], ast.Name)):
                    _lv_name = _local_stmt.targets[0].id
                    _lv_val = _local_stmt.value
                    if isinstance(_lv_val, ast.List):
                        _lv_items = []
                        _lv_ok = True
                        for _e in _lv_val.elts:
                            if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                                _lv_items.append(_e.value)
                            else:
                                _lv_ok = False
                                break
                        if _lv_ok and _lv_items:
                            _local_var_str_list[_lv_name] = list(_lv_items)
                    elif isinstance(_lv_val, ast.Name) and _lv_val.id in _local_var_str_list:
                        _local_var_str_list[_lv_name] = list(_local_var_str_list[_lv_val.id])
                    elif (isinstance(_lv_val, ast.Attribute)
                          and isinstance(_lv_val.value, ast.Name)
                          and _lv_val.value.id == 'self'):
                        _ref_items = class_str_list_attrs.get(cname, {}).get(_lv_val.attr)
                        if _ref_items:
                            _local_var_str_list[_lv_name] = list(_ref_items)

                # Iter11: detect `for [i,] name in [enumerate(]self.xxx_names[)]:`
                # If self.xxx_names is a known literal string list, record the loop var
                # so subsequent setattr(self, name, Cls(...)) can be expanded.
                # AST-based: parse via "<line>\n    pass" so we get a real ast.For.
                _for_node = None
                _stripped_for_line = line.rstrip()
                if _stripped_for_line.lstrip().startswith('for ') and _stripped_for_line.endswith(':'):
                    try:
                        _for_mod = ast.parse(_stripped_for_line + "\n    pass")
                    except SyntaxError:
                        _for_mod = None
                    if (_for_mod and len(_for_mod.body) == 1
                            and isinstance(_for_mod.body[0], ast.For)):
                        _for_node = _for_mod.body[0]

                def _self_attr_simple(_n):
                    if (isinstance(_n, ast.Attribute)
                            and isinstance(_n.value, ast.Name)
                            and _n.value.id == 'self'):
                        return _n.attr
                    return None

                def _str_list_from_node(_n):
                    if not isinstance(_n, ast.List):
                        return None
                    out = []
                    for _e in _n.elts:
                        if isinstance(_e, ast.Constant) and isinstance(_e.value, str):
                            out.append(_e.value)
                        else:
                            return None
                    return out if out else None

                _m_for_use_hit = False
                _m_for_lit_hit = False
                _m_for_tup_hit = False
                if _for_node is not None:
                    _iter = _for_node.iter
                    _tgt = _for_node.target
                    # Determine "second variable" (the name var) for both
                    # Name and Tuple targets.
                    _name_var = None
                    if isinstance(_tgt, ast.Name):
                        _name_var = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and len(_tgt.elts) >= 2 \
                            and all(isinstance(e, ast.Name) for e in _tgt.elts):
                        # `for i, name in enumerate(...)` → name var = last
                        _name_var = _tgt.elts[-1].id

                    # Strip outer enumerate(...) wrapper.
                    _enum_inner = None
                    if (isinstance(_iter, ast.Call)
                            and isinstance(_iter.func, ast.Name)
                            and _iter.func.id == 'enumerate'
                            and len(_iter.args) >= 1):
                        _enum_inner = _iter.args[0]

                    # Rule: for x in self.xxx: / for i, x in enumerate(self.xxx):
                    _src_self = _self_attr_simple(_iter) or (
                        _self_attr_simple(_enum_inner) if _enum_inner is not None else None)
                    if _src_self and _name_var and (
                            isinstance(_tgt, ast.Name)
                            or (_enum_inner is not None and isinstance(_tgt, ast.Tuple))):
                        _items = class_str_list_attrs.get(cname, {}).get(_src_self)
                        if _items:
                            _loop_var_to_str_items[_name_var] = list(_items)
                            _loop_indent_stack.append((_cur_indent, _name_var))
                            _m_for_use_hit = True

                    # Rule: for x in [literals]: / for i, x in enumerate([literals]):
                    if not _m_for_use_hit and _name_var:
                        _lit_node = _iter if isinstance(_iter, ast.List) else (
                            _enum_inner if _enum_inner is not None else None)
                        _items = _str_list_from_node(_lit_node) if _lit_node is not None else None
                        if _items and (
                                isinstance(_tgt, ast.Name)
                                or (_enum_inner is not None and isinstance(_tgt, ast.Tuple))):
                            _loop_var_to_str_items[_name_var] = list(_items)
                            _loop_indent_stack.append((_cur_indent, _name_var))
                            _m_for_lit_hit = True

                    # Rule: for (name, a, b, ...) in [(s, ..), (s, ..), ...]:
                    if (not _m_for_use_hit and not _m_for_lit_hit
                            and isinstance(_tgt, ast.Tuple)
                            and len(_tgt.elts) >= 2
                            and all(isinstance(e, ast.Name) for e in _tgt.elts)
                            and isinstance(_iter, ast.List)):
                        _first_var = _tgt.elts[0].id
                        _str_items = []
                        _ok = True
                        for _outer_elt in _iter.elts:
                            if not isinstance(_outer_elt, ast.Tuple) or not _outer_elt.elts:
                                _ok = False
                                break
                            _first_inner = _outer_elt.elts[0]
                            if not (isinstance(_first_inner, ast.Constant)
                                    and isinstance(_first_inner.value, str)):
                                _ok = False
                                break
                            _str_items.append(_first_inner.value)
                        if _ok and _str_items:
                            _loop_var_to_str_items[_first_var] = _str_items
                            _loop_indent_stack.append((_cur_indent, _first_var))
                            _m_for_tup_hit = True

                    # Phase B Rule 4: for x in local_var / for i, x in enumerate(local_var)
                    # where local_var is a tracked string-list local variable.
                    if (not _m_for_use_hit and not _m_for_lit_hit and not _m_for_tup_hit
                            and _name_var):
                        _local_iter_node = _iter if _enum_inner is None else _enum_inner
                        if isinstance(_local_iter_node, ast.Name):
                            _local_items = _local_var_str_list.get(_local_iter_node.id)
                            if _local_items:
                                _loop_var_to_str_items[_name_var] = list(_local_items)
                                _loop_indent_stack.append((_cur_indent, _name_var))

                _ast_init_item = _ast_init_assignments_by_line.get(phys_lineno)

                _ast_ctor_assign = _fe.parse_local_ctor_assign(_local_stmt) if _fe else None
                if _ast_ctor_assign:
                    _ctor_attr = _ast_ctor_assign.get("attr")
                    _cls_full = _ast_ctor_assign.get("class_full") or ""
                    _cls = _cls_full.split('.')[-1]
                    if _cls:
                        _kw_lens = {}
                        for k, v_expr in (_ast_ctor_assign.get("kwargs") or {}).items():
                            v_node = _eval_expr_node(v_expr)
                            v = _eval_int_node(v_node, fname, cname, mname)
                            if v is not None:
                                ctor_kw_int_args[_cls][k].add(v)
                            n = _eval_list_len_node(v_node, fname, cname, mname)
                            if n is not None:
                                ctor_kw_list_lens[_cls][k].add(n)
                                _kw_lens[k] = n
                        if _kw_lens and _ctor_attr:
                            instance_kw_list_lens[(cname, _ctor_attr)] = dict(_kw_lens)
                elif _ast_init_item:
                    _ctor_attr = _ast_init_item.get("attr")
                    _cls_full = _ast_init_item.get("class") or ""
                    _cls = _cls_full.split('.')[-1]
                    if _cls:
                        _kw_lens = {}
                        for k, v_expr in (_ast_init_item.get("kwargs") or {}).items():
                            v_node = _eval_expr_node(v_expr)
                            v = _eval_int_node(v_node, fname, cname, mname)
                            if v is not None:
                                ctor_kw_int_args[_cls][k].add(v)
                            n = _eval_list_len_node(v_node, fname, cname, mname)
                            if n is not None:
                                ctor_kw_list_lens[_cls][k].add(n)
                                _kw_lens[k] = n
                        if _kw_lens and _ctor_attr:
                            instance_kw_list_lens[(cname, _ctor_attr)] = dict(_kw_lens)

                # Direct assignment: self.xxx = ClassName(...)
                if _ast_ctor_assign or _ast_init_item:
                    if _ast_ctor_assign:
                        attr_name = _ast_ctor_assign.get("attr")
                        cls_ref_full = _ast_ctor_assign.get("class_full") or ""
                    else:
                        attr_name = _ast_init_item.get("attr")
                        cls_ref_full = _ast_init_item.get("class") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    native_cls_ref = None
                    if cls_ref_full.startswith("nn."):
                        _nn_short = cls_ref_full.split(".", 1)[1]  # e.g. "Linear", "ModuleDict"
                        if _is_nn_leaf_stub(_nn_short):
                            native_cls_ref = cls_ref_full
                    if cls_ref in nn_module_classes:
                        # Iter13 Step1: track conditional attrs regardless of mode.
                        if _active_branch is not None:
                            _entry = class_conditional_attrs[cname].setdefault(
                                attr_name,
                                {"train_class": None, "infer_class": None,
                                 "train_loc": None, "infer_loc": None})
                            if _active_branch == "train":
                                _entry["train_class"] = cls_ref
                                _entry["train_loc"] = (fname, phys_lineno)
                            else:
                                _entry["infer_class"] = cls_ref
                                _entry["infer_loc"] = (fname, phys_lineno)
                        # Iter13 Step1: when conditional_mode is non-default and
                        # this assignment lives in the OTHER branch, skip the
                        # actual write so the chosen mode wins.
                        _skip = (_active_branch is not None
                                 and conditional_mode != "default"
                                 and _active_branch != conditional_mode)
                        if not _skip:
                            attrs[attr_name] = cls_ref
                            attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                    elif native_cls_ref:
                        attrs[attr_name] = native_cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        torch_native_module_classes.add(native_cls_ref)
                    else:
                        # Detect LG input source modules:
                        # LG.feature_column(...), LG.dense_feature(...),
                        # LG.get_sample_rate(...), LG.get_bias(...)
                        _ast_lg = _fe.parse_local_lg_assign(_local_stmt) if _fe else None
                        if _ast_lg:
                            attr_name = _ast_lg.get("attr")
                            attrs[attr_name] = "__LG_InputSource"
                            attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                            input_source_attrs[cname].add(attr_name)

                # Detect fc_dict[...].get_vector(...) assigned to self.xxx
                # Pattern: self.xxx = fc_dict[slot].get_vector(...) or similar
                _ast_fc_gv = _fe.parse_local_fc_get_vector_assign(_local_stmt) if _fe else None
                if _ast_fc_gv:
                    attr_name = _ast_fc_gv.get("attr")
                    if attr_name not in attrs:
                        attrs[attr_name] = "__LG_InputSource"
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        input_source_attrs[cname].add(attr_name)

                _ast_container_ctor = _fe.parse_local_container_ctor(_local_stmt) if _fe else None
                if _ast_container_ctor and _ast_container_ctor.get("kind") == "ModuleList":
                    cont = _ast_container_ctor.get("attr")
                    _record_container_kind(cname, cont, "ModuleList")
                    _call = _ast_container_ctor.get("call")
                    _arg0 = _call.args[0] if _call and _call.args else None
                    if isinstance(_arg0, ast.ListComp) and isinstance(_arg0.elt, ast.Call):
                        _generators = _arg0.generators or []
                        _gen0 = _generators[0] if len(_generators) == 1 else None
                        _cls_ref = _node_to_text(_arg0.elt.func).split('.')[-1]
                        _n = None
                        if _gen0 and isinstance(_gen0.iter, ast.Call) and (_fe._expr_leaf_name(_gen0.iter.func) == "range") and len(_gen0.iter.args) == 1:
                            _n = _eval_int_node(_gen0.iter.args[0], fname, cname, mname, parent_cls=preferred_root, parent_attr='stack')
                        # Phase E1.1: literal list iter, e.g.
                        # ``[Cls(name) for name in ['a', 'b', 'c']]``
                        if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.List):
                            _n = len(_gen0.iter.elts)
                        # Phase E1.1: file-level / local UPPER_CASE str-list constant
                        # (resolved via ConstantTable.file_str_list_consts when available).
                        if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.Name):
                            _len = _eval_list_len_node(_gen0.iter, fname, cname, mname)
                            if _len is not None:
                                _n = _len
                        if _n is not None and _cls_ref in nn_module_classes:
                            attrs.setdefault(cont, _cls_ref)
                            attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                            for i in range(_n):
                                _ensure_elem(cont, cname, _elem_attr_list(cont, i), _cls_ref, fname, phys_lineno, attrs)
                    elif isinstance(_arg0, ast.List):
                        for i, _elt in enumerate(_arg0.elts):
                            if isinstance(_elt, ast.Call):
                                cls_ref = _node_to_text(_elt.func).split('.')[-1]
                                if cls_ref in nn_module_classes:
                                    attrs.setdefault(cont, cls_ref)
                                    attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                    _ensure_elem(cont, cname, _elem_attr_list(cont, i), cls_ref, fname, phys_lineno, attrs)

                if _ast_container_ctor and _ast_container_ctor.get("kind") == "ModuleDict":
                    cont = _ast_container_ctor.get("attr")
                    _record_container_kind(cname, cont, "ModuleDict")
                    _call = _ast_container_ctor.get("call")
                    _arg0 = _call.args[0] if _call and _call.args else None
                    if isinstance(_arg0, ast.Dict):
                        for _k_node, _v_node in zip(_arg0.keys, _arg0.values):
                            if not isinstance(_v_node, ast.Call):
                                continue
                            _key_text = _node_to_text(_k_node)
                            k = _normalize_str_key(_key_text)
                            cls_ref = _node_to_text(_v_node.func).split('.')[-1]
                            if k is not None and cls_ref in nn_module_classes:
                                attrs.setdefault(cont, cls_ref)
                                attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                _ensure_elem(cont, cname, _elem_attr_dict(cont, k), cls_ref, fname, phys_lineno, attrs)
                # ModuleList append: self.xxx.append(ClassName(...))
                _ast_append_ctor = _fe.parse_local_append_ctor(_local_stmt) if _fe else None
                if _ast_append_ctor:
                    attr_name = _ast_append_ctor.get("attr")
                    cls_ref_full = _ast_append_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    if cls_ref in nn_module_classes:
                        # 保留基线：container 名仍映射到子类
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        # 追加展开：container[idx]
                        idx = len(container_elems[cname][attr_name])
                        _ensure_elem(attr_name, cname, _elem_attr_list(attr_name, idx), cls_ref, fname, phys_lineno, attrs)
                        # If we did not previously see a Module-typed declaration for
                        # this attr, fall back to ``list`` (could be plain list of Modules).
                        _record_container_kind(cname, attr_name, "list")
                # ModuleDict / list assignment: self.xxx[key] = ClassName(...)
                _ast_subscript_ctor = _fe.parse_local_subscript_ctor(_local_stmt) if _fe else None
                if _ast_subscript_ctor:
                    attr_name = _ast_subscript_ctor.get("attr")
                    key_expr = _ast_subscript_ctor.get("key_expr") or ""
                    cls_ref_full = _ast_subscript_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    if cls_ref in nn_module_classes:
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        # `self.xx[key] = Module(...)` is most commonly used with
                        # plain ``dict`` instances.  If we already saw a
                        # ``ModuleDict`` declaration earlier, _record_container_kind
                        # will keep the higher-priority kind.
                        _record_container_kind(cname, attr_name, "dict")
                        k = _normalize_str_key(key_expr)
                        if k is not None:
                            _ensure_elem(attr_name, cname, _elem_attr_dict(attr_name, k), cls_ref, fname, phys_lineno, attrs)
                        else:
                            # Iter13 Step2: try to resolve dynamic key to a list
                            # of string literals via class str-list attrs, local
                            # for-loop variable, or file/global UPPER_CASE
                            # constants. If successful, expand into per-key
                            # elem attrs (no [*] fallback). Otherwise warn,
                            # record an unresolved entry, and fall back to
                            # the legacy `[*]` wildcard.
                            _keys = _resolve_str_list(
                                key_expr, fname, cname, _loop_var_to_str_items)
                            if _keys:
                                for _k in _keys:
                                    _ensure_elem(attr_name, cname,
                                                 _elem_attr_dict(attr_name, _k),
                                                 cls_ref, fname, phys_lineno, attrs)
                            else:
                                _reason = "non-literal key expression"
                                print(f"[WARN] ModuleDict key not enumerable: "
                                      f"class={cname} attr={attr_name} expr={key_expr!r}",
                                      file=sys.stderr)
                                print(f"[ERROR] Cannot statically enumerate keys for "
                                      f"{cname}.{attr_name}, reason: {_reason}",
                                      file=sys.stderr)
                                unresolved_moduledict.append({
                                    "class": cname,
                                    "attr": attr_name,
                                    "expr": key_expr,
                                    "reason": _reason,
                                    "file": fname,
                                    "line": phys_lineno,
                                })
                                # 动态 key：补一个通配节点，保证 forward 的依赖能连上
                                _ensure_elem(attr_name, cname, f"{attr_name}[*]", cls_ref, fname, phys_lineno, attrs)
                # Iter16: detect local-variable constructor assignments
                #     var = ClassName(...)
                # where ClassName is a known nn.Module class and `var` is NOT
                # `self.xxx` (the `self.xxx = Cls(...)` form is handled by m
                # above). The collected entries are looked up by the two-step
                # setattr extension below, so that
                #     combine_tower = DenseTower(...)
                #     setattr(self, f"combine_tower_mha_{name}", combine_tower)
                # is treated as a direct setattr-with-ctor.
                _ast_local_ctor = _fe.parse_local_var_ctor(_local_stmt) if _fe else None
                if _ast_local_ctor:
                    _lv_name = _ast_local_ctor.get("name")
                    _lv_cls = (_ast_local_ctor.get("class_full") or "").split('.')[-1]
                    # Avoid clobbering reserved names; skip `self`/`super`.
                    if (_lv_name not in ("self", "super")
                            and _lv_cls in nn_module_classes):
                        _local_var_ctor[_lv_name] = (_lv_cls, fname, phys_lineno)

                # setattr(self, name_var_or_literal, ClassName(...))
                # - If name is a literal string, register the real attr name.
                # - Iter11: if name is a loop variable iterating over a known literal
                #   string list (e.g. `for name in self.xxx_names: setattr(self, name, Cls(...))`),
                #   expand into per-element real attrs.
                # - Otherwise, register a synthetic attr ``<ClassName>`` (i.e. the class name itself,
                #   no prefix) so the child class is at least connected in the DAG.
                #   We additionally track this attr in `dynamic_attrs_per_class` so that the
                #   downstream data-flow analyzer can wildcard-route `getattr(self, dynamic_var)(...)`
                #   calls to these dynamically-registered modules.
                # Iter16: also support the two-step pattern where the third
                # argument is a bare local variable previously assigned via
                # `<var> = ClassName(...)` (see _local_var_ctor above).
                _ast_setattr_ctor = _fe.parse_local_setattr_ctor(_local_stmt) if _fe else None
                _is_two_step = False
                if _ast_setattr_ctor:
                    name_expr = (_ast_setattr_ctor.get("name_expr") or "").strip()
                    if _ast_setattr_ctor.get("two_step"):
                        _bv = _ast_setattr_ctor.get("local_var")
                        if _bv in _local_var_ctor:
                            cls_ref_full = _local_var_ctor[_bv][0]
                            _is_two_step = True
                        else:
                            cls_ref_full = ""
                    else:
                        cls_ref_full = _ast_setattr_ctor.get("class_full") or ""
                    if cls_ref_full:
                        cls_ref = cls_ref_full.split('.')[-1]
                        if cls_ref in nn_module_classes:
                            # name literal: 'xxx' / "xxx" / f'xxx'(no {}) / f"xxx"(no {})
                            real_attr = None
                            _fstr_handled = False
                            try:
                                _name_node = ast.parse(name_expr, mode='eval').body
                            except SyntaxError:
                                _name_node = None
                            if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                real_attr = _name_node.value
                            elif isinstance(_name_node, ast.JoinedStr):
                                # f-string with no interpolations → all values are
                                # ast.Constant strings; concatenate them.
                                if all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                       for v in _name_node.values):
                                    real_attr = ''.join(v.value for v in _name_node.values)
                                else:
                                    # Phase B: f-string with loop variable interpolation
                                    _fstr_expanded = _expand_fstring_with_loop_vars(
                                        _name_node, _loop_var_to_str_items)
                                    if _fstr_expanded:
                                        for _real in _fstr_expanded:
                                            attrs[_real] = cls_ref
                                            attr_def_loc.setdefault((cname, _real), (fname, phys_lineno))
                                        _fstr_handled = True
                            if real_attr:
                                attrs[real_attr] = cls_ref
                                attr_def_loc.setdefault((cname, real_attr), (fname, phys_lineno))
                            elif not _fstr_handled and name_expr in _loop_var_to_str_items:
                                # Iter11: expand using the literal string list driven by the loop variable.
                                for _real in _loop_var_to_str_items[name_expr]:
                                    attrs[_real] = cls_ref
                                    attr_def_loc.setdefault((cname, _real), (fname, phys_lineno))
                            elif not _fstr_handled:
                                # Iter11: use the class name itself as the synthetic attr name
                                # (no `__setattr_` prefix). Tracked in dynamic_attrs_per_class.
                                synth_attr = cls_ref
                                # Only register if no real literal attr already maps to this class
                                if synth_attr not in attrs and not any(v == cls_ref for v in attrs.values()):
                                    attrs[synth_attr] = cls_ref
                                    attr_def_loc.setdefault((cname, synth_attr), (fname, phys_lineno))
                                    dynamic_attrs_per_class[cname].add(synth_attr)
        class_attrs[cname] = attrs

    # ------------------------------------------------------------------
    # 二次扫描（保守增强）：识别 range(N) 循环中的 `var = ClassName(...)` + `self.list.append(var)`
    # 用于展开像 Transformer.resblocks 这种"先构造到局部变量再 append"的写法。
    # 仅当 N 可解析为 int 时展开为 N 个 [i] 节点；解析失败则不影响基线（仍保留 container 映射）。
    # ------------------------------------------------------------------

    # AST helpers — replace the legacy regexes _FOR_RANGE_RE / _FOR_ENUM_RE /
    # _FOR_ITER_RE / _ASSIGN_CTOR_RE / _APPEND_VAR_RE / _APPEND_CTOR_RE /
    # _SELF_SET_RE.
    def _parse_for_header(_text: str):
        """Return (kind, base_indent, iter_expr) for a single ``for ...:`` line.

        kind: 'range' | 'enumerate' | 'iter' | None.
        Returns None when the line is not a parseable for-header.
        """
        if not _text:
            return None
        _stripped = _text.rstrip()
        _left = _stripped.lstrip()
        if not _left.startswith('for ') or not _stripped.endswith(':'):
            return None
        try:
            _mod = ast.parse(_stripped + "\n    pass")
        except SyntaxError:
            return None
        if not (_mod.body and isinstance(_mod.body[0], ast.For)):
            return None
        _f = _mod.body[0]
        _indent = len(_text) - len(_text.lstrip())
        _it = _f.iter
        # range(...)
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'range'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('range', _indent, _expr)
        # enumerate(<iter>)
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'enumerate'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('enumerate', _indent, _expr)
        # plain `for x in <Name|Attr>:`  — only Name/dotted Attr (no Call).
        if isinstance(_f.target, ast.Name) and isinstance(_it, (ast.Name, ast.Attribute)):
            try:
                _expr = ast.unparse(_it)
            except Exception:
                _expr = ""
            return ('iter', _indent, _expr)
        return None

    def _parse_assign_ctor_line(_text: str):
        """For ``var = ClassName(...)`` (simple Name target, Call value with
        Name/Attribute func), return (var_name, class_leaf). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Assign):
            return None
        _a = _mod.body[0]
        if len(_a.targets) != 1 or not isinstance(_a.targets[0], ast.Name):
            return None
        if not isinstance(_a.value, ast.Call):
            return None
        _f = _a.value.func
        if isinstance(_f, ast.Name):
            _leaf = _f.id
        elif isinstance(_f, ast.Attribute):
            _leaf = _f.attr
        else:
            return None
        return (_a.targets[0].id, _leaf)

    def _parse_append_var_line(_text: str):
        """For ``self.cont.append(var)`` (Name arg) return (cont, var). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Expr):
            return None
        _c = _mod.body[0].value
        if not isinstance(_c, ast.Call):
            return None
        _func = _c.func
        if not (isinstance(_func, ast.Attribute) and _func.attr == 'append'):
            return None
        _owner = _func.value
        if not (isinstance(_owner, ast.Attribute)
                and isinstance(_owner.value, ast.Name)
                and _owner.value.id == 'self'):
            return None
        if len(_c.args) != 1 or _c.keywords:
            return None
        if not isinstance(_c.args[0], ast.Name):
            return None
        return (_owner.attr, _c.args[0].id)

    def _parse_append_ctor_line(_text: str):
        """For ``self.cont.append(ClassName(...))`` return (cont, class_leaf). Else None."""
        try:
            _mod = ast.parse(_text)
        except SyntaxError:
            return None
        if not _mod.body or not isinstance(_mod.body[0], ast.Expr):
            return None
        _c = _mod.body[0].value
        if not isinstance(_c, ast.Call):
            return None
        _func = _c.func
        if not (isinstance(_func, ast.Attribute) and _func.attr == 'append'):
            return None
        _owner = _func.value
        if not (isinstance(_owner, ast.Attribute)
                and isinstance(_owner.value, ast.Name)
                and _owner.value.id == 'self'):
            return None
        if len(_c.args) < 1:
            return None
        _arg0 = _c.args[0]
        if not isinstance(_arg0, ast.Call):
            return None
        _af = _arg0.func
        if isinstance(_af, ast.Name):
            _leaf = _af.id
        elif isinstance(_af, ast.Attribute):
            _leaf = _af.attr
        else:
            return None
        return (_owner.attr, _leaf)

    # Iter17: per-class container -> driving kw param name (for per-instance prune).
    # class_container_kw[cname][container_attr] = kw_name (the ctor kw whose
    # list-length determines the iteration length).
    class_container_kw = defaultdict(dict)

    def _resolve_iter_len(expr_node, fname: str, cname: str, mname: str = "__init__",
                          parent_cls: str = None, parent_attr: str = None):
        n = _eval_list_len_node(expr_node, fname, cname, mname, parent_cls, parent_attr)
        kw_name = None
        if isinstance(expr_node, ast.Name):
            kw_name = expr_node.id
        elif (isinstance(expr_node, ast.Attribute)
              and isinstance(expr_node.value, ast.Name)
              and expr_node.value.id == "self"):
            kw_name = self_to_param.get(expr_node.attr)
        return n, kw_name

    def _resolve_range_n(expr_node, fname: str, cname: str, mname: str = "__init__",
                         parent_cls: str = None, parent_attr: str = None):
        return _eval_int_node(expr_node, fname, cname, mname, parent_cls, parent_attr)

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        attrs = class_attrs.get(cname, {})
        _fe = _get_ast_frontend(fname)
        # 建立 self.attr = param 的映射（如 self.layers = layers）
        self_to_param = _fe.get_self_param_aliases(cname, "__init__") if _fe else {}

        # Iter17 guard: collect the set of self.<attr> containers that are
        # accessed via dynamic indexing (`self.x[idx]` where idx is NOT a
        # literal integer) in any non-__init__ method of this class.  When a
        # container is used dynamically the analyzer's downstream logic
        # already handles it via a wildcard route; expanding it to per-index
        # nodes via AST loop expansion would only add nodes the dynamic call
        # can't supply var_history evidence for (Rule6_out regression).
        dyn_indexed_containers = _fe.get_dynamic_indexed_self_attrs(cname) if _fe else set()

        # 扫描各方法，找 for-range 的 append(var)
        for mname, (ms, me) in info['methods'].items():
            method_lines = lines[ms - 1: min(me, len(lines))]
            _ast_loop_records = _fe.get_loop_expansion_records(cname, mname) if _fe else []
            if _ast_loop_records:
                _ast_line_to_records = defaultdict(list)
                for _rec in _ast_loop_records:
                    _ast_line_to_records[_rec.get("line")].append(_rec)
                for _line_no in sorted(_ast_line_to_records):
                    for _rec in _ast_line_to_records[_line_no]:
                        _loop_kind = _rec.get("loop_kind")
                        _iter_expr = (_rec.get("iter_expr") or "").strip()
                        _iter_expr_node = _eval_expr_node(_iter_expr)
                        _kw_for_loop = None
                        if _loop_kind == "range":
                            n = _resolve_range_n(_iter_expr_node, fname, cname, mname)
                        elif _loop_kind == "enumerate":
                            n, _kw_for_loop = _resolve_iter_len(_iter_expr_node, fname, cname, mname)
                        else:
                            if _iter_expr.startswith('self.') or _iter_expr in ('range',):
                                continue
                            n, _kw_for_loop = _resolve_iter_len(_iter_expr_node, fname, cname, mname)
                        if n is None:
                            continue
                        cont = _rec.get("container_attr")
                        cls_ref = _rec.get("class_leaf")
                        if not cont or cls_ref not in nn_module_classes:
                            continue
                        if (_loop_kind in {"enumerate", "iter"}) and cont in dyn_indexed_containers:
                            continue
                        attrs.setdefault(cont, cls_ref)
                        attr_def_loc.setdefault((cname, cont), (fname, _line_no))
                        for k in range(n):
                            _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, _line_no, attrs)
                        _record_container_kind(cname, cont, "ModuleList")
                        if _kw_for_loop is not None:
                            class_container_kw[cname][cont] = _kw_for_loop
                continue

            # AST loop extraction should be the primary path. Keep a regex
            # fallback here for older sources whose loop bodies are not yet
            # covered by get_loop_expansion_records(), but route evaluation
            # through ConstantResolver only.
            _logical_method = _join_logical_lines(method_lines, ms)
            var_to_cls = {}
            i = 0
            while i < len(method_lines):
                raw = method_lines[i]
                _hdr = _parse_for_header(raw)
                if _hdr is None:
                    i += 1
                    continue
                _kind, base_indent, _iter_expr = _hdr
                _kw_for_loop = None
                if _kind == 'range':
                    n = _resolve_range_n(_eval_expr_node(_iter_expr), fname, cname, mname)
                elif _kind == 'enumerate':
                    n, _kw_for_loop = _resolve_iter_len(_eval_expr_node(_iter_expr), fname, cname, mname)
                else:  # 'iter'
                    if _iter_expr.startswith('self.') or _iter_expr in ('range',):
                        i += 1
                        continue
                    n, _kw_for_loop = _resolve_iter_len(_eval_expr_node(_iter_expr), fname, cname, mname)
                _logical_starts = {start_ln: text for (start_ln, text) in _logical_method}
                j = i + 1
                while j < len(method_lines):
                    body_raw = method_lines[j]
                    if not body_raw.strip():
                        j += 1
                        continue
                    ind = len(body_raw) - len(body_raw.lstrip())
                    if ind <= base_indent:
                        break
                    am = _parse_assign_ctor_line(body_raw)
                    if am is not None:
                        vname, cls_ref = am
                        if cls_ref in nn_module_classes:
                            var_to_cls[vname] = cls_ref
                    ap = _parse_append_var_line(body_raw)
                    ap_ctor = None
                    _phys_at_j = ms + j
                    _joined_text = _logical_starts.get(_phys_at_j)
                    if ap is None and _joined_text:
                        ap_ctor = _parse_append_ctor_line(' ' * (base_indent + 1) + _joined_text.lstrip())
                    if ap is None and ap_ctor is None:
                        ap_ctor = _parse_append_ctor_line(body_raw)
                    if ap is not None and n is not None:
                        cont, vname = ap
                        cls_ref = var_to_cls.get(vname)
                        if cls_ref and cls_ref in nn_module_classes:
                            attrs.setdefault(cont, cls_ref)
                            attr_def_loc.setdefault((cname, cont), (fname, ms + j))
                            for k in range(n):
                                _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, ms + j, attrs)
                            _record_container_kind(cname, cont, "ModuleList")
                            if _kw_for_loop is not None:
                                class_container_kw[cname][cont] = _kw_for_loop
                    elif ap_ctor is not None and n is not None and _kind in ('enumerate', 'iter'):
                        cont, cls_ref = ap_ctor
                        if cont in dyn_indexed_containers:
                            j += 1
                            continue
                        if cls_ref in nn_module_classes:
                            attrs.setdefault(cont, cls_ref)
                            attr_def_loc.setdefault((cname, cont), (fname, ms + j))
                            for k in range(n):
                                _ensure_elem(cont, cname, _elem_attr_list(cont, k), cls_ref, fname, ms + j, attrs)
                            _record_container_kind(cname, cont, "ModuleList")
                            if _kw_for_loop is not None:
                                class_container_kw[cname][cont] = _kw_for_loop
                    j += 1
                i = j

        class_attrs[cname] = attrs

    # ------------------------------------------------------------------
    # Iter11 unroll2 — Rule 2: range + f-string setattr expansion
    #   for i in range(N):                               # N from literal/CONST or
    #       setattr(self, f"query_tower_{i}", Cls(...))  # from ctor kw param
    # If N is resolvable, expand into N real attrs (e.g. query_tower_0..N-1).
    # Otherwise, fall back to a single wildcard synthetic attr (`<prefix>[*]`)
    # tracked in dynamic_attrs_per_class so getattr-based calls can wildcard-route.
    # This pass runs AFTER the main scan so `ctor_kw_int_args` is fully populated.
    # ------------------------------------------------------------------
    def _parse_for_range_var(_text: str):
        """For ``for <var> in range(<expr>):`` return (base_indent, var, expr_text).
        Else None."""
        if not _text:
            return None
        _stripped = _text.rstrip()
        if not _stripped.lstrip().startswith('for ') or not _stripped.endswith(':'):
            return None
        try:
            _mod = ast.parse(_stripped + "\n    pass")
        except SyntaxError:
            return None
        if not (_mod.body and isinstance(_mod.body[0], ast.For)):
            return None
        _f = _mod.body[0]
        if not isinstance(_f.target, ast.Name):
            return None
        _it = _f.iter
        if not (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'range'
                and len(_it.args) >= 1):
            return None
        try:
            _expr = ast.unparse(_it.args[0])
        except Exception:
            _expr = ""
        _indent = len(_text) - len(_text.lstrip())
        return (_indent, _f.target.id, _expr)

    def _find_setattr_fstr_call(_body_line: str):
        """Find the first ``setattr(self, f"...", ClassName(...))`` call in a
        joined logical line. Returns (template_node, class_leaf, template_text)
        where template_node is the ast.JoinedStr literal and template_text is
        ast.unparse(template_node) (best effort). Returns None if not found.
        """
        if not _body_line:
            return None
        try:
            _mod = ast.parse(_body_line.strip())
        except SyntaxError:
            return None
        for _n in ast.walk(_mod):
            if not isinstance(_n, ast.Call):
                continue
            _func = _n.func
            if not (isinstance(_func, ast.Name) and _func.id == 'setattr'):
                continue
            if len(_n.args) < 3:
                continue
            _arg0, _arg1, _arg2 = _n.args[0], _n.args[1], _n.args[2]
            if not (isinstance(_arg0, ast.Name) and _arg0.id == 'self'):
                continue
            if not isinstance(_arg1, ast.JoinedStr):
                continue
            if not isinstance(_arg2, ast.Call):
                continue
            _vf = _arg2.func
            if isinstance(_vf, ast.Name):
                _leaf = _vf.id
            elif isinstance(_vf, ast.Attribute):
                _leaf = _vf.attr
            else:
                continue
            return (_arg1, _leaf)
        return None

    def _joinedstr_references_var(_jstr: ast.JoinedStr, var_name: str) -> bool:
        """True if any FormattedValue inside <_jstr> references the bare Name <var_name>."""
        for _v in _jstr.values:
            if isinstance(_v, ast.FormattedValue) and isinstance(_v.value, ast.Name) \
                    and _v.value.id == var_name:
                return True
        return False

    def _expand_fstring_template_ast(_jstr: ast.JoinedStr, loop_var: str, idx_value):
        """AST-based version of _expand_fstring_template.
        Substitutes references to <loop_var> in FormattedValue with idx_value.
        Returns None if any FormattedValue references something other than
        <loop_var>.
        Supports format spec ``[0]?<width>d`` for zero-padding.
        """
        out_parts = []
        for _v in _jstr.values:
            if isinstance(_v, ast.Constant) and isinstance(_v.value, str):
                out_parts.append(_v.value)
                continue
            if not isinstance(_v, ast.FormattedValue):
                return None
            if not (isinstance(_v.value, ast.Name) and _v.value.id == loop_var):
                return None
            # Format spec — only digit-width supported.
            _spec_text = None
            if _v.format_spec is not None:
                if not isinstance(_v.format_spec, ast.JoinedStr):
                    return None
                _parts = []
                for _sv in _v.format_spec.values:
                    if isinstance(_sv, ast.Constant) and isinstance(_sv.value, str):
                        _parts.append(_sv.value)
                    else:
                        return None
                _spec_text = ''.join(_parts)
            if _spec_text:
                # Match the original "0?<digits>d" semantics.
                _spec = _spec_text.strip()
                _digits = _spec[:-1] if _spec.endswith('d') else None
                if _digits is not None and _digits.startswith('0'):
                    _digits = _digits[1:]
                if _digits is not None and _digits.isdigit():
                    out_parts.append(str(idx_value).zfill(int(_digits)))
                else:
                    out_parts.append(str(idx_value))
            else:
                out_parts.append(str(idx_value))
        return ''.join(out_parts)

    def _wildcard_template_from_jstr(_jstr: ast.JoinedStr) -> str:
        """Replace every FormattedValue in <_jstr> with ``*`` and return the resulting string."""
        _parts = []
        for _v in _jstr.values:
            if isinstance(_v, ast.Constant) and isinstance(_v.value, str):
                _parts.append(_v.value)
            else:
                _parts.append('*')
        return ''.join(_parts)

    for (fname, cname), info in class_map.items():
        if cname not in nn_module_classes:
            continue
        lines = source_files.get(fname, [])
        attrs = class_attrs.get(cname, {})

        for mname, (ms, me) in info['methods'].items():
            method_lines = lines[ms - 1: min(me, len(lines))]
            i = 0
            while i < len(method_lines):
                raw = method_lines[i]

                _hdr = _parse_for_range_var(raw)
                if _hdr is None:
                    i += 1
                    continue
                base_indent, loop_var, range_expr = _hdr
                n = _resolve_range_n(_eval_expr_node(range_expr), fname, cname, mname)

                # Walk body and join multi-line statements (parenthesis-aware)
                # so that setattr(...) spanning multiple lines is matched as one.
                body_buf = ""
                body_buf_start = None
                body_open = 0
                joined = []  # list of (start_lineno, joined_line)
                j = i + 1
                while j < len(method_lines):
                    body_raw = method_lines[j]
                    if not body_raw.strip():
                        j += 1
                        continue
                    ind = len(body_raw) - len(body_raw.lstrip())
                    if not body_buf and ind <= base_indent:
                        break
                    stripped = _strip_inline_comment(body_raw.strip()).strip()
                    if not stripped:
                        j += 1
                        continue
                    if not body_buf:
                        body_buf_start = ms + j
                    body_buf += (" " if body_buf else "") + stripped
                    body_open += stripped.count('(') + stripped.count('[') + stripped.count('{')
                    body_open -= stripped.count(')') + stripped.count(']') + stripped.count('}')
                    if body_open <= 0:
                        joined.append((body_buf_start, body_buf))
                        body_buf = ""
                        body_buf_start = None
                        body_open = 0
                    j += 1
                if body_buf:
                    joined.append((body_buf_start, body_buf))

                for body_lineno, body_line in joined:
                    _hit = _find_setattr_fstr_call(body_line)
                    if _hit is None:
                        continue
                    template_jstr, cls_ref = _hit
                    if cls_ref not in nn_module_classes:
                        continue
                    # Skip f-strings that DO NOT reference the loop variable —
                    # those aren't this rule's domain (e.g. setattr(self, f"{name}_out", v)
                    # with `name` from a different loop). Let other passes / wildcards
                    # handle them.
                    if not _joinedstr_references_var(template_jstr, loop_var):
                        continue
                    if n is not None:
                        # Bounded expansion
                        for idx in range(n):
                            real = _expand_fstring_template_ast(template_jstr, loop_var, idx)
                            if real is None:
                                # Unsupported interpolation — fall back to wildcard
                                synth = _wildcard_template_from_jstr(template_jstr) or cls_ref
                                if synth not in attrs:
                                    attrs[synth] = cls_ref
                                    attr_def_loc.setdefault((cname, synth), (fname, body_lineno))
                                    dynamic_attrs_per_class[cname].add(synth)
                                break
                            if real not in attrs:
                                attrs[real] = cls_ref
                                attr_def_loc.setdefault((cname, real), (fname, body_lineno))
                    else:
                        # Unbounded — conservative wildcard node:
                        # replace {loop_var}/{loop_var:fmt} with '*'
                        synth = _wildcard_template_from_jstr(template_jstr) or cls_ref
                        if synth not in attrs:
                            attrs[synth] = cls_ref
                            attr_def_loc.setdefault((cname, synth), (fname, body_lineno))
                            dynamic_attrs_per_class[cname].add(synth)

                i = j

        class_attrs[cname] = attrs


    # Build data dependency edges between child modules
    dep_edges, dep_edge_locs, split_info = _build_data_dependency_edges(
        source_files,
        class_map,
        class_attrs,
        class_str_list_attrs,
        dynamic_attrs_per_class,
        ast_frontends=ast_frontends,
    )

    # Inject split nodes into class_attrs so they appear as DAG nodes.
    # e.g. if GAUBlock has split_info = {"norm": ["norm#0", "norm#1"]},
    # add "norm#0" and "norm#1" to class_attrs["GAUBlock"] with the same class ref as "norm".
    for cname, smap in split_info.items():
        if cname not in class_attrs:
            continue
        for orig_attr, split_names in smap.items():
            cls_ref = class_attrs[cname].get(orig_attr)
            if cls_ref is None:
                continue
            # Remove the original attr from class_attrs (replaced by splits)
            del class_attrs[cname][orig_attr]
            for sn in split_names:
                class_attrs[cname][sn] = cls_ref

    # Build tree using forward() call order + __init__ attrs
    tree = {}
    all_child_classes = set()
    all_tree_classes = set(nn_module_classes) | set(torch_native_module_classes)
    for cname in all_tree_classes:
        if cname not in class_attrs:
            class_attrs[cname] = {}
        attrs = class_attrs[cname]
        call_order = module_call_order.get(cname, [])

        # Order children by forward() call order
        children_ordered = []
        seen = set()
        for attr in call_order:
            if attr in attrs and attrs[attr] not in seen:
                children_ordered.append(attrs[attr])
                seen.add(attrs[attr])
                all_child_classes.add(attrs[attr])
        # Add remaining attrs not called in forward
        for attr, cls_ref in attrs.items():
            if cls_ref not in seen:
                children_ordered.append(cls_ref)
                seen.add(cls_ref)
                all_child_classes.add(cls_ref)

        tree[cname] = {"children": children_ordered, "attrs": attrs,
                       "dep_edges": dep_edges.get(cname, []),
                       "dep_edge_locs": dep_edge_locs.get(cname, {}),
                       "containers": {},
                       "is_torch_native": cname in torch_native_module_classes,
                       # Iter16: expose dynamic setattr-registered attrs so
                       # downstream scanners (e.g. _scan_root_result_edges)
                       # can fall back to wildcard producers when seeing
                       # `getattr(self, <non-literal>)`.
                       "dynamic_setattr_attrs": sorted(dynamic_attrs_per_class.get(cname, set())),
                       # Iter17: per-class container -> kw param driving its
                       # iteration length. Used by build_dag_recursive for
                       # per-instance children pruning.
                       "container_kw": dict(class_container_kw.get(cname, {}))}

    # Find root(s): classes that are NOT children of any other class
    roots = [c for c in nn_module_classes if c not in all_child_classes and c in tree]
    if not roots:
        roots = list(tree.keys())[:1]

    # Prefer roots that have children (real model roots), sort by descendant count
    def count_descendants(cname, visited=None):
        if visited is None:
            visited = set()
        if cname in visited or cname not in tree:
            return 0
        visited.add(cname)
        total = len(tree[cname]["children"])
        for child in tree[cname]["children"]:
            total += count_descendants(child, visited)
        return total

    roots_with_children = [r for r in roots if tree.get(r, {}).get("children")]
    if roots_with_children:
        roots = sorted(roots_with_children, key=lambda r: -count_descendants(r))
    # If "RootModule" is present, put it first
    if "RootModule" in roots:
        roots = ["RootModule"] + [r for r in roots if r != "RootModule"]
    # Honour caller-supplied preferred root (e.g. derived from timeline wrapper name)
    if preferred_root and preferred_root in roots:
        roots = [preferred_root] + [r for r in roots if r != preferred_root]

    # Filter tree to only include classes reachable from the primary root.
    # This eliminates orphan classes that are defined but never instantiated.
    if roots:
        reachable = set()
        def _collect_reachable(cname):
            if cname in reachable or cname not in tree:
                return
            reachable.add(cname)
            for child in tree[cname].get("children", []):
                _collect_reachable(child)
        for r in roots[:1]:  # Only from primary root
            _collect_reachable(r)
        # Keep only reachable classes in tree, filter roots
        tree = {k: v for k, v in tree.items() if k in reachable}
        roots = [r for r in roots if r in reachable]
        if not roots and reachable:
            roots = sorted(reachable, key=lambda r: -count_descendants(r))[:1]

    # Stash attr definition locations on the tree for downstream consumers
    # (HTML flowchart click-interactions). Format: tree[cname]["attr_def_loc"][attr] = (fname, lineno)
    for cname in tree:
        per_class = {}
        for attr in tree[cname].get("attrs", {}):
            loc = attr_def_loc.get((cname, attr))
            if loc:
                per_class[attr] = loc
        tree[cname]["attr_def_loc"] = per_class

    # Build first_call_loc: first forward() call line for each attr.
    # Strategy 1: derive from dep_edge_locs (edges between siblings).
    # Strategy 2: ask ASTFrontend for the earliest reachable call site,
    # including dynamic getattr(self, f"...") calls via pure AST walk.
    for cname in tree:
        edge_locs_c = tree[cname].get("dep_edge_locs", {})
        first_call = {}  # attr -> (file, lineno)
        for (from_a, to_a), ev in edge_locs_c.items():
            if not isinstance(ev, dict):
                continue
            f = ev.get("file", "")
            if from_a and ev.get("from_line"):
                if from_a not in first_call or ev["from_line"] < first_call[from_a][1]:
                    first_call[from_a] = (f, ev["from_line"])
            if to_a and ev.get("to_line"):
                if to_a not in first_call or ev["to_line"] < first_call[to_a][1]:
                    first_call[to_a] = (f, ev["to_line"])
        known_attrs_c = set(tree[cname].get("attrs", {}).keys())
        _ast_first_call_map = {}
        for (fname, cm_name), cm_info in class_map.items():
            if cm_name != cname:
                continue
            _fe = _get_ast_frontend(fname)
            if _fe:
                for _attr_name in known_attrs_c:
                    _loc = _fe.get_first_call_loc(cname, _attr_name)
                    if _loc:
                        _ast_first_call_map[_attr_name] = _loc
            break
        for _attr_name, _loc in _ast_first_call_map.items():
            if _attr_name not in first_call or _loc[1] < first_call[_attr_name][1]:
                first_call[_attr_name] = _loc
        tree[cname]["first_call_loc"] = first_call

    # Stash input_source_attrs on the tree for downstream consumers
    for cname in tree:
        tree[cname]["input_source_attrs"] = input_source_attrs.get(cname, set())
        tree[cname]["static_diagnostics"] = list(_new_eval_diagnostics)

    # Iter14 container-group: stash recorded container kinds on the tree so the
    # DAG renderer can build container parent groups labelled e.g.
    # ``ModuleDict[tokens_module_dict][64]``.  Format:
    # tree[cname]["container_kinds"][container_attr] = "ModuleDict" / "ModuleList" / "dict" / "list"
    for cname in tree:
        per_class = {}
        for (kc, ka), kind in container_kinds.items():
            if kc == cname:
                per_class[ka] = kind
        tree[cname]["container_kinds"] = per_class

    # Iter13 Step1: stash conditional attrs metadata on the tree for caller's
    # tab decision. Filter to attrs where BOTH branches actually recorded a
    # class (single-branch conditionals are not "real" conditional attrs).
    for cname in tree:
        ca = class_conditional_attrs.get(cname, {})
        meaningful = {a: info for a, info in ca.items()
                      if info.get("train_class") and info.get("infer_class")
                      and info.get("train_class") != info.get("infer_class")}
        tree[cname]["conditional_attrs"] = meaningful
    # Iter13 Step2: stash unresolved moduledict warnings on the tree.
    for cname in tree:
        tree[cname]["unresolved_moduledict"] = [u for u in unresolved_moduledict
                                                 if u.get("class") == cname]

    # Iter17: stash per-instance kw-list-lens on each parent class so
    # build_dag_recursive can prune children counts per call site.
    for cname in tree:
        tree[cname]["instance_kw_list_lens"] = {
            attr: dict(kws) for (pcn, attr), kws in instance_kw_list_lens.items()
            if pcn == cname
        }

    # Iter18: stash AST ground-truth lookups on every tree[cname] entry.
    # This is purely additive metadata; the existing regex pass remains the
    # primary driver of edges/expansions.  Downstream consumers can use these
    # to:
    #   • cross-check ModuleList/ModuleDict element type
    #     (``_ast_container_elems[container_attr] -> [child_class, ...]``)
    #   • confirm dynamic setattr child class lists
    #   • spot missing or spurious attrs vs the existing class_attrs
    #
    # Sibling keys are prefixed with ``_ast_`` to avoid clashing with the
    # existing keys consumed by generate_html_flowchart.  We do NOT add
    # synthetic top-level keys to ``tree`` itself (that would break the many
    # ``for cname in tree`` loops downstream).
    if _ast_extractor is not None:
        for cname in tree:
            tree[cname]["_ast_class_attrs"] = dict(_ast_extractor.ast_class_attrs.get(cname, {}))
            tree[cname]["_ast_container_elems"] = {
                k: list(v) for k, v in _ast_extractor.ast_container_elems.get(cname, {}).items()
            }
            tree[cname]["_ast_container_kinds"] = dict(_ast_extractor.ast_container_kinds.get(cname, {}))
            tree[cname]["_ast_dynamic_attrs"] = list(_ast_extractor.ast_dynamic_attrs.get(cname, []))
            tree[cname]["_ast_is_nn_module"] = (cname in _ast_extractor.nn_module_classes)

    return tree, roots, class_map


def _build_runtime_module_hierarchy(events):
    """Build module containment hierarchy from trace events.
    Returns: {class_name: {"children": [child_names], "parents": set(parent_names)}}
    """
    mod_events = []
    for e in events:
        name = e.get('name', '')
        if not name.startswith('nn.Module:') or e.get('ph') != 'X':
            continue
        short = name.replace('nn.Module: ', '').replace('nn.Module:', '')
        short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
        base = re.sub(r'_\d+$', '', short)
        mod_events.append({
            'name': base,
            'ts': e['ts'],
            'dur': e.get('dur', 0),
            'end': e['ts'] + e.get('dur', 0),
            'tid': e.get('tid'),
        })

    parent_child = defaultdict(set)
    child_parent = defaultdict(set)
    all_tids = set(e['tid'] for e in mod_events)
    top_level = set()

    for tid in all_tids:
        tid_events = [e for e in mod_events if e['tid'] == tid]
        tid_events.sort(key=lambda x: (x['ts'], -x['dur']))
        stack = []
        for e in tid_events:
            while stack and stack[-1]['end'] <= e['ts']:
                stack.pop()
            if stack:
                parent_child[stack[-1]['name']].add(e['name'])
                child_parent[e['name']].add(stack[-1]['name'])
            else:
                top_level.add(e['name'])
            stack.append(e)

    hierarchy = {}
    all_names = set(e['name'] for e in mod_events)
    for name in all_names:
        children = sorted(parent_child.get(name, set()))
        parents = child_parent.get(name, set())
        hierarchy[name] = {"children": children, "parents": parents}

    return hierarchy, top_level


def _merge_runtime_into_tree(tree, roots, runtime_hierarchy, runtime_top_level, class_durations):
    """Merge runtime-only modules into the static tree so all timeline modules are connected.
    Runtime modules not in source tree get virtual entries connected via containment edges.
    DDPRootModule -> RootModule connection is established by stripping DDP prefix.
    """
    static_classes = set(tree.keys())
    runtime_classes = set(runtime_hierarchy.keys())
    runtime_only = runtime_classes - static_classes

    if not runtime_only:
        return tree, roots

    # Establish mapping from runtime wrapper to source root
    # DDPRootModule/FSDPRootModule -> RootModule
    wrapper_to_source = {}
    for rmod in runtime_only:
        stripped = re.sub(r'^(FSDP|DDP|DistributedDataParallel|Wrapped)', '', rmod)
        if stripped in static_classes and stripped != rmod:
            wrapper_to_source[rmod] = stripped

    # Add runtime-only modules to tree
    for rmod in runtime_only:
        rinfo = runtime_hierarchy[rmod]
        # Children: only include those that are also runtime-only OR are wrapper targets
        children = []
        attrs = {}
        for child in rinfo["children"]:
            if child in runtime_only or child in wrapper_to_source:
                attr_name = child[0].lower() + child[1:]
                children.append(child)
                attrs[attr_name] = child
            elif child in static_classes:
                attr_name = child[0].lower() + child[1:]
                children.append(child)
                attrs[attr_name] = child

        # If this runtime module wraps a source module, add that as child
        if rmod in wrapper_to_source:
            src_target = wrapper_to_source[rmod]
            if src_target not in children:
                attr_name = src_target[0].lower() + src_target[1:]
                children.append(src_target)
                attrs[attr_name] = src_target

        # Build sequential dep edges (runtime modules are pipeline-sequential)
        dep_edges = []
        attr_names_ordered = list(attrs.keys())
        for i in range(len(attr_names_ordered) - 1):
            dep_edges.append((attr_names_ordered[i], attr_names_ordered[i + 1]))

        tree[rmod] = {"children": children, "attrs": attrs, "dep_edges": dep_edges,
                       "dep_edge_locs": {}, "attr_def_loc": {}}

    # Determine new roots: runtime top-level modules that contain everything
    # Prefer a single root that encompasses all others
    new_roots = []
    # Find the broadest runtime top-level module (most descendants)
    def count_rt_descendants(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name not in tree:
            return 0
        visited.add(name)
        total = len(tree[name]["children"])
        for child in tree[name]["children"]:
            total += count_rt_descendants(child, visited)
        return total

    # Runtime top-level modules become candidate roots
    rt_roots = [r for r in runtime_top_level if r in tree]
    if rt_roots:
        rt_roots_sorted = sorted(rt_roots, key=lambda r: -count_rt_descendants(r))
        # All runtime top-level modules are roots (they are independent execution units)
        new_roots = list(rt_roots_sorted)
        # Check reachability from all runtime roots
        reachable = set()
        def collect_reachable(name, visited):
            if name in visited or name not in tree:
                return
            visited.add(name)
            reachable.add(name)
            for child in tree[name]["children"]:
                collect_reachable(child, visited)
        for rr in rt_roots:
            collect_reachable(rr, set())

        # Add source roots that are NOT reachable from any runtime root
        for sr in roots:
            if sr not in reachable:
                new_roots.append(sr)
    else:
        new_roots = roots

    return tree, new_roots


def _validate_timeline_modules(trace_events, tree):
    """Validate that all user model nn.Module classes in timeline are present in the DAG tree.
    Runtime wrapper modules (not in user source) are expected to be absent.
    Returns validation info dict.
    """
    if not trace_events:
        return None

    timeline_modules = set()
    for e in trace_events:
        name = e.get('name', '')
        if not name.startswith('nn.Module:') or e.get('ph') != 'X':
            continue
        short = name.replace('nn.Module: ', '').replace('nn.Module:', '')
        short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
        base = re.sub(r'_\d+$', '', short)
        timeline_modules.add(base)

    if not timeline_modules:
        return None

    tree_classes = set(tree.keys())
    # Timeline modules that ARE in the source tree (user model modules)
    in_tree = timeline_modules & tree_classes
    # Timeline modules that are NOT in source tree (runtime wrappers)
    runtime_only = timeline_modules - tree_classes

    # Check connectivity: all tree classes should be reachable from root via dep_edges or children
    connected = set()
    def _check_connected(cname, visited=None):
        if visited is None:
            visited = set()
        if cname in visited or cname not in tree:
            return
        visited.add(cname)
        connected.add(cname)
        for child in tree[cname].get("children", []):
            _check_connected(child, visited)
    for root in tree_classes:
        if root not in [c for info in tree.values() for c in info.get("children", [])]:
            _check_connected(root)

    validation = {
        "timeline_total": len(timeline_modules),
        "in_source_tree": sorted(in_tree),
        "runtime_wrappers": sorted(runtime_only),
        "tree_classes": sorted(tree_classes),
        "connected_classes": sorted(connected),
        "all_connected": tree_classes == connected,
    }

    # Print validation summary
    print(f"  📋 Timeline nn.Module 验证:")
    print(f"     Timeline 中共 {len(timeline_modules)} 个 Module 类")
    if in_tree:
        print(f"     ✅ 用户模型 Module (在 DAG 中): {sorted(in_tree)}")
    if runtime_only:
        print(f"     ℹ️  运行时 wrapper Module (不在用户源码中, 无需加入 DAG): {len(runtime_only)} 个")
    disconnected = tree_classes - connected
    if disconnected:
        print(f"     ⚠️  DAG 中未连接的类: {sorted(disconnected)}")
    else:
        print(f"     ✅ DAG 中所有 {len(tree_classes)} 个源码 Module 类均已正确连接")

    # Cross-check: any timeline module that has the same name as a source nn.Module
    # MUST also exist in the DAG tree. If a class is defined in user source but
    # filtered out of the tree (e.g. because the wrong root was picked), surface it
    # as a validation failure so the caller can iterate.
    # We re-load the full nn_module_classes list from class_map by inspecting the
    # tree's union with whatever we know. For a stricter check the caller should
    # have already used preferred_root logic.
    user_source_in_timeline_missing = []
    # We can only detect this with extra info; not done here to avoid import churn.
    # Validation output above is the authoritative pass/fail summary.

    return validation


# ---------------------------------------------------------------------------
# HTML flowchart rendering moved to frontend_html.py.
#
# Historically this file embedded a ~4800 line copy of the HTML/JS template
# generator (lines 7524..12306 in the iter17_timing baseline). That copy has
# been removed in favour of an explicit, one-direction import from the
# sibling ``frontend_html`` module:
#
#     analyze_trace  ──provides─▶  ASTFrontend, _build_class_map, ...
#     frontend_html  ──provides─▶  generate_html_flowchart, *_dual,
#                                  build_timing_data_from_trace,
#                                  _generate_flowchart_html
#
# The four names are re-exported through a module-level ``__getattr__``
# (PEP 562) instead of an eager ``from frontend_html import ...`` so the
# import remains safe even when a downstream caller imports
# ``frontend_html`` first (in which case Python is in the middle of loading
# ``frontend_html`` when this file finishes executing — an eager import
# back into ``frontend_html`` would deadlock with a "partially initialized
# module" error).
#
# The lazy form preserves the public surface area used by ``main()`` below
# as well as by ``testset/test_dag_rules.py`` (which calls
# ``mod.generate_html_flowchart(...)`` after loading this file via
# ``importlib.util.spec_from_file_location``) — both call sites resolve
# the attribute *after* both modules have finished loading, so the lazy
# lookup always sees a fully-initialized ``frontend_html``.
# ---------------------------------------------------------------------------
_FRONTEND_HTML_REEXPORT_NAMES = (
    "generate_html_flowchart",
    "generate_html_flowchart_dual",
    "build_timing_data_from_trace",
    "_generate_flowchart_html",
)


def __getattr__(name):  # noqa: D401 (PEP 562 module-level __getattr__)
    """Lazily re-export ``frontend_html`` entry points.

    Triggered only when ``analyze_trace.<name>`` is accessed for one of the
    HTML rendering functions.  By the time any consumer reaches this lookup
    both modules have finished loading, so we can safely import the symbol
    without re-entering a half-initialised ``frontend_html``.
    """
    if name in _FRONTEND_HTML_REEXPORT_NAMES:
        import frontend_html as _fh
        value = getattr(_fh, name)
        # Cache on this module so repeat lookups are O(1) and so that the
        # importlib proxy used in the ``analyze_trace_runtime`` mode also
        # picks up the name through its live globals delegation.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'analyze_trace' has no attribute {name!r}")


def main():
    # Lazy, function-scope import of the HTML rendering entry points.  See the
    # comment above ``_FRONTEND_HTML_REEXPORT_NAMES`` for why module-level
    # eager imports would risk a circular-import deadlock.  By the time
    # ``main()`` is invoked both modules are fully loaded.
    from frontend_html import (
        generate_html_flowchart_dual,
        build_timing_data_from_trace,
    )

    parser = argparse.ArgumentParser(description="分析 PyTorch 训练 trace JSON 文件")
    parser.add_argument("trace_file", nargs="?", default=None, help="trace JSON 文件路径 (可选，若仅生成源码流程图则不需要)")
    parser.add_argument("--max-level", type=int, default=None, help="最大显示的 Module 层级深度")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 Markdown 报告文件路径")
    parser.add_argument("--no-tree", action="store_true", help="不输出 Module 层级树")
    parser.add_argument("--json-output", type=str, default=None, help="输出 JSON 格式分析结果")
    parser.add_argument("--code-path", type=str, default=None, help="模型源码路径（目录或 .tar.gz）")
    parser.add_argument("--screenshot", action="store_true", help="生成 Chrome Tracing 可视化截图")
    parser.add_argument("--html-flowchart", type=str, default=None, help="生成 HTML 模块流程图路径")
    parser.add_argument("--use-new-eval", action="store_true",
                        help="兼容旧脚本参数；PR3 起 ConstantResolver 已默认启用")
    parser.add_argument("--ab-eval-report", type=str, default=None,
                        help="写入当前求值器状态摘要 JSON（PR3 起不再产出 legacy/new diff）")
    args = parser.parse_args()

    global USE_NEW_EVAL
    USE_NEW_EVAL = True
    if args.use_new_eval:
        print("  [PR3] --use-new-eval 已是默认行为；当前仍使用 ConstantResolver")

    # Mode 1: Source-code only flowchart (no trace required)
    if args.html_flowchart and args.code_path and not args.trace_file:
        print(f"  正在加载模型源码: {args.code_path}")
        source_files = load_model_code(args.code_path)
        print(f"  加载了 {len(source_files)} 个源文件: {', '.join(sorted(source_files.keys())[:10])}")
        print("  正在生成 HTML 模块流程图 (纯源码静态结构, 双 Tab: 训练/推理)...")
        result = generate_html_flowchart_dual(source_files, timing_data=None, meta=None, output_path=args.html_flowchart)
        if result:
            print(f"  HTML 流程图已保存到: {args.html_flowchart}")
        _emit_ab_summary_if_enabled(args)
        return

    # Mode 2: Full trace analysis (trace_file required)
    if not args.trace_file:
        print("  ❌ 错误: 需要提供 trace 文件路径，或同时指定 --code-path 和 --html-flowchart 生成纯源码流程图")
        sys.exit(1)

    print(f"  正在加载 trace 文件: {args.trace_file}")
    data = load_trace(args.trace_file)
    events = data.get("traceEvents", [])
    print(f"  共 {len(events)} 个事件")

    # Metadata
    meta = extract_metadata(data, events)

    # Detect trace type
    trace_type = detect_trace_type(events)
    if trace_type != "training":
        print(f"  ⚠️ 检测到这是 {trace_type} trace，当前 skill 仅支持训练 trace 分析")
        print(f"  如确认是训练 trace，将继续分析")

    # Detect enhanced trace (Code Location + stack_traces)
    has_code_loc, has_stack_traces = detect_enhanced_trace(events)
    is_enhanced = has_code_loc and has_stack_traces
    if is_enhanced:
        print(f"  检测到增强 trace（含 Code Location 和 stack_traces）")
        if not args.code_path:
            print(f"  ❌ 错误: 增强 trace 需要提供模型源码才能进行源码级分析")
            print(f"  请通过 --code-path 参数提供模型源码目录或 tar.gz 压缩包")
            print(f"  示例: python3 scripts/analyze_trace.py trace.json --code-path code_commit.tar.gz")
            sys.exit(1)

    # Load model source code
    source_files = {}
    if args.code_path:
        print(f"  正在加载模型源码: {args.code_path}")
        source_files = load_model_code(args.code_path)
        print(f"  加载了 {len(source_files)} 个源文件: {', '.join(sorted(source_files.keys())[:10])}")
        if len(source_files) > 10:
            print(f"    ... 等共 {len(source_files)} 个文件")

    # Step duration: collect ALL ProfilerStep events and compute average
    profiler_steps = []
    for e in events:
        if e.get("cat") == "user_annotation" and "ProfilerStep" in e.get("name", ""):
            profiler_steps.append({"ts": e["ts"], "dur": e.get("dur", 0), "end": e["ts"] + e.get("dur", 0)})
    profiler_steps.sort(key=lambda x: x["ts"])
    num_steps = len(profiler_steps)

    if num_steps > 0:
        step_dur_us = sum(s["dur"] for s in profiler_steps) / num_steps
    else:
        step_dur_us = 0
        for e in events:
            if e.get("cat") == "Trace":
                step_dur_us = e.get("dur", 0)
                break
        if step_dur_us == 0:
            ts_vals = [e.get("ts", 0) for e in events if e.get("ph") == "X"]
            end_vals = [e.get("ts", 0) + e.get("dur", 0) for e in events if e.get("ph") == "X"]
            if ts_vals and end_vals:
                step_dur_us = max(end_vals) - min(ts_vals)

    print(f"  ProfilerStep 数量: {num_steps}")
    print(f"  平均 Step 耗时: {format_duration(step_dur_us)}")

    # Build parent relationships
    print("  正在构建事件调用关系...")
    build_parent_index(events)
    thread_indices = add_func_call_parent(events)

    # Classify threads
    print("  正在分析线程结构...")
    thread_info = classify_threads(events, meta)

    # Step decomposition
    step_decomp = analyze_step_decomposition(events, thread_info, profiler_steps)

    # Module analysis
    print("  正在分析 Module 耗时...")
    mod_info = analyze_modules_by_thread(events, thread_info, step_dur_us, profiler_steps)
    print(f"  发现 {len(mod_info['depth_map'])} 种 Module 类型")

    # GPU timeline analysis
    print("  正在分析 GPU 时间线...")
    gpu_info = analyze_gpu_timeline(events, thread_info, step_dur_us, profiler_steps)

    # Device/Host analysis
    print("  正在分析 Device/Host 耗时...")
    dh_info = analyze_device_host(events, step_dur_us, thread_info, profiler_steps)

    # Worker thread analysis
    workers = analyze_worker_threads(events, thread_info, meta)

    # Source code hotspot analysis
    src_info = None
    if source_files and has_stack_traces:
        print("  正在分析源码热点...")
        src_info = analyze_source_hotspots(events, source_files)
        print(f"  映射到 {len(src_info['sorted_classes'])} 个 Class, {len(src_info['sorted_methods'])} 个 Method")
        # Enrich top kernel HostModule mapping with source-code nn.Module class names
        if gpu_info:
            print("  正在将 TopKernel HostModule 映射到源码 nn.Module...")
            enrich_kernel_modules_with_source(events, gpu_info, src_info)

    # Per-thread timeline analysis
    print("  正在分析多线程时间线...")
    timelines = analyze_per_thread_timeline(events, thread_info, meta)
    print(f"  分析了 {len(timelines)} 个线程时间线")

    # Kernel call stack analysis
    kernel_stacks = {}
    if gpu_info and is_enhanced:
        print("  正在提取 Kernel 调用栈...")
        kernel_stacks = analyze_kernel_call_stacks(events, gpu_info)
        print(f"  提取了 {len(kernel_stacks)} 个 Kernel 的调用栈")

    # Communication / Compute overlap analysis
    print("  正在分析通信/计算 Overlap...")
    overlap_info = analyze_comm_compute_overlap(events, thread_info, profiler_steps)
    if overlap_info:
        print(f"  通信流: {overlap_info['num_comm_streams']} 条, Memcpy 流: {overlap_info['num_memcpy_streams']} 条")
        if overlap_info["comm_total"] > 0:
            print(f"  通信 overlap: {overlap_info['comm_overlap_pct']:.1f}%, 暴露开销: {format_duration(overlap_info['comm_exposed'])}")

    # lagrange_torch references
    lagrange_refs = extract_lagrange_refs(events, gpu_info, kernel_stacks)
    if lagrange_refs:
        print(f"  发现 {len(lagrange_refs)} 个 lagrange_torch 相关 Kernel")

    # Screenshot
    screenshot_path = None
    if args.screenshot:
        print("  正在生成 Trace 可视化截图...")
        output_dir = os.path.dirname(args.output) if args.output else "."
        if not output_dir:
            output_dir = "."
        screenshot_path, err = generate_trace_screenshot(args.trace_file, output_dir)
        if screenshot_path:
            print(f"  截图已保存到: {screenshot_path}")
        else:
            print(f"  ⚠️ 截图生成失败: {err}")

    print()

    # === Print Reports ===
    print_trace_overview(meta, trace_type, step_dur_us, thread_info, step_decomp, num_steps)
    print_thread_overview(thread_info, meta, workers)

    # Per-thread timeline
    print_per_thread_timeline(timelines)

    if not args.no_tree:
        print_header("Module 层级结构")
        print_module_tree(mod_info, "ROOT", 0, step_dur_us, mod_info.get("module_thread_info"))
        print()

    print_module_report(mod_info, step_dur_us, args.max_level)
    print_gpu_timeline_report(gpu_info, step_dur_us)

    # Kernel call stacks
    if kernel_stacks:
        print_kernel_call_stacks(kernel_stacks, gpu_info)

    print_device_host_report(dh_info)

    # Comm/compute overlap
    if overlap_info:
        print_comm_compute_overlap(overlap_info, gpu_info)

    if src_info:
        print_source_hotspot_report(src_info, source_files)

    # lagrange_torch references
    if lagrange_refs:
        print_lagrange_refs(lagrange_refs)

    # Save markdown
    if args.output:
        save_markdown_report(meta, trace_type, step_dur_us, step_decomp, thread_info, workers,
                             mod_info, gpu_info, dh_info, args.output, args.max_level,
                             src_info=src_info, source_files=source_files,
                             timelines=timelines, kernel_stacks=kernel_stacks,
                             lagrange_refs=lagrange_refs, screenshot_path=screenshot_path,
                             num_steps=num_steps, overlap_info=overlap_info)

    # Save JSON
    if args.json_output:
        result = {
            "trace_type": trace_type,
            "device": meta.get("device_name", "N/A"),
            "step_duration_us": step_dur_us,
            "step_decomposition": step_decomp,
            "modules": {
                name.replace("nn.Module: ", ""): {
                    "inclusive_us": mod_info["module_durations"].get(name, 0),
                    "exclusive_us": mod_info["module_exclusive"].get(name, 0),
                    "inclusive_pct": mod_info["module_durations"].get(name, 0) / step_dur_us * 100 if step_dur_us > 0 else 0,
                    "exclusive_pct": mod_info["module_exclusive"].get(name, 0) / step_dur_us * 100 if step_dur_us > 0 else 0,
                    "calls": mod_info["module_call_counts"].get(name, 0),
                    "depth": mod_info["depth_map"].get(name, 0),
                    "parent": mod_info["module_parent_map"].get(name, "ROOT").replace("nn.Module: ", ""),
                    "thread": list(mod_info["module_thread_info"].get(name, {}).get("roles", set())),
                }
                for name in mod_info["module_durations"]
            },
            "gpu_timeline": {
                "gpu_utilization": gpu_info["gpu_utilization"] if gpu_info else 0,
                "total_kernel_dur_us": gpu_info["total_kernel_dur"] if gpu_info else 0,
                "total_gap_us": gpu_info["total_gap"] if gpu_info else 0,
                "num_kernels": gpu_info["num_kernels"] if gpu_info else 0,
                "top_kernels": [
                    {
                        "name": kname,
                        "dur_us": info["dur"],
                        "count": info["count"],
                        "host_module": (gpu_info["top_kernel_modules"].get(kname, []) or [("N/A",)])[0][0],
                    }
                    for kname, info in (gpu_info["top_kernels"] if gpu_info else [])
                ],
            } if gpu_info else None,
            "device_host": {
                "host_total_us": dh_info["host_total_us"],
                "device_total_us": dh_info["device_total_us"],
                "host_pct": dh_info["host_total_us"] / step_dur_us * 100 if step_dur_us > 0 else 0,
                "device_pct": dh_info["device_total_us"] / step_dur_us * 100 if step_dur_us > 0 else 0,
                "device_breakdown": {
                    cat: {"dur_us": dur, "pct": dur / step_dur_us * 100 if step_dur_us > 0 else 0}
                    for cat, dur in dh_info["device_breakdown"].items()
                },
            },
            "comm_compute_overlap": {
                "comm_total_us": overlap_info["comm_total"],
                "comm_overlap_us": overlap_info["comm_overlap"],
                "comm_exposed_us": overlap_info["comm_exposed"],
                "comm_overlap_pct": overlap_info["comm_overlap_pct"],
                "memcpy_total_us": overlap_info["memcpy_total"],
                "memcpy_overlap_us": overlap_info["memcpy_overlap"],
                "memcpy_exposed_us": overlap_info["memcpy_exposed"],
                "memcpy_overlap_pct": overlap_info["memcpy_overlap_pct"],
                "compute_total_us": overlap_info["compute_total"],
            } if overlap_info else None,
        }
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  JSON 结果已保存到: {args.json_output}")

    # HTML Flowchart (source-code first, timing overlay optional)
    if args.html_flowchart:
        if not source_files:
            print("  ⚠️ 跳过 HTML 流程图生成: 需要提供源码 (--code-path)")
            print("  流程图基于源码静态结构生成，无源码时无法确定模块层级关系")
        else:
            # Extract roots for timing rollup (Step 3 orphan aggregation)
            _, roots, _ = build_static_module_tree(source_files, conditional_mode="train")
            # Build timing data from trace to overlay on static structure
            timing_data = build_timing_data_from_trace(events, source_files, mod_info, step_dur_us, profiler_steps, src_info=src_info, roots=roots)
            print("  正在生成 HTML 模块流程图 (源码结构 + 运行时层级 + 时间填充, 双 Tab: 训练/推理)...")
            result = generate_html_flowchart_dual(source_files, timing_data=timing_data, meta=meta, output_path=args.html_flowchart, trace_events=events)
            if result:
                print(f"  HTML 流程图已保存到: {args.html_flowchart}")

    # PR2: print A/B diff summary at the very end so the user can see it
    # alongside the regular analysis output.
    _emit_ab_summary_if_enabled(args)


def _emit_ab_summary_if_enabled(args):
    """PR3: optionally persist the current evaluator status for debugging."""
    report = getattr(args, "ab_eval_report", None)
    if not report:
        return
    payload = {
        "mode": "constant_resolver_only",
        "use_new_eval": True,
        "total": len(_AB_EVAL_DIFFS),
        "matches": 0,
        "mismatches": 0,
        "all": _AB_EVAL_DIFFS,
    }
    try:
        with open(report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [PR3] evaluator status saved to: {report}")
    except Exception as e:
        print(f"  ⚠️ failed to write evaluator status report: {e}")


def _ast_frontend_smoke_test():
    target = "testset/extracted/5698781/modelcode/cvr.py"
    fe = ASTFrontend(path=target)
    nn_classes = [c for c in fe.class_registry if fe.is_nn_module(c)]
    print(f"[ASTFrontend smoke test] file={target}")
    print(f"nn.Module classes: {len(nn_classes)}")
    for class_name in nn_classes:
        assigns = fe.get_init_assignments_ast(class_name)
        print(f"  {class_name}: {len(assigns)} init assignments")


if __name__ == "__main__":
    main()
