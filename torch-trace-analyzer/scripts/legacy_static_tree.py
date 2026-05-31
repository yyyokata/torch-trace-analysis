#!/usr/bin/env python3

import ast
import logging
import sys
from collections import defaultdict

import torch.nn as _torch_nn

from ast_constants import ConstantTable
from ast_resolver import ConstantResolver
from ast_types import Scope
from common_utils import _join_logical_lines, _strip_inline_comment
from legacy_ast_module_tree import _AST_ModuleTreeExtractor
from legacy_dataflow_edges import _build_data_dependency_edges
from legacy_dataflow_support import (
    _build_source_dependency_order,
    _expand_fstring_with_loop_vars,
    _resolve_str_list_iterative,
)
from source_index import _build_ast_frontends, _build_class_map

LOG = logging.getLogger(__name__)

_NATIVE_CONTAINER_NAMES: frozenset = frozenset({
    'ModuleDict', 'ModuleList', 'Sequential',
    'ParameterDict', 'ParameterList',
})

def _is_nn_leaf_stub(class_name: str) -> bool:
    if class_name in _NATIVE_CONTAINER_NAMES:
        return False
    cls = getattr(_torch_nn, class_name, None)
    if cls is None:
        return False
    try:
        return issubclass(cls, _torch_nn.Module)
    except TypeError:
        return False

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
    _ast_extractor = _AST_ModuleTreeExtractor(source_files, ast_frontends=ast_frontends)
    file_int_consts = {}
    file_str_list_globals_raw = {}  # {fname: {VARNAME: raw_inner_string}}
    def _is_upper_snake_name(_s: str) -> bool:
        return bool(_s) and _s[0:1].isupper() and all(
            c.isupper() or c.isdigit() or c == "_" for c in _s
        )
    nn_import_aliases = {}  # {alias_name: short_name}, e.g. {"TorchLinear": "Linear"}
    for _fname, _lines in source_files.items():
        int_consts = {}
        str_list_consts_raw = {}
        _fe = _get_ast_frontend(_fname)
        _file_tree = _fe.tree if _fe is not None else None
        if _file_tree is not None:
            for _node in _file_tree.body:
                if isinstance(_node, ast.ImportFrom) and _node.module == "torch.nn":
                    for _alias in _node.names:
                        if _alias.name == "*":
                            continue
                        _alias_name = _alias.asname or _alias.name
                        nn_import_aliases[_alias_name] = _alias.name
                    continue
                if not isinstance(_node, ast.Assign) or len(_node.targets) != 1:
                    continue
                _t = _node.targets[0]
                if not (isinstance(_t, ast.Name) and _is_upper_snake_name(_t.id)):
                    continue
                _v = _node.value
                if isinstance(_v, ast.Constant) and isinstance(_v.value, int) and not isinstance(_v.value, bool):
                    if _v.value >= 0:
                        int_consts[_t.id] = _v.value
                elif isinstance(_v, ast.List):
                    try:
                        _inner_text = ", ".join(ast.unparse(_e) for _e in _v.elts)
                    except Exception:
                        _inner_text = ""
                    str_list_consts_raw[_t.id] = _inner_text
        file_int_consts[_fname] = int_consts
        file_str_list_globals_raw[_fname] = str_list_consts_raw
    def _extract_nn_short(cls_ref_full, nn_import_aliases=None):
        """Normalize a constructor ref into _is_nn_leaf_stub's short name.
        Supports nn.Linear, torch.nn.Linear, and from-torch.nn import aliases.
        Returns (short_name, canonical_ref) or (None, None).
        """
        cls_ref_full = (cls_ref_full or "").strip()
        if cls_ref_full.startswith("nn."):
            short = cls_ref_full.split(".", 1)[1]
            return short, cls_ref_full
        if cls_ref_full.startswith("torch.nn."):
            short = cls_ref_full.split(".", 2)[2]
            return short, "nn." + short
        if nn_import_aliases and cls_ref_full in nn_import_aliases:
            short = nn_import_aliases[cls_ref_full]
            return short, "nn." + short
        return None, None
    global_int_const_values = defaultdict(set)
    for _fname, _cs in file_int_consts.items():
        for _k, _v in _cs.items():
            global_int_const_values[_k].add(_v)
    file_str_list_globals = {}  # {fname: {VARNAME: [str, ...]}}
    ctor_kw_int_args = defaultdict(lambda: defaultdict(set))
    ctor_kw_list_lens = defaultdict(lambda: defaultdict(set))
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
    for (fname, cname), info in class_map.items():
        if "forward" in info["methods"]:
            nn_module_classes.add(cname)
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
    class_attrs = {}  # {class_name: {attr_name: class_ref}}
    torch_native_module_classes = set()  # {"nn.LayerNorm", "nn.Linear", ...}
    attr_def_loc = {}
    input_source_attrs = defaultdict(set)  # {class_name: {attr_name, ...}}
    class_str_list_attrs = defaultdict(dict)
    dynamic_attrs_per_class = defaultdict(set)  # {class_name: {attr_name, ...}}
    class_conditional_attrs = defaultdict(dict)
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
        text = text.rstrip(':').strip()
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
        flipped = False
        for _sub in ast.walk(_cond_node):
            if isinstance(_sub, ast.UnaryOp) and isinstance(_sub.op, ast.Not):
                _tail = _leaf_token(_sub.operand)
                if _tail and (_tail in train_words or _tail in serve_words):
                    flipped = True
                    break
        if has_train and not has_serve:
            base = "train"
        elif has_serve and not has_train:
            base = "infer"
        else:
            return None
        if flipped:
            return "infer" if base == "train" else "train"
        return base
    container_elems = defaultdict(lambda: defaultdict(list))
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
    for _fname, _slots in file_str_list_globals_raw.items():
        parsed = {}
        for _name, _raw in _slots.items():
            _items = _parse_str_literal_list(_raw)
            if _items:
                parsed[_name] = _items
        file_str_list_globals[_fname] = parsed
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
        for mname, (ms, me) in info["methods"].items():
            _ast_method_range = _fe.get_method_lines(cname, mname) if _fe else None
            if _ast_method_range:
                ms, me = _ast_method_range
            method_lines = lines[ms - 1: min(me, len(lines))]
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
            _loop_var_to_str_items = {}  # {loop_var_name: [str_item, ...]}
            _loop_indent_stack = []  # [(indent, loop_var)] tracking active loops
            _cond_branch_stack = []  # [(if_indent, branch_polarity)]
            _local_var_ctor: dict = {}
            _local_var_str_list: dict = {}
            def _self_attr_from_target(_target):
                return _fe._extract_self_attr_name(_target) if _fe else None
            def _node_to_text(_node):
                return _fe._local_node_to_text(_node) if _fe else ""
            for phys_lineno, line in logical_lines:
                _raw_line_for_indent = lines[phys_lineno - 1] if 0 < phys_lineno <= len(lines) else ""
                _cur_indent = len(_raw_line_for_indent) - len(_raw_line_for_indent.lstrip())
                if line.lstrip().startswith('#'):
                    continue
                _local_stmt = _fe.parse_local_stmt(line) if _fe else None
                while _loop_indent_stack and _cur_indent <= _loop_indent_stack[-1][0]:
                    _, _popped_var = _loop_indent_stack.pop()
                    _loop_var_to_str_items.pop(_popped_var, None)
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
                        _cond_branch_stack[-1] = (_if_ind,
                                                  "infer" if _polarity == "train" else "train")
                        break
                    elif _cur_indent == _if_ind:
                        _cond_branch_stack.pop()
                    else:
                        break
                if _is_if_line and _hdr_node is not None:
                    try:
                        _cond_text = ast.unparse(_hdr_node.test)
                    except Exception:
                        _cond_text = ""
                    _pol = _cond_branch_polarity(_cond_text)
                    if _pol is not None:
                        _cond_branch_stack.append((_cur_indent, _pol))
                _active_branch = _cond_branch_stack[-1][1] if _cond_branch_stack else None
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
                    _name_var = None
                    if isinstance(_tgt, ast.Name):
                        _name_var = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and len(_tgt.elts) >= 2 \
                            and all(isinstance(e, ast.Name) for e in _tgt.elts):
                        _name_var = _tgt.elts[-1].id
                    _enum_inner = None
                    if (isinstance(_iter, ast.Call)
                            and isinstance(_iter.func, ast.Name)
                            and _iter.func.id == 'enumerate'
                            and len(_iter.args) >= 1):
                        _enum_inner = _iter.args[0]
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
                if _ast_ctor_assign or _ast_init_item:
                    if _ast_ctor_assign:
                        attr_name = _ast_ctor_assign.get("attr")
                        cls_ref_full = _ast_ctor_assign.get("class_full") or ""
                    else:
                        attr_name = _ast_init_item.get("attr")
                        cls_ref_full = _ast_init_item.get("class") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    native_cls_ref = None
                    _nn_short, _nn_canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                    if _nn_short and _is_nn_leaf_stub(_nn_short):
                        native_cls_ref = _nn_canonical
                    if cls_ref in nn_module_classes:
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
                        _lg_match = _fe.match_self_attr_call_assign(_local_stmt, func_prefixes={"LG."}) if _fe else None
                        if _lg_match:
                            attr_name, _lg_call = _lg_match
                            attrs[attr_name] = "__LG_InputSource"
                            attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                            input_source_attrs[cname].add(attr_name)
                _ast_fc_gv = _fe.parse_local_fc_get_vector_assign(_local_stmt) if _fe else None
                if _ast_fc_gv:
                    attr_name = _ast_fc_gv.get("attr")
                    if attr_name not in attrs:
                        attrs[attr_name] = "__LG_InputSource"
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        input_source_attrs[cname].add(attr_name)
                _container_match = _fe.match_self_attr_call_assign(
                    _local_stmt,
                    func_names={"ModuleList", "ModuleDict", "Sequential"},
                ) if _fe else None
                if _container_match:
                    cont, _call = _container_match
                    _kind = (_fe._expr_leaf_name(_call.func) or "").split('.')[-1]
                    _arg0 = _call.args[0] if _call and _call.args else None
                    if _kind == "ModuleList":
                        _record_container_kind(cname, cont, "ModuleList")
                        if isinstance(_arg0, ast.ListComp) and isinstance(_arg0.elt, ast.Call):
                            _generators = _arg0.generators or []
                            _gen0 = _generators[0] if len(_generators) == 1 else None
                            _cls_ref = _node_to_text(_arg0.elt.func).split('.')[-1]
                            _n = None
                            if _gen0 and isinstance(_gen0.iter, ast.Call) and (_fe._expr_leaf_name(_gen0.iter.func) == "range") and len(_gen0.iter.args) == 1:
                                _n = _eval_int_node(_gen0.iter.args[0], fname, cname, mname, parent_cls=preferred_root, parent_attr='stack')
                            if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.List):
                                _n = len(_gen0.iter.elts)
                            if _n is None and _gen0 is not None and isinstance(_gen0.iter, ast.Name):
                                _len = _eval_list_len_node(_gen0.iter, fname, cname, mname)
                                if _len is not None:
                                    _n = _len
                            if _n is not None:
                                _short, _canonical = _extract_nn_short(_node_to_text(_arg0.elt.func), nn_import_aliases)
                                if _short and _is_nn_leaf_stub(_short):
                                    attrs.setdefault(cont, _canonical)
                                    attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                    torch_native_module_classes.add(_canonical)
                                    for i in range(_n):
                                        _ensure_elem(cont, cname, _elem_attr_list(cont, i), _canonical, fname, phys_lineno, attrs)
                                elif _cls_ref in nn_module_classes:
                                    attrs.setdefault(cont, _cls_ref)
                                    attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                    for i in range(_n):
                                        _ensure_elem(cont, cname, _elem_attr_list(cont, i), _cls_ref, fname, phys_lineno, attrs)
                        elif isinstance(_arg0, ast.List):
                            for i, _elt in enumerate(_arg0.elts):
                                if isinstance(_elt, ast.Call):
                                    cls_ref_full = _node_to_text(_elt.func)
                                    cls_ref = cls_ref_full.split('.')[-1]
                                    _short, _canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                                    if _short and _is_nn_leaf_stub(_short):
                                        attrs.setdefault(cont, _canonical)
                                        attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                        torch_native_module_classes.add(_canonical)
                                        _ensure_elem(cont, cname, _elem_attr_list(cont, i), _canonical, fname, phys_lineno, attrs)
                                    elif cls_ref in nn_module_classes:
                                        attrs.setdefault(cont, cls_ref)
                                        attr_def_loc.setdefault((cname, cont), (fname, phys_lineno))
                                        _ensure_elem(cont, cname, _elem_attr_list(cont, i), cls_ref, fname, phys_lineno, attrs)
                    if _kind == "ModuleDict":
                        _record_container_kind(cname, cont, "ModuleDict")
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
                _ast_append_ctor = _fe.parse_local_append_ctor(_local_stmt) if _fe else None
                if _ast_append_ctor:
                    attr_name = _ast_append_ctor.get("attr")
                    cls_ref_full = _ast_append_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    _short, _canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                    if _short and _is_nn_leaf_stub(_short):
                        attrs[attr_name] = _canonical
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        torch_native_module_classes.add(_canonical)
                        idx = len(container_elems[cname][attr_name])
                        _ensure_elem(attr_name, cname, _elem_attr_list(attr_name, idx), _canonical, fname, phys_lineno, attrs)
                        _record_container_kind(cname, attr_name, "list")
                    elif cls_ref in nn_module_classes:
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        idx = len(container_elems[cname][attr_name])
                        _ensure_elem(attr_name, cname, _elem_attr_list(attr_name, idx), cls_ref, fname, phys_lineno, attrs)
                        _record_container_kind(cname, attr_name, "list")
                _ast_subscript_ctor = _fe.parse_local_subscript_ctor(_local_stmt) if _fe else None
                if _ast_subscript_ctor:
                    attr_name = _ast_subscript_ctor.get("attr")
                    key_expr = _ast_subscript_ctor.get("key_expr") or ""
                    cls_ref_full = _ast_subscript_ctor.get("class_full") or ""
                    cls_ref = cls_ref_full.split('.')[-1]
                    if cls_ref in nn_module_classes:
                        attrs[attr_name] = cls_ref
                        attr_def_loc.setdefault((cname, attr_name), (fname, phys_lineno))
                        _record_container_kind(cname, attr_name, "dict")
                        k = _normalize_str_key(key_expr)
                        if k is not None:
                            _ensure_elem(attr_name, cname, _elem_attr_dict(attr_name, k), cls_ref, fname, phys_lineno, attrs)
                        else:
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
                                _ensure_elem(attr_name, cname, f"{attr_name}[*]", cls_ref, fname, phys_lineno, attrs)
                _ast_local_ctor = _fe.parse_local_var_ctor(_local_stmt) if _fe else None
                if _ast_local_ctor:
                    _lv_name = _ast_local_ctor.get("name")
                    _lv_cls = (_ast_local_ctor.get("class_full") or "").split('.')[-1]
                    if (_lv_name not in ("self", "super")
                            and _lv_cls in nn_module_classes):
                        _local_var_ctor[_lv_name] = (_lv_cls, fname, phys_lineno)
                _ast_setattr_ctor = _fe.parse_local_setattr_ctor(_local_stmt) if _fe else None
                if _ast_setattr_ctor:
                    name_expr = (_ast_setattr_ctor.get("name_expr") or "").strip()
                    if _ast_setattr_ctor.get("two_step"):
                        _bv = _ast_setattr_ctor.get("local_var")
                        if _bv in _local_var_ctor:
                            cls_ref_full = _local_var_ctor[_bv][0]
                        else:
                            cls_ref_full = ""
                    else:
                        cls_ref_full = _ast_setattr_ctor.get("class_full") or ""
                    if cls_ref_full:
                        cls_ref = cls_ref_full.split('.')[-1]
                        _native_short, _native_canonical = _extract_nn_short(cls_ref_full, nn_import_aliases)
                        if _native_short and _is_nn_leaf_stub(_native_short):
                            real_attr = None
                            try:
                                _name_node = ast.parse(name_expr, mode='eval').body
                            except SyntaxError:
                                _name_node = None
                            if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                real_attr = _name_node.value
                            elif isinstance(_name_node, ast.JoinedStr):
                                if all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                       for v in _name_node.values):
                                    real_attr = ''.join(v.value for v in _name_node.values)
                            if real_attr:
                                attrs[real_attr] = _native_canonical
                                attr_def_loc.setdefault((cname, real_attr), (fname, phys_lineno))
                                torch_native_module_classes.add(_native_canonical)
                        elif cls_ref in nn_module_classes:
                            real_attr = None
                            _fstr_handled = False
                            try:
                                _name_node = ast.parse(name_expr, mode='eval').body
                            except SyntaxError:
                                _name_node = None
                            if isinstance(_name_node, ast.Constant) and isinstance(_name_node.value, str):
                                real_attr = _name_node.value
                            elif isinstance(_name_node, ast.JoinedStr):
                                if all(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                       for v in _name_node.values):
                                    real_attr = ''.join(v.value for v in _name_node.values)
                                else:
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
                                for _real in _loop_var_to_str_items[name_expr]:
                                    attrs[_real] = cls_ref
                                    attr_def_loc.setdefault((cname, _real), (fname, phys_lineno))
                            elif not _fstr_handled:
                                synth_attr = cls_ref
                                if synth_attr not in attrs and not any(v == cls_ref for v in attrs.values()):
                                    attrs[synth_attr] = cls_ref
                                    attr_def_loc.setdefault((cname, synth_attr), (fname, phys_lineno))
                                    dynamic_attrs_per_class[cname].add(synth_attr)
        class_attrs[cname] = attrs
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
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'range'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('range', _indent, _expr)
        if (isinstance(_it, ast.Call)
                and isinstance(_it.func, ast.Name)
                and _it.func.id == 'enumerate'
                and len(_it.args) >= 1):
            try:
                _expr = ast.unparse(_it.args[0])
            except Exception:
                _expr = ""
            return ('enumerate', _indent, _expr)
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
        self_to_param = _fe.get_self_param_aliases(cname, "__init__") if _fe else {}
        dyn_indexed_containers = _fe.get_dynamic_indexed_self_attrs(cname) if _fe else set()
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
                    if not _joinedstr_references_var(template_jstr, loop_var):
                        continue
                    if n is not None:
                        for idx in range(n):
                            real = _expand_fstring_template_ast(template_jstr, loop_var, idx)
                            if real is None:
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
                        synth = _wildcard_template_from_jstr(template_jstr) or cls_ref
                        if synth not in attrs:
                            attrs[synth] = cls_ref
                            attr_def_loc.setdefault((cname, synth), (fname, body_lineno))
                            dynamic_attrs_per_class[cname].add(synth)
                i = j
        class_attrs[cname] = attrs
    dep_edges, dep_edge_locs, split_info = _build_data_dependency_edges(
        source_files,
        class_map,
        class_attrs,
        class_str_list_attrs,
        dynamic_attrs_per_class,
        ast_frontends=ast_frontends,
    )
    for cname, smap in split_info.items():
        if cname not in class_attrs:
            continue
        for orig_attr, split_names in smap.items():
            cls_ref = class_attrs[cname].get(orig_attr)
            if cls_ref is None:
                continue
            del class_attrs[cname][orig_attr]
            for sn in split_names:
                class_attrs[cname][sn] = cls_ref
    tree = {}
    all_child_classes = set()
    all_tree_classes = set(nn_module_classes) | set(torch_native_module_classes)
    for cname in all_tree_classes:
        if cname not in class_attrs:
            class_attrs[cname] = {}
        attrs = class_attrs[cname]
        call_order = module_call_order.get(cname, [])
        children_ordered = []
        seen = set()
        for attr in call_order:
            if attr in attrs and attrs[attr] not in seen:
                children_ordered.append(attrs[attr])
                seen.add(attrs[attr])
                all_child_classes.add(attrs[attr])
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
                       "dynamic_setattr_attrs": sorted(dynamic_attrs_per_class.get(cname, set())),
                       "container_kw": dict(class_container_kw.get(cname, {}))}
    roots = [c for c in nn_module_classes if c not in all_child_classes and c in tree]
    if not roots:
        roots = list(tree.keys())[:1]
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
    if "RootModule" in roots:
        roots = ["RootModule"] + [r for r in roots if r != "RootModule"]
    if preferred_root and preferred_root in roots:
        roots = [preferred_root] + [r for r in roots if r != preferred_root]
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
        tree = {k: v for k, v in tree.items() if k in reachable}
        roots = [r for r in roots if r in reachable]
        if not roots and reachable:
            roots = sorted(reachable, key=lambda r: -count_descendants(r))[:1]
    for cname in tree:
        per_class = {}
        for attr in tree[cname].get("attrs", {}):
            loc = attr_def_loc.get((cname, attr))
            if loc:
                per_class[attr] = loc
        tree[cname]["attr_def_loc"] = per_class
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
    for cname in tree:
        tree[cname]["input_source_attrs"] = input_source_attrs.get(cname, set())
        for (fname, cm_name), _cm_info in class_map.items():
            if cm_name != cname:
                continue
            _fe = _get_ast_frontend(fname)
            if _fe:
                from attr_scanner import AttrScanner as _AttrScanner
                _scanner = _AttrScanner()
                result_attrs_list = _scanner._scan_result_attrs(cname, _fe)
                result_attrs = {ra.attr_name: ra for ra in result_attrs_list}
                if result_attrs:
                    tree[cname]["result_attrs"] = result_attrs
            break
        tree[cname]["static_diagnostics"] = list(_new_eval_diagnostics)
    for cname in tree:
        per_class = {}
        for (kc, ka), kind in container_kinds.items():
            if kc == cname:
                per_class[ka] = kind
        tree[cname]["container_kinds"] = per_class
    for cname in tree:
        ca = class_conditional_attrs.get(cname, {})
        meaningful = {a: info for a, info in ca.items()
                      if info.get("train_class") and info.get("infer_class")
                      and info.get("train_class") != info.get("infer_class")}
        tree[cname]["conditional_attrs"] = meaningful
    for cname in tree:
        tree[cname]["unresolved_moduledict"] = [u for u in unresolved_moduledict
                                                 if u.get("class") == cname]
    for cname in tree:
        tree[cname]["instance_kw_list_lens"] = {
            attr: dict(kws) for (pcn, attr), kws in instance_kw_list_lens.items()
            if pcn == cname
        }
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
