#!/usr/bin/env python3

import ast
import re
import textwrap
from collections import defaultdict

from common_utils import _join_logical_lines, _strip_inline_comment
from source_index import _build_ast_frontends

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
    def _extract_all_names(node: ast.AST) -> list[str]:
        """从 AST node walk 出所有 Name.id（去重，保序，过滤 self/cls）"""
        seen, result = set(), []
        if node is None:
            return result
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in seen and n.id not in ("self", "cls"):
                seen.add(n.id)
                result.append(n.id)
        return result
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
        dps, dev = _collect_dict_reads(expr, dict_slots, fname)
        if dps:
            producers.update(dps)
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
            var_producers = {}
            tensor_alias_producers = {}
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
                    _carriers = []
                    if isinstance(target_var, str) and target_var.isidentifier():
                        _carriers = [target_var]
                    base.append({"var": target_var, "file": fname,
                                 "line": lineno, "text": step_text,
                                 "carriers": _carriers})
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
                _candidates = set(var_lineage.keys()) | set(var_producers.keys())
                if not _candidates:
                    return []
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
            dict_slots = defaultdict(dict)
            edges = set()
            edge_locs = {}
            _loop_var_to_real_attrs = {}
            _loop_var_to_str_items_local = {}
            _str_list_attrs_for_class = (class_str_list_attrs.get(cname) or {})
            known_attrs = set(attrs.keys())
            dynamic_setattr_attrs = sorted(
                [a for a in (dynamic_attrs_per_class.get(cname) or set()) if a in known_attrs]
            )
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
                if container in container_to_elems and container_to_elems[container]:
                    return list(container_to_elems[container])
                return [container]
            _all_methods = info["methods"]
            _helper_method_ranges = []
            _fwd_text = "\n".join(lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))])
            _method_text_cache = {"forward": _fwd_text}
            for _mname, _mrange in _all_methods.items():
                if _mname in ("forward", "__init__"):
                    continue
                _method_text_cache[_mname] = "\n".join(
                    lines[_mrange[0] - 1: min(_mrange[1], len(lines))])
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
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if (
                                isinstance(f, ast.Attribute)
                                and isinstance(f.value, ast.Name)
                                and f.value.id == "self"
                                and f.attr in attrs
                            ):
                                return True
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
                        if _method_has_attr_call(_method_text_cache.get(_mname, "")):
                            _reachable_helpers.add(_mname)
                _frontier = _next
            for _mname in _seen_methods - {"forward"}:
                _reachable_helpers.add(_mname)
            for _mname in sorted(_reachable_helpers):
                _mrange = _all_methods.get(_mname)
                if _mrange is not None:
                    _helper_method_ranges.append(_mrange)
            fwd_lines = lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))]
            logical_lines = _join_logical_lines(fwd_lines, fwd_range[0])
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
            _method_ranges_for_counting = [(fwd_range[0], fwd_range[1])]
            for _hr in _helper_method_ranges:
                _method_ranges_for_counting.append((_hr[0], _hr[1]))
            _attr_max_calls_in_single_method = defaultdict(int)
            for _mr_start, _mr_end in _method_ranges_for_counting:
                _method_lines = lines[_mr_start - 1: min(_mr_end, len(lines))]
                _method_text = "\n".join(_method_lines)
                _method_call_count = defaultdict(int)
                _body = _parse_method_ast(_method_text)
                if not _body:
                    for _ca, _cnt in _method_call_count.items():
                        if _cnt > _attr_max_calls_in_single_method[_ca]:
                            _attr_max_calls_in_single_method[_ca] = _cnt
                    continue
                def _is_self_attr(node, attr_name=None):
                    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"):
                        return False
                    return attr_name is None or node.attr == attr_name
                for top in _body:
                    for sub in ast.walk(top):
                        if isinstance(sub, ast.Call):
                            _func = sub.func
                            if _is_self_attr(_func) and _func.attr in known_attrs:
                                _method_call_count[_func.attr] += 1
                                continue
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
                for _ca, _cnt in _method_call_count.items():
                    if _cnt > _attr_max_calls_in_single_method[_ca]:
                        _attr_max_calls_in_single_method[_ca] = _cnt
            _multi_call_candidates = {a for a, cnt in _attr_max_calls_in_single_method.items() if cnt > 1}
            split_node_map = {}
            for phys_lineno, line in logical_lines:
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
                    return None
                _line_mod = None
                try:
                    _line_mod = ast.parse(line)
                except SyntaxError:
                    _line_mod = None
                _line_stmt = (_line_mod.body[0]
                              if _line_mod and len(_line_mod.body) == 1
                              else None)
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
                _ast_dict_write = _ast_stmt if _ast_stmt and _ast_stmt.get("kind") == "dict_write" else None
                _ast_subscript_assign = None
                if _ast_dict_write:
                    dname = (_ast_dict_write.get("dict_var") or "").strip()
                    slot_key = _ast_dict_write.get("dict_key")
                    slot_key = slot_key if slot_key is not None else '*'
                    rhs = (_ast_dict_write.get("rhs_text") or "").strip()
                    _v_rhs_node = _ast_dict_write.get("rhs_node")
                else:
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
                loop_var = None
                loop_attr = None
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
                    _loop_var_candidate = None
                    if isinstance(_tgt, ast.Name):
                        _loop_var_candidate = _tgt.id
                    elif isinstance(_tgt, ast.Tuple) and _tgt.elts:
                        _last = _tgt.elts[-1]
                        if isinstance(_last, ast.Name):
                            _loop_var_candidate = _last.id
                    if _loop_var_candidate is not None:
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
                    if loop_attr in container_to_elems and container_to_elems[loop_attr]:
                        elems = container_to_elems[loop_attr]
                        var_producers[loop_var] = (set(elems), phys_lineno)
                        _record_lineage(loop_var, [], phys_lineno)
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
                if loop_var and loop_attr and loop_attr in _str_list_attrs_for_class:
                    real_items = _str_list_attrs_for_class[loop_attr]
                    _loop_var_to_str_items_local[loop_var] = list(real_items)
                    real_module_attrs = [a for a in real_items if a in known_attrs]
                    if real_module_attrs:
                        _loop_var_to_real_attrs[loop_var] = real_module_attrs
                    continue
                m_setattr_tensor = None
                _setattr_call = None
                _setattr_name_node = None
                _setattr_src_var = None
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
                    if src_var in var_producers:
                        src_producers, _src_pl = var_producers[src_var]
                        alias_name_to_producers = {}
                        if (
                            isinstance(_setattr_name_node, ast.Constant)
                            and isinstance(_setattr_name_node.value, str)
                        ):
                            alias_name_to_producers[_setattr_name_node.value] = set(src_producers)
                        elif isinstance(_setattr_name_node, ast.JoinedStr):
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
                                items_set = set(items)
                                can_refine = (
                                    all(it in known_attrs for it in items)
                                    and items_set & set(src_producers) == items_set & set(known_attrs)
                                )
                                for it in items:
                                    _expanded_parts = []
                                    for _part in _setattr_name_node.values:
                                        if isinstance(_part, ast.Constant) and isinstance(_part.value, str):
                                            _expanded_parts.append(_part.value)
                                        elif isinstance(_part, ast.FormattedValue):
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
                            if isinstance(_setattr_name_node, ast.Name):
                                _name_arg = _setattr_name_node.id
                                if _name_arg in _loop_var_to_str_items_local:
                                    for it in _loop_var_to_str_items_local[_name_arg]:
                                        alias_name_to_producers[it] = set(src_producers)
                        for an, prods in alias_name_to_producers.items():
                            existing = tensor_alias_producers.get(
                                an, (set(), _src_pl))[0]
                            tensor_alias_producers[an] = (
                                set(existing) | set(prods), _src_pl)
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
                    pass
                if not called_attrs:
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
                        lhs = _line_stmt.targets[0].id
                        try:
                            rhs = ast.unparse(_line_stmt.value)
                        except Exception:
                            rhs = ""
                    else:
                        lhs = None
                        rhs = None
                    _rhs_ast = None
                    if _ast_stmt:
                        _rhs_ast = _ast_stmt.get("rhs_node")
                    if _rhs_ast is None and isinstance(_line_stmt, (ast.Assign, ast.AugAssign)):
                        _rhs_ast = _line_stmt.value
                    if lhs is not None and rhs is not None:
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
                                var_producers[lhs] = (set([a for a in resolved if a in known_attrs]), phys_lineno)
                                _record_lineage(lhs, _vars_in(rhs), phys_lineno)
                                continue
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
                            _compat_aliases = None
                            if tensor_alias_producers:
                                _name_lit_prefix = ""
                                _name_lit_suffix = ""
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
                            _has_self_call = False
                            if isinstance(_rhs_ast, ast.AST):
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
                    _ast_tuple_targets = []
                    if _ast_stmt and _ast_stmt.get("kind") == "assign":
                        _targets = [v for v in (_ast_stmt.get("targets") or []) if v and v != '_']
                        if len(_targets) > 1:
                            _ast_tuple_targets = list(_targets)
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
                    def _filter_chain_by_producer(chain, producer, var_producers):
                        result = []
                        for step in chain or []:
                            producers_of_step = var_producers.get(step.get("var", ""), (set(), None))[0]
                            if not producers_of_step or producer in producers_of_step:
                                result.append(step)
                        return result
                    for producer in consumed_producers:
                        if producer != attr:
                            edges.add((producer, attr))
                            ev = consumed_var_for_producer.get(producer, ("?", phys_lineno))
                            _carrier = ev[0]
                            _chain = []
                            if isinstance(_carrier, str) and _carrier in var_lineage:
                                _chain = list(var_lineage.get(_carrier) or [])
                                _chain = _filter_chain_by_producer(_chain, producer, var_producers)
                            _consumer_text = _line_text(phys_lineno)
                            if _consumer_text:
                                _consumer_carriers = _extract_all_names(call_node)
                                _consumer_step = {"var": _carrier if isinstance(_carrier, str) else "",
                                                  "file": fname,
                                                  "line": phys_lineno,
                                                  "text": _consumer_text,
                                                  "role": "consumer",
                                                  "carriers": _consumer_carriers}
                                if not _chain or _chain[-1].get("line") != phys_lineno:
                                    _chain.append(_consumer_step)
                            edge_locs.setdefault((producer, attr), {
                                "file": fname,
                                "var": ev[0],
                                "from_line": ev[1],
                                "to_line": phys_lineno,
                                "lineage": _chain,
                            })
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
                        if v_clean in var_lineage and var_lineage[v_clean]:
                            var_lineage[v_clean][-1]["carriers"] = list(_ast_lhs_targets)
                            if _ast_assign_like and _ast_assign_like.get("rhs_node") is not None:
                                var_lineage[v_clean][-1]["arg_carriers"] = _extract_all_names(_ast_assign_like.get("rhs_node"))
                    _handled_lhs = True
                if not _handled_lhs:
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
                    if not _shape_handled and called_attrs:
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
                _attrs_to_split = _multi_call_candidates & cycle_nodes
                if _attrs_to_split:
                    split_node_map = {}
                    for a in _attrs_to_split:
                        cnt = _attr_max_calls_in_single_method[a]
                        split_names = [f"{a}#{i}" for i in range(cnt)]
                        split_node_map[a] = split_names
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
                        idx = 0
                        for i, ln in enumerate(sorted_lns):
                            if lineno >= ln:
                                idx = i
                        return min(idx, cnt - 1)
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
