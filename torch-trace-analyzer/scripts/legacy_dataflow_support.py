#!/usr/bin/env python3

import ast

from source_index import _build_ast_frontends

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


