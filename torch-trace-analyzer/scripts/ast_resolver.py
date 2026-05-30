from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

try:
    from scripts.ast_constants import ConstantTable
    from scripts.ast_types import IntValue, Scope, _RECURSING
except ModuleNotFoundError:
    from ast_constants import ConstantTable
    from ast_types import IntValue, Scope, _RECURSING

if TYPE_CHECKING:
    from scripts.ast_frontend import ASTFrontend


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

