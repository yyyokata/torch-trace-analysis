from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ast_types import IntValue, ListValue, Scope

if TYPE_CHECKING:
    from ast_frontend import ASTFrontend


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
        from ast_resolver import ConstantResolver

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
        from ast_resolver import ConstantResolver

        resolver = ConstantResolver(self)
        iv = resolver.eval_int(expr_node, scope)
        if iv is None:
            return None
        return IntValue(iv.value, "callsite_bind")

    def _resolve_bound_list_value(self,
                                  expr_node: ast.expr,
                                  scope_key: Tuple[str, str, str]) -> Optional[ListValue]:
        fname, cname, mname = scope_key
        from ast_resolver import ConstantResolver

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
        from ast_resolver import ConstantResolver

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
        from ast_resolver import ConstantResolver

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

