import ast
from collections import defaultdict

try:
    from scripts.ast_constants import ConstantTable
    from scripts.ast_resolver import ConstantResolver
    from scripts.ast_types import Scope
except ModuleNotFoundError:
    from ast_constants import ConstantTable
    from ast_resolver import ConstantResolver
    from ast_types import Scope


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
        owner_value = owner.value
        if isinstance(owner_value, ast.Name):
            owner_attr = owner_value.id
        else:
            owner_attr = self._extract_self_attr_name(owner_value)
        if not owner_attr:
            return None
        if not ("dict" in owner_attr or "fc" in owner_attr):
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

