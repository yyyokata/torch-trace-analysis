import ast
from typing import Iterator, Optional

from ast_frontend import ASTFrontend
from ast_resolver import ConstantResolver
from ast_types import Scope
from attr_types import (
    AttrType,
    CallLoc,
    ContainerAttr,
    ForwardArgAttr,
    InputAttr,
    ModuleAttr,
    ResultAttr,
    ReturnValAttr,
)


class AttrScanner:
    _LG_PREFIXES = {"LG."}
    _CONTAINER_LEAF_NAMES = {"ModuleList", "ModuleDict", "Sequential"}
    _LG_RESULT_FUNC_NAME = "result"
    _HEAD_METHOD = "head"
    _NATIVE_MODULE_PREFIXES = ("nn.", "torch.nn.")
    _TRAIN_KEYWORDS = {"train", "training"}
    _INFER_KEYWORDS = {"infer", "inference", "serving", "serve"}

    def __init__(self, resolver: Optional[ConstantResolver] = None, conditional_mode: str = "infer"):
        if conditional_mode not in {"infer", "train"}:
            raise ValueError(f"Invalid conditional_mode: {conditional_mode}")
        self._resolver = resolver
        self._conditional_mode = conditional_mode
        self._attr_id_counter = 0
        self.attrs_by_id: dict[int, AttrType] = {}

    def _next_id(self) -> int:
        self._attr_id_counter += 1
        return self._attr_id_counter

    def _register(self, attr: AttrType) -> AttrType:
        attr.attr_id = self._next_id()
        self.attrs_by_id[attr.attr_id] = attr
        return attr

    def scan_class(self, class_name: str, fe: ASTFrontend) -> dict[str, list[AttrType]]:
        return {
            "inputs": self._scan_input_attrs(class_name, fe),
            "modules": self._scan_module_attrs(class_name, fe),
            "containers": self._scan_container_attrs(class_name, fe),
            "results": self._scan_result_attrs(class_name, fe),
            "forward_args": self._scan_forward_arg_attrs(class_name, fe),
            "return_vals": self._scan_return_val_attrs(class_name, fe),
        }

    def _scan_input_attrs(self, class_name: str, fe: ASTFrontend) -> list[InputAttr]:
        attrs: list[InputAttr] = []
        for stmt in self._iter_resolved_init_stmts(class_name, fe):
            attrs.extend(self._scan_dict_backed_inputs(stmt, class_name, fe))
            matched = fe.match_self_attr_call_assign(stmt, func_prefixes=self._LG_PREFIXES)
            if matched is None:
                continue
            attr_name, call = matched
            func_text = fe.match_call_by_func(call, func_prefixes=self._LG_PREFIXES)
            if func_text is None:
                continue
            attrs.append(
                self._register(
                    InputAttr(
                        attr_name=attr_name,
                        class_name=class_name,
                        def_loc=fe._node_loc(stmt),
                        kind=func_text.split(".")[-1],
                        source_expr=fe._local_node_to_text(call),
                        is_native=False,
                    )
                )
            )
        return attrs

    def _scan_module_attrs(self, class_name: str, fe: ASTFrontend) -> list[ModuleAttr]:
        attrs: list[ModuleAttr] = []
        for stmt in self._iter_resolved_init_stmts(class_name, fe):
            matched = fe.match_self_attr_call_assign(stmt)
            if matched is None:
                continue
            attr_name, call = matched
            if fe.match_call_by_func(call, func_prefixes=self._LG_PREFIXES) is not None:
                continue
            leaf_name = (fe._expr_leaf_name(call.func) or "").split(".")[-1]
            if leaf_name in self._CONTAINER_LEAF_NAMES:
                continue
            func_text = fe.call_func_text(call)
            attrs.append(
                self._register(
                    ModuleAttr(
                        attr_name=attr_name,
                        class_name=func_text,
                        def_loc=fe._node_loc(stmt),
                        source_expr=fe._local_node_to_text(call),
                        is_native=func_text.startswith(self._NATIVE_MODULE_PREFIXES),
                    )
                )
            )
        return attrs

    def _scan_container_attrs(self, class_name: str, fe: ASTFrontend) -> list[ContainerAttr]:
        attrs: list[ContainerAttr] = []
        for stmt in self._iter_resolved_init_stmts(class_name, fe):
            matched = fe.match_self_attr_call_assign(stmt, func_names=self._CONTAINER_LEAF_NAMES)
            if matched is None:
                continue
            attr_name, call = matched
            func_text = fe.call_func_text(call)
            container_kind = (fe._expr_leaf_name(call.func) or "").split(".")[-1]
            container = self._register(
                ContainerAttr(
                    attr_name=attr_name,
                    class_name=func_text,
                    def_loc=fe._node_loc(stmt),
                    container_kind=container_kind,
                    source_expr=fe._local_node_to_text(call),
                )
            )
            for key, child in self._expand_container_items(class_name, fe, container, call):
                container.add_child(key, self._register(child))
            attrs.append(container)
        return attrs

    def _scan_result_attrs(self, class_name: str, fe: ASTFrontend) -> list[ResultAttr]:
        method_node = fe._get_method_node(class_name, "forward")
        if method_node is None:
            return []
        result_defs: dict[str, CallLoc] = {}
        results: list[ResultAttr] = []
        stmts = self._sorted_stmts(method_node)
        for stmt in stmts:
            target = fe.stmt_assign_target(stmt)
            call = fe.stmt_assign_call_value(stmt)
            if target is not None and call is not None and isinstance(target, ast.Name):
                if fe.match_call_by_func(
                    call,
                    func_prefixes=self._LG_PREFIXES,
                    func_names={self._LG_RESULT_FUNC_NAME},
                ) is not None:
                    result_defs[target.id] = fe._node_loc(stmt)
                    continue
            matched = fe.match_expr_or_stmt_call(stmt, func_attr=self._HEAD_METHOD)
            if matched is None:
                continue
            owner_text, head_call = matched
            if owner_text not in result_defs:
                continue
            args = fe.call_arg_texts(head_call)
            kwargs = fe.call_kwarg_texts(head_call)
            results.append(
                self._register(
                    ResultAttr(
                        attr_name=owner_text,
                        class_name=class_name,
                        def_loc=result_defs[owner_text],
                        source_expr=fe._local_node_to_text(head_call),
                        head_name=kwargs.get("name") or self._arg_at(args, 0),
                        classifier_type=kwargs.get("classifier_type") or self._arg_at(args, 5),
                        prediction_expr=kwargs.get("prediction") or self._arg_at(args, 1),
                        label_expr=kwargs.get("label") or self._arg_at(args, 2),
                        sample_rate_expr=kwargs.get("sample_rate") or self._arg_at(args, 3),
                        loss_expr=kwargs.get("loss") or self._arg_at(args, 4),
                        is_native=False,
                    )
                )
            )
        return results

    def _scan_forward_arg_attrs(self, class_name: str, fe: ASTFrontend) -> list[ForwardArgAttr]:
        method_node = fe._get_method_node(class_name, "forward")
        if method_node is None:
            return []
        args = list(method_node.args.args)
        if args and args[0].arg == "self":
            args = args[1:]
        return [
            self._register(
                ForwardArgAttr(
                    attr_name=arg.arg,
                    class_name=class_name,
                    def_loc=fe._node_loc(arg),
                    arg_index=index,
                )
            )
            for index, arg in enumerate(args)
        ]

    def _scan_return_val_attrs(self, class_name: str, fe: ASTFrontend) -> list[ReturnValAttr]:
        method_node = fe._get_method_node(class_name, "forward")
        if method_node is None:
            return []
        attrs: list[ReturnValAttr] = []
        for return_node in self._sorted_returns(method_node):
            if return_node.value is None:
                continue
            if isinstance(return_node.value, ast.Tuple):
                for index, _elt in enumerate(return_node.value.elts):
                    attrs.append(
                        self._register(
                            ReturnValAttr(
                                attr_name=f"return_{index}",
                                class_name=class_name,
                                def_loc=fe._node_loc(return_node),
                                ret_index=index,
                            )
                        )
                    )
                continue
            attrs.append(
                self._register(
                    ReturnValAttr(
                        attr_name="return_0",
                        class_name=class_name,
                        def_loc=fe._node_loc(return_node),
                        ret_index=0,
                    )
                )
            )
        return attrs

    def _iter_resolved_init_stmts(self, class_name: str, fe: ASTFrontend) -> Iterator[ast.stmt]:
        def _walk(body):
            for stmt in body:
                if isinstance(stmt, ast.If):
                    branch = self._select_if_branch(class_name, fe, stmt)
                    yield from _walk(branch)
                    continue
                yield stmt

        yield from _walk(list(fe.iter_init_stmts_with_branches(class_name) or []))

    def _select_if_branch(self, class_name: str, fe: ASTFrontend, stmt: ast.If) -> list[ast.stmt]:
        scope = Scope(file=fe.path or "", cls=class_name, method="__init__")
        if self._resolver is not None:
            resolved = self._resolver.eval_bool(stmt.test, scope)
            if resolved is not None:
                return stmt.body if resolved else stmt.orelse
        mode_branch = self._resolve_mode_branch(stmt.test)
        if mode_branch is None:
            raise ValueError(f"Cannot resolve branch condition: {fe._local_node_to_text(stmt.test)}")
        return stmt.body if mode_branch else stmt.orelse

    def _resolve_mode_branch(self, test: ast.expr) -> Optional[bool]:
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
            lit = self._string_literal(test.left) or self._string_literal(test.comparators[0])
            if lit in self._TRAIN_KEYWORDS | self._INFER_KEYWORDS:
                wants_train = self._conditional_mode == "train"
                positive = lit in self._TRAIN_KEYWORDS
                if isinstance(test.ops[0], ast.Eq):
                    return wants_train if positive else not wants_train
                if isinstance(test.ops[0], ast.NotEq):
                    return not wants_train if positive else wants_train
        text = ast.unparse(test).lower() if hasattr(ast, "unparse") else ""
        if any(word in text for word in self._TRAIN_KEYWORDS):
            return self._conditional_mode == "train"
        if any(word in text for word in self._INFER_KEYWORDS):
            return self._conditional_mode == "infer"
        return None

    def _scan_dict_backed_inputs(self, stmt: ast.stmt, class_name: str, fe: ASTFrontend) -> list[InputAttr]:
        target = fe.stmt_assign_target(stmt)
        if target is None or not isinstance(stmt, ast.Assign):
            return []
        owner_attr = fe._extract_self_attr_name(target)
        if owner_attr is None or not isinstance(stmt.value, ast.Dict):
            return []
        out: list[InputAttr] = []
        for key_node, value_node in zip(stmt.value.keys, stmt.value.values):
            key = self._string_literal(key_node)
            if key is None or not isinstance(value_node, ast.Call):
                continue
            func_text = fe.match_call_by_func(value_node, func_prefixes=self._LG_PREFIXES)
            if func_text is None:
                continue
            input_name = self._input_name_from_call(value_node, key)
            out.append(
                self._register(
                    InputAttr(
                        attr_name=input_name,
                        class_name=class_name,
                        def_loc=fe._node_loc(value_node),
                        kind=func_text.split(".")[-1],
                        owner_expr=owner_attr,
                        slot_expr=key,
                        source_expr=fe._local_node_to_text(value_node),
                        is_native=False,
                    )
                )
            )
        return out

    def _expand_container_items(
        self,
        class_name: str,
        fe: ASTFrontend,
        container: ContainerAttr,
        call: ast.Call,
    ) -> list[tuple[int | str, ModuleAttr]]:
        if self._resolver is None or not call.args:
            return []
        items_node = call.args[0]
        if container.container_kind in {"ModuleList", "Sequential"}:
            return self._expand_list_like_items(class_name, fe, container, items_node)
        if container.container_kind == "ModuleDict" and isinstance(items_node, ast.Dict):
            out = []
            for key_node, value_node in zip(items_node.keys, items_node.values):
                key = self._string_literal(key_node)
                if key is None or not isinstance(value_node, ast.Call):
                    continue
                child = self._child_module_attr(fe, container, key, value_node)
                if child is not None:
                    out.append((key, child))
            return out
        return []

    def _expand_list_like_items(
        self,
        class_name: str,
        fe: ASTFrontend,
        container: ContainerAttr,
        items_node: ast.expr,
    ) -> list[tuple[int, ModuleAttr]]:
        if isinstance(items_node, ast.List):
            out = []
            for index, item in enumerate(items_node.elts):
                if not isinstance(item, ast.Call):
                    continue
                child = self._child_module_attr(fe, container, index, item)
                if child is not None:
                    out.append((index, child))
            return out
        if isinstance(items_node, ast.ListComp) and isinstance(items_node.elt, ast.Call):
            scope = Scope(file=fe.path or "", cls=class_name, method="__init__")
            count = self._resolve_listcomp_count(class_name, fe, items_node, scope)
            if count is None:
                return []
            out = []
            for index in range(count):
                child = self._child_module_attr(fe, container, index, items_node.elt)
                if child is not None:
                    out.append((index, child))
            return out
        return []

    def _resolve_listcomp_count(
        self,
        class_name: str,
        fe: ASTFrontend,
        node: ast.ListComp,
        scope: Scope,
    ) -> Optional[int]:
        count = self._resolver.eval_list_len(node, scope)
        if count is not None:
            return count
        if len(node.generators) != 1:
            return None
        iterator = node.generators[0].iter
        if not isinstance(iterator, ast.Call):
            return None
        if not isinstance(iterator.func, ast.Name) or iterator.func.id != "range" or len(iterator.args) != 1:
            return None
        resolved = self._resolver.eval_int(iterator.args[0], scope)
        if resolved is not None:
            return resolved.value
        if isinstance(iterator.args[0], ast.Name):
            return self._lookup_init_param_default(class_name, fe, iterator.args[0].id)
        return None

    def _lookup_init_param_default(self, class_name: str, fe: ASTFrontend, arg_name: str) -> Optional[int]:
        method_node = fe._get_method_node(class_name, "__init__")
        if method_node is None:
            return None
        args = list(method_node.args.args)
        defaults = list(method_node.args.defaults)
        start = len(args) - len(defaults)
        for arg, default in zip(args[start:], defaults):
            if arg.arg == arg_name and isinstance(default, ast.Constant) and isinstance(default.value, int):
                return default.value
        return None

    def _child_module_attr(
        self,
        fe: ASTFrontend,
        container: ContainerAttr,
        key: int | str,
        call: ast.Call,
    ) -> Optional[ModuleAttr]:
        func_text = fe.call_func_text(call)
        return ModuleAttr(
            attr_name=f"{container.attr_name}[{key}]",
            class_name=func_text,
            def_loc=fe._node_loc(call),
            source_expr=fe._local_node_to_text(call),
            is_native=func_text.startswith(self._NATIVE_MODULE_PREFIXES),
        )

    def _input_name_from_call(self, call: ast.Call, default_name: str) -> str:
        for kw in call.keywords:
            if kw.arg == "name":
                lit = self._string_literal(kw.value)
                if isinstance(lit, str):
                    return lit
        return default_name

    def _string_literal(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _sorted_stmts(self, method_node: ast.AST) -> list[ast.stmt]:
        stmts = [node for node in ast.walk(method_node) if isinstance(node, ast.stmt)]
        return sorted(
            stmts,
            key=lambda node: (
                getattr(node, "lineno", 10**9),
                getattr(node, "col_offset", 10**9),
            ),
        )

    def _sorted_returns(self, method_node: ast.AST) -> list[ast.Return]:
        returns = [node for node in ast.walk(method_node) if isinstance(node, ast.Return)]
        return sorted(
            returns,
            key=lambda node: (
                getattr(node, "lineno", 10**9),
                getattr(node, "col_offset", 10**9),
            ),
        )

    def _arg_at(self, args: list[str], index: int) -> Optional[str]:
        if 0 <= index < len(args):
            return args[index]
        return None
