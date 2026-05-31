import ast
from typing import Optional

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
from ast_frontend import ASTFrontend


class AttrScanner:
    _LG_PREFIXES = {"LG."}
    _CONTAINER_LEAF_NAMES = {"ModuleList", "ModuleDict", "Sequential"}
    _LG_RESULT_FUNC_NAME = "result"
    _HEAD_METHOD = "head"
    _NATIVE_MODULE_PREFIXES = ("nn.", "torch.nn.")

    def __init__(self):
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
        for stmt in fe.iter_init_stmts(class_name) or []:
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
        for stmt in fe.iter_init_stmts(class_name) or []:
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
                        class_name=class_name,
                        def_loc=fe._node_loc(stmt),
                        source_expr=fe._local_node_to_text(call),
                        is_native=func_text.startswith(self._NATIVE_MODULE_PREFIXES),
                    )
                )
            )
        return attrs

    def _scan_container_attrs(self, class_name: str, fe: ASTFrontend) -> list[ContainerAttr]:
        attrs: list[ContainerAttr] = []
        for stmt in fe.iter_init_stmts(class_name) or []:
            matched = fe.match_self_attr_call_assign(stmt, func_names=self._CONTAINER_LEAF_NAMES)
            if matched is None:
                continue
            attr_name, call = matched
            container_kind = (fe._expr_leaf_name(call.func) or "").split(".")[-1]
            attrs.append(
                self._register(
                    ContainerAttr(
                        attr_name=attr_name,
                        class_name=class_name,
                        def_loc=fe._node_loc(stmt),
                        container_kind=container_kind,
                        source_expr=fe._local_node_to_text(call),
                    )
                )
            )
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
