import ast
import warnings
from typing import Optional

from attr_scanner import AttrScanner
from ast_frontend import ASTFrontend
from attr_types import ContainerAttr, ForwardArgAttr, InputAttr, ModuleAttr, ResultAttr, ReturnValAttr
from dag_types import DAG, DagContext, DataFlowEdge, EvidenceStep, InputNode, ModuleNode, ResultNode, VarEvidence


class DagBuilder:
    _NATIVE_PREFIXES = ("nn.", "torch.nn.")

    def __init__(self, fe: ASTFrontend, scanner: AttrScanner):
        self._fe = fe
        self._scanner = scanner
        self._node_id_counter = 0
        self.registry: dict[int, object] = {}
        self._building: set[str] = set()

    def build(self, root_class: str, const_table=None) -> DagContext:
        self._node_id_counter = 0
        self.registry = {}
        root = self._build_class_dag(root_class)
        return DagContext(
            registry=dict(self.registry),
            root=root,
            const_table=const_table,
            attr_registry=dict(self._scanner.attrs_by_id),
        )

    def _build_class_dag(self, class_name: str) -> DAG:
        if class_name in self._building:
            return DAG(inputs=[], outputs=[], nodes=[], edges=[])
        self._building.add(class_name)
        try:
            attrs = self._scanner.scan_class(class_name, self._fe)
            container_children = [child for container in attrs["containers"] for child in container.items.values()]
            input_ids, output_ids, node_ids, attr_to_node = self._build_nodes(attrs, container_children)
            edges = self._build_containment_edges(attrs["containers"], attr_to_node)
            edges.extend(self._build_dataflow_edges(class_name, attrs, attr_to_node))
            self._attach_inner_dags(attrs["modules"] + container_children, attr_to_node)
            return DAG(inputs=input_ids, outputs=output_ids, nodes=node_ids, edges=edges)
        finally:
            self._building.discard(class_name)

    def _build_nodes(self, attrs_by_kind: dict[str, list], container_children: list[ModuleAttr]):
        input_ids, output_ids, node_ids, attr_to_node = [], [], [], {}
        for attr in attrs_by_kind["inputs"] + attrs_by_kind["forward_args"]:
            node = self._make_node(attr, InputNode)
            input_ids.append(node.node_id)
            attr_to_node[attr.attr_id] = node.node_id
        for attr in attrs_by_kind["results"] + attrs_by_kind["return_vals"]:
            node = self._make_node(attr, ResultNode)
            output_ids.append(node.node_id)
            attr_to_node[attr.attr_id] = node.node_id
        for attr in attrs_by_kind["modules"] + attrs_by_kind["containers"] + container_children:
            node = self._make_node(attr, ModuleNode)
            node_ids.append(node.node_id)
            attr_to_node[attr.attr_id] = node.node_id
        return input_ids, output_ids, node_ids, attr_to_node

    def _make_node(self, attr, node_cls):
        self._node_id_counter += 1
        node = node_cls(node_id=self._node_id_counter, call_loc=attr.def_loc, attr=attr)
        self.registry[node.node_id] = node
        return node

    def _attach_inner_dags(self, module_attrs: list[ModuleAttr], attr_to_node: dict[int, int]) -> None:
        for attr in module_attrs:
            if attr.class_name.startswith(self._NATIVE_PREFIXES):
                continue
            target = self._resolve_class_name(attr.class_name)
            if target is None:
                continue
            node = self.registry[attr_to_node[attr.attr_id]]
            node.inner_dag = self._build_class_dag(target)

    def _resolve_class_name(self, class_name: str) -> Optional[str]:
        if class_name in self._fe.class_registry:
            return class_name
        leaf = class_name.split(".")[-1]
        if leaf in self._fe.class_registry:
            return leaf
        return None

    def _build_containment_edges(self, container_attrs, attr_to_node) -> list[DataFlowEdge]:
        edges = []
        for container in container_attrs:
            src_id = attr_to_node.get(container.attr_id)
            for child in container.items.values():
                dst_id = attr_to_node.get(child.attr_id)
                if src_id is None or dst_id is None:
                    continue
                edges.append(DataFlowEdge(src_id=src_id, dst_id=dst_id, is_containment=True, evidence=[]))
        return edges

    def _build_dataflow_edges(self, class_name: str, attrs: dict[str, list], attr_to_node: dict[int, int]) -> list[DataFlowEdge]:
        module_nodes = {attr.attr_name: attr_to_node[attr.attr_id] for attr in attrs["modules"]}
        input_nodes = {attr.attr_name: attr_to_node[attr.attr_id] for attr in attrs["inputs"] if isinstance(attr, InputAttr)}
        for arg in attrs["forward_args"]:
            input_nodes.setdefault(arg.attr_name, attr_to_node[arg.attr_id])
        var_producers: dict[str, int] = {}
        var_steps: dict[str, list[EvidenceStep]] = {}
        aliases: list[tuple[str, str, object]] = []
        for attr in attrs["forward_args"]:
            node_id = attr_to_node[attr.attr_id]
            var_producers[attr.attr_name] = node_id
            var_steps[attr.attr_name] = [EvidenceStep(loc=attr.def_loc, role="producer", var=attr.attr_name)]
        for stmt in self._fe.iter_forward_stmts(class_name):
            alias = self._fe.match_simple_alias(stmt)
            if alias is not None:
                aliases.append((alias[0], alias[1], stmt))
            dict_read = self._match_input_dict_read(stmt, input_nodes)
            if dict_read is not None:
                lhs, producer_id, step = dict_read
                var_producers[lhs] = producer_id
                producer_loc = self.registry[producer_id].call_loc
                var_steps[lhs] = [EvidenceStep(loc=producer_loc, role="producer", var=lhs), step]
            assign = self._match_self_module_assign(stmt, module_nodes)
            if assign is not None:
                lhs, producer_id = assign
                var_producers[lhs] = producer_id
                var_steps[lhs] = [EvidenceStep(loc=self._fe._node_loc(stmt), role="producer", var=lhs)]
        changed = True
        while changed:
            changed = False
            for lhs, rhs, stmt in aliases:
                producer_id = var_producers.get(rhs)
                if producer_id is None or var_producers.get(lhs) == producer_id:
                    continue
                var_producers[lhs] = producer_id
                var_steps[lhs] = list(var_steps.get(rhs, [])) + [
                    EvidenceStep(loc=self._fe._node_loc(stmt), role="step", var=lhs)
                ]
                changed = True
        edges, path_id = [], 0
        for stmt in self._fe.iter_forward_stmts(class_name):
            call = self._fe.stmt_assign_call_value(stmt)
            if call is None or not isinstance(call.func, ast.Attribute):
                continue
            if not (isinstance(call.func.value, ast.Name) and call.func.value.id == "self"):
                continue
            dst_id = module_nodes.get(call.func.attr)
            if dst_id is None:
                continue
            for arg in call.args:
                if not isinstance(arg, ast.Name):
                    continue
                src_id = var_producers.get(arg.id)
                if src_id is None:
                    continue
                evidence = VarEvidence(
                    var=arg.id,
                    path_id=path_id,
                    steps=list(var_steps.get(arg.id, [])) + [
                        EvidenceStep(loc=self._fe._node_loc(arg), role="consumer", var=arg.id)
                    ],
                )
                edges.append(DataFlowEdge(src_id=src_id, dst_id=dst_id, is_containment=False, evidence=[evidence]))
                path_id += 1
        return edges

    def _match_self_module_assign(self, stmt: ast.stmt, module_nodes: dict[str, int]) -> Optional[tuple[str, int]]:
        target = self._fe.stmt_assign_target(stmt)
        call = self._fe.stmt_assign_call_value(stmt)
        if not isinstance(target, ast.Name) or call is None or not isinstance(call.func, ast.Attribute):
            return None
        if not (isinstance(call.func.value, ast.Name) and call.func.value.id == "self"):
            return None
        producer_id = module_nodes.get(call.func.attr)
        if producer_id is None:
            return None
        return target.id, producer_id

    def _match_input_dict_read(self, stmt: ast.stmt, input_nodes: dict[str, int]):
        target = self._fe.stmt_assign_target(stmt)
        if not isinstance(target, ast.Name) or not isinstance(stmt, ast.Assign):
            return None
        value = stmt.value
        if not isinstance(value, ast.Subscript):
            return None
        key_node = value.slice.value if hasattr(ast, "Index") and isinstance(value.slice, ast.Index) else value.slice
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            producer_id = input_nodes.get(key_node.value)
            if producer_id is None:
                return None
            return (
                target.id,
                producer_id,
                EvidenceStep(loc=self._fe._node_loc(stmt), role="step", var=self._fe._local_node_to_text(value)),
            )
        warnings.warn(
            f"Unknown dict input producer for {self._fe._local_node_to_text(value)}",
            UserWarning,
            stacklevel=2,
        )
        return None
