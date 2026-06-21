from __future__ import annotations

from attr_types import (
    CallLoc,
    ConstantAttr,
    ContainerAttr,
    ForwardArgAttr,
    FunctionalAttr,
    InputAttr,
    ModuleAttr,
    ParamAttr,
    ResultAttr,
    ReturnValAttr,
)
from dag_types import DAG, DagNode, DataFlowEdge, InputNode, ModuleNode, ResultNode


def _serialize_call_loc(loc: CallLoc | None) -> dict | None:
    if loc is None:
        return None
    return {
        "file": loc.file,
        "line": loc.line,
        "col": loc.col,
        "frames": [
            {"file": frame.file, "line": frame.line, "function_name": frame.function_name}
            for frame in loc.frames
        ],
    }


def _format_display_attr_name(attr) -> str:
    """通过 parent/container_index 链追溯生成 display attr_name，不修改 attr 字段。"""
    if attr.parent is None:
        return attr.attr_name

    indices: list[int | str] = []
    cur = attr
    while cur.parent is not None:
        if cur.container_index is None:
            raise RuntimeError(
                f"attr {cur.attr_name!r} has parent container but container_index is None"
            )
        indices.append(cur.container_index)
        cur = cur.parent

    if not cur.attr_name:
        raise RuntimeError(
            f"container path root for attr {attr.attr_name!r} has empty attr_name"
        )

    display = cur.attr_name
    for idx in reversed(indices):
        display += f"[{idx}]"
    return display


def _serialize_io_node(node: DagNode, io_subtype: str) -> dict:
    # 严格分层下，dag.inputs / dag.outputs 在不同层级承载不同 attr 类型：
    #   - 顶层 dag.inputs 为 InputAttr（LG 全局来源），inner_dag.inputs 为 ForwardArgAttr（子图入参端口）
    #   - 顶层 dag.outputs 为 ResultAttr（LG.Result 输出），inner_dag.outputs 为 ReturnValAttr（子图返回端口）
    # 因此 input/output 桶按真实 attr 类型分派出具体 io_subtype。
    if io_subtype == "input":
        if not isinstance(node, InputNode):
            raise RuntimeError(f"node {node.node_id} is not an InputNode")
        if isinstance(node.attr, InputAttr):
            resolved_subtype = "input"
            label = node.attr.attr_name or node.attr.kind or f"input_{node.node_id}"
            attr_name = node.attr.attr_name
            class_name = node.attr.class_name
        elif isinstance(node.attr, ForwardArgAttr):
            resolved_subtype = "forward_arg"
            label = f"arg_{node.attr.arg_index}"
            attr_name = node.attr.attr_name
            class_name = node.attr.class_name
        else:
            raise RuntimeError(f"node {node.node_id} is not an Input/ForwardArg input node")
    elif io_subtype == "param":
        if not isinstance(node, InputNode) or not isinstance(node.attr, ParamAttr):
            raise RuntimeError(f"node {node.node_id} is not a Param input node")
        resolved_subtype = "param"
        label = node.attr.param_name or f"param_{node.node_id}"
        # ParamAttr 的 attr_name 与 param_name 同值（构造点统一赋值），class_name 为 type(_tensor).__name__。
        attr_name = node.attr.attr_name
        class_name = node.attr.class_name
    elif io_subtype == "const":
        if not isinstance(node, InputNode) or not isinstance(node.attr, ConstantAttr):
            raise RuntimeError(f"node {node.node_id} is not a Constant input node")
        resolved_subtype = "const"
        label = f"{node.attr.attr_name}({node.attr.op_name})" if node.attr.attr_name else node.attr.op_name
        attr_name = node.attr.attr_name
        class_name = node.attr.class_name
    elif io_subtype == "output":
        if not isinstance(node, ResultNode):
            raise RuntimeError(f"node {node.node_id} is not a ResultNode")
        if isinstance(node.attr, ResultAttr):
            resolved_subtype = "output"
            label = node.attr.head_name or f"output_{node.node_id}"
            attr_name = node.attr.head_name
            class_name = node.attr.class_name
        elif isinstance(node.attr, ReturnValAttr):
            resolved_subtype = "return_val"
            label = node.attr.ret_key or f"ret_{node.attr.ret_index}"
            attr_name = node.attr.ret_key or f"ret_{node.attr.ret_index}"
            class_name = node.attr.class_name
        else:
            raise RuntimeError(f"node {node.node_id} is not a Result/ReturnVal output node")
    else:
        raise RuntimeError(f"unsupported io subtype: {io_subtype}")

    if not attr_name:
        raise RuntimeError(
            f"io node {node.node_id} (subtype={resolved_subtype}) has empty attr_name"
        )
    if not class_name:
        raise RuntimeError(
            f"io node {node.node_id} (subtype={resolved_subtype}) has empty class_name"
        )

    return {
        "node_id": node.node_id,
        "io_subtype": resolved_subtype,
        "label": label,
        "attr_name": attr_name,
        "class_name": class_name,
        "call_loc": _serialize_call_loc(node.call_loc),
    }


def _serialize_module_node(node: ModuleNode, dag: DAG, registry: dict[int, DagNode]) -> dict:
    if node.metadata.get("is_container") is True:
        if not isinstance(node.attr, ContainerAttr):
            raise RuntimeError(f"container node {node.node_id} attr must be ContainerAttr")
        attr_name = _format_display_attr_name(node.attr)
        class_name = node.attr.container_kind
        label = (
            f"{attr_name}({class_name})"
            if attr_name
            else class_name
        )
        attr_id_to_node_id = {id(n.attr): nid for nid, n in registry.items()}
        children_nodes = [
            attr_id_to_node_id[id(child_attr)]
            for child_attr in node.attr.items.values()
            if id(child_attr) in attr_id_to_node_id
        ]
        return {
            "node_id": node.node_id,
            "label": label,
            "attr_name": attr_name,
            "class_name": class_name,
            "call_loc": _serialize_call_loc(node.call_loc),
            "is_container": True,
            "container_kind": node.attr.container_kind,
            "attr_type": "container",
            "children_nodes": children_nodes,
            "inner_dag": (
                serialize_dag(node.inner_dag, registry)
                if node.inner_dag is not None
                else None
            ),
        }

    if isinstance(node.attr, ModuleAttr):
        attr_type = "module"
        attr_name = _format_display_attr_name(node.attr)
        class_name = node.attr.class_name
    elif isinstance(node.attr, FunctionalAttr):
        attr_type = "functional"
        attr_name = _format_display_attr_name(node.attr)
        class_name = node.attr.class_name
    else:
        raise RuntimeError(f"module node {node.node_id} has unsupported attr type {type(node.attr).__name__}")

    if attr_name and class_name:
        label = f"{attr_name}({class_name})"
    elif attr_name:
        label = attr_name
    elif class_name:
        label = class_name
    else:
        raise RuntimeError(f"module node {node.node_id} has empty attr_name and class_name")

    return {
        "node_id": node.node_id,
        "label": label,
        "attr_name": attr_name,
        "class_name": class_name,
        "call_loc": _serialize_call_loc(node.call_loc),
        "def_loc": _serialize_call_loc(node.attr.def_loc) if isinstance(node.attr, ModuleAttr) else None,
        "class_def_loc": _serialize_call_loc(node.attr.class_def_loc) if isinstance(node.attr, ModuleAttr) else None,
        "attr_type": attr_type,
        "is_native": node.is_native,
        **node.metadata,
        "inner_dag": (
            serialize_dag(node.inner_dag, registry)
            if node.inner_dag is not None
            else None
        ),
    }


def _serialize_edge(edge: DataFlowEdge) -> dict:
    return {
        "src_id": edge.src_id,
        "dst_id": edge.dst_id,
        "flows": {
            str(idx): {
                "shape": tf.shape,
                "dtype": tf.dtype,
                "steps": [
                    {"loc": _serialize_call_loc(step.loc), "var": step.var}
                    for step in tf.steps
                ],
            }
            for idx, tf in edge.flows.items()
        },
    }


def _get_registry_node(node_id: int, registry: dict[int, DagNode]) -> DagNode:
    if node_id not in registry:
        raise RuntimeError(f"node_id {node_id} not found in registry")
    return registry[node_id]


def serialize_dag(dag: DAG, registry: dict[int, DagNode]) -> dict:
    return {
        "input_nodes": [
            _serialize_io_node(_get_registry_node(node_id, registry), "input") for node_id in dag.inputs
        ],
        "param_nodes": [
            _serialize_io_node(_get_registry_node(node_id, registry), "param") for node_id in dag.params
        ],
        "const_nodes": [
            _serialize_io_node(_get_registry_node(node_id, registry), "const") for node_id in dag.consts
        ],
        "output_nodes": [
            _serialize_io_node(_get_registry_node(node_id, registry), "output") for node_id in dag.outputs
        ],
        "nodes": [_serialize_dag_node(node_id, dag, registry) for node_id in dag.direct_nodes],
        "edges": [_serialize_edge(edge) for edge in dag.edges],
    }


def _serialize_dag_node(node_id: int, dag: DAG, registry: dict[int, DagNode]) -> dict:
    node = _get_registry_node(node_id, registry)
    if not isinstance(node, ModuleNode):
        raise RuntimeError(f"dag.direct_nodes contains non-ModuleNode {node_id}")
    return _serialize_module_node(node, dag, registry)


__all__ = ["serialize_dag"]
