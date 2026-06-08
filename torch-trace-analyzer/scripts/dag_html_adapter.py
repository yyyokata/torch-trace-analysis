from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


_REQUIRED_TOP_LEVEL_FIELDS = (
    "nodes",
    "edges",
    "input_nodes",
    "param_nodes",
    "const_nodes",
    "output_nodes",
)

_IO_SECTION_TO_SUBTYPE = {
    "input_nodes": "input",
    "param_nodes": "param",
    "const_nodes": "const",
    "output_nodes": "output",
}

_IO_SUBTYPE_TO_LABEL = {
    "input": "Input",
    "param": "Param",
    "const": "Const",
    "output": "Result",
}


@dataclass
class _AdapterState:
    leaf_nodes_by_id: dict[int, dict] = field(default_factory=dict)
    group_by_id: dict[int, dict] = field(default_factory=dict)
    group_list: list[dict] = field(default_factory=list)
    parent_group_of_child: dict[int, int] = field(default_factory=dict)
    label_by_id: dict[int, str] = field(default_factory=dict)
    raw_edges: list[dict] = field(default_factory=list)


@dataclass
class _BuiltNode:
    kind: str
    payload: dict


def adapt_serialized_dag(serialized: dict, model_id: str, mode: str) -> dict:
    _validate_serialized_top_level(serialized)
    state = _AdapterState()
    root_groups, _top_level_leaves, _ = _walk_serialized_level(serialized, depth=0, state=state)
    if not root_groups:
        raise RuntimeError("serialized DAG has no root groups")

    global_edges = _route_edges(state)
    root_group_ids = [group["id"] for group in root_groups]
    return {
        "input_node_ids": _extract_io_node_ids(serialized["input_nodes"]),
        "param_node_ids": _extract_io_node_ids(serialized["param_nodes"]),
        "const_node_ids": _extract_io_node_ids(serialized["const_nodes"]),
        "output_node_ids": _extract_io_node_ids(serialized["output_nodes"]),
        "root_groups": root_group_ids,
        "groups": state.group_list,
        "nodes": list(state.leaf_nodes_by_id.values()),
        "edges": global_edges,
        "meta": {
            "model_id": model_id,
            "mode": mode,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_modules": len(state.group_list) + len(state.leaf_nodes_by_id),
            "roots": [group["label"] for group in root_groups],
        },
        "has_timing": False,
    }


def _validate_serialized_top_level(serialized: dict) -> None:
    for field in _REQUIRED_TOP_LEVEL_FIELDS:
        if field not in serialized:
            raise RuntimeError(f"missing field: {field}")


def _extract_io_node_ids(entries: list[dict]) -> list[int]:
    node_ids: list[int] = []
    for entry in entries:
        node_id = _require_field(entry, "node_id")
        node_ids.append(node_id)
    return node_ids


def _walk_serialized_level(
    serialized: dict,
    depth: int,
    state: _AdapterState,
) -> tuple[list[dict], list[dict], list[dict]]:
    _validate_serialized_top_level(serialized)
    state.raw_edges.extend(serialized["edges"])

    direct_groups: list[dict] = []
    direct_leaves: list[dict] = []
    call_order: list[dict] = []

    for section_name, io_subtype in _IO_SECTION_TO_SUBTYPE.items():
        for entry in serialized[section_name]:
            leaf = _build_io_leaf(entry, io_subtype=io_subtype, depth=depth)
            _register_leaf_node(state, leaf)
            direct_leaves.append(leaf)
            call_order.append({"id": leaf["id"], "type": "node"})

    node_entries = serialized["nodes"]
    node_entry_by_id: dict[int, dict] = {}
    container_child_ids: set[int] = set()
    for entry in node_entries:
        node_id = _require_field(entry, "node_id")
        if node_id in node_entry_by_id:
            raise RuntimeError(f"duplicate node id in serialized nodes: {node_id}")
        node_entry_by_id[node_id] = entry
        if entry.get("children_nodes") is not None:
            children = entry["children_nodes"]
            if not isinstance(children, list):
                raise RuntimeError(f"container node {node_id} children_nodes must be list")
            for child_id in children:
                if child_id in container_child_ids:
                    raise RuntimeError(f"duplicate container child id: {child_id}")
                container_child_ids.add(child_id)

    for entry in node_entries:
        node_id = entry["node_id"]
        if node_id in container_child_ids:
            continue
        built = _build_structured_node(
            entry=entry,
            depth=depth,
            state=state,
            node_entry_by_id=node_entry_by_id,
            visiting=set(),
        )
        if built.kind == "group":
            direct_groups.append(built.payload)
            call_order.append({"id": built.payload["id"], "type": "group"})
        else:
            direct_leaves.append(built.payload)
            call_order.append({"id": built.payload["id"], "type": "node"})
    return direct_groups, direct_leaves, call_order


def _build_structured_node(
    entry: dict,
    depth: int,
    state: _AdapterState,
    node_entry_by_id: dict[int, dict],
    visiting: set[int],
) -> _BuiltNode:
    node_id = _require_field(entry, "node_id")
    if node_id in visiting:
        raise RuntimeError(f"cycle detected while building group tree at node {node_id}")

    if _is_group_node(entry):
        visiting.add(node_id)
        group = _build_group_node(
            entry=entry,
            depth=depth,
            state=state,
            node_entry_by_id=node_entry_by_id,
            visiting=visiting,
        )
        visiting.remove(node_id)
        return _BuiltNode(kind="group", payload=group)

    leaf = _build_non_io_leaf(entry, depth=depth)
    _register_leaf_node(state, leaf)
    return _BuiltNode(kind="leaf", payload=leaf)


def _is_group_node(entry: dict) -> bool:
    children = entry.get("children_nodes")
    return entry.get("inner_dag") is not None or (children is not None and len(children) > 0)


def _build_group_node(
    entry: dict,
    depth: int,
    state: _AdapterState,
    node_entry_by_id: dict[int, dict],
    visiting: set[int],
) -> dict:
    node_id = _require_field(entry, "node_id")
    label = _require_field(entry, "label")
    location_fields = _call_loc_to_src_fields(entry.get("call_loc"))

    children_nodes: list[int] = []
    children_groups: list[dict] = []
    call_order: list[dict] = []
    if entry.get("children_nodes") is not None:
        for child_id in entry["children_nodes"]:
            if child_id not in node_entry_by_id:
                raise RuntimeError(f"container group {node_id} child {child_id} missing node definition")
            child_entry = node_entry_by_id[child_id]
            built = _build_structured_node(
                entry=child_entry,
                depth=depth + 1,
                state=state,
                node_entry_by_id=node_entry_by_id,
                visiting=visiting,
            )
            _assign_parent_group(state, child_id, node_id)
            if built.kind == "group":
                children_groups.append(built.payload)
                call_order.append({"id": built.payload["id"], "type": "group"})
            else:
                children_nodes.append(built.payload["id"])
                call_order.append({"id": built.payload["id"], "type": "node"})
        node_type = "container_group"
    else:
        inner_dag = entry.get("inner_dag")
        if inner_dag is None:
            raise RuntimeError(f"group node {node_id} missing inner_dag")
        child_groups, child_leaves, call_order = _walk_serialized_level(inner_dag, depth=depth + 1, state=state)
        for group in child_groups:
            _assign_parent_group(state, group["id"], node_id)
        for leaf in child_leaves:
            _assign_parent_group(state, leaf["id"], node_id)
        children_groups = child_groups
        children_nodes = [leaf["id"] for leaf in child_leaves]
        node_type = "group"

    group = {
        "id": node_id,
        "label": label,
        "depth": depth,
        "node_type": node_type,
        "children_nodes": children_nodes,
        "children_groups": children_groups,
        "call_order": call_order,
        "internal_edges": [],
        **location_fields,
        **_timing_defaults(),
    }
    _register_group_node(state, group)
    return group


def _build_io_leaf(entry: dict, io_subtype: str, depth: int) -> dict:
    node_id = _require_field(entry, "node_id")
    label = _require_field(entry, "label")
    call_loc = entry.get("call_loc")
    return {
        "id": node_id,
        "label": label,
        "depth": depth,
        "node_type": io_subtype,
        "class_name": label,
        **_call_loc_to_src_fields(call_loc),
        **_timing_defaults(),
    }


def _build_non_io_leaf(entry: dict, depth: int) -> dict:
    node_id = _require_field(entry, "node_id")
    label = _require_field(entry, "label")
    attr_type = _require_field(entry, "attr_type")
    if attr_type not in {"module", "functional"}:
        raise RuntimeError(f"unsupported attr_type: {attr_type}")
    return {
        "id": node_id,
        "label": label,
        "depth": depth,
        "node_type": attr_type,
        "class_name": label,
        **_call_loc_to_src_fields(entry.get("call_loc")),
        **_timing_defaults(),
    }


def _route_edges(state: _AdapterState) -> list[dict]:
    global_edges: list[dict] = []
    known_ids = set(state.label_by_id)
    for edge in state.raw_edges:
        if "src_id" not in edge:
            raise RuntimeError("edge missing src_id")
        if "dst_id" not in edge:
            raise RuntimeError("edge missing dst_id")
        src_id = edge["src_id"]
        dst_id = edge["dst_id"]
        if src_id not in known_ids:
            raise RuntimeError(f"edge src_id {src_id} not found in node index")
        if dst_id not in known_ids:
            raise RuntimeError(f"edge dst_id {dst_id} not found in node index")
        if edge.get("is_containment") is True:
            continue

        edge_type = _edge_type_from_flag(edge.get("is_containment"))
        src_label = state.label_by_id[src_id]
        dst_label = state.label_by_id[dst_id]
        parent_src = state.parent_group_of_child.get(src_id)
        parent_dst = state.parent_group_of_child.get(dst_id)

        if parent_src is not None and parent_src == parent_dst:
            group = state.group_by_id[parent_src]
            from_child = f"{src_id}__out" if src_id in state.group_by_id else src_id
            to_child = f"{dst_id}__in" if dst_id in state.group_by_id else dst_id
            group["internal_edges"].append(
                {
                    "from_child": from_child,
                    "to_child": to_child,
                    "type": edge_type,
                    "from_attr": src_label,
                    "to_attr": dst_label,
                    "parent_class": group["label"],
                    "tensor_info": edge.get("tensor_info"),
                    "evidence": edge.get("evidence"),
                }
            )
            continue

        global_edges.append(
            {
                "from": src_id,
                "to": dst_id,
                "type": edge_type,
                "from_attr": src_label,
                "to_attr": dst_label,
                "parent_class": "",
                "tensor_info": edge.get("tensor_info"),
                "evidence": edge.get("evidence"),
            }
        )
    return global_edges


def _edge_type_from_flag(is_containment: bool | None) -> str:
    if is_containment is True:
        return "containment"
    if is_containment is False:
        return "dep"
    raise RuntimeError(f"unsupported is_containment value: {is_containment!r}")


def _assign_parent_group(state: _AdapterState, child_id: int, parent_id: int) -> None:
    existing_parent = state.parent_group_of_child.get(child_id)
    if existing_parent is not None and existing_parent != parent_id:
        raise RuntimeError(f"child {child_id} already belongs to group {existing_parent}")
    state.parent_group_of_child[child_id] = parent_id


def _register_leaf_node(state: _AdapterState, leaf: dict) -> None:
    node_id = leaf["id"]
    if node_id in state.leaf_nodes_by_id or node_id in state.group_by_id:
        raise RuntimeError(f"duplicate node id in adapter output: {node_id}")
    state.leaf_nodes_by_id[node_id] = leaf
    state.label_by_id[node_id] = leaf["label"]


def _register_group_node(state: _AdapterState, group: dict) -> None:
    node_id = group["id"]
    if node_id in state.group_by_id or node_id in state.leaf_nodes_by_id:
        raise RuntimeError(f"duplicate node id in adapter output: {node_id}")
    state.group_by_id[node_id] = group
    state.group_list.append(group)
    state.label_by_id[node_id] = group["label"]


def _call_loc_to_src_fields(call_loc: dict | None) -> dict:
    if call_loc is None:
        return {
            "src_file": None,
            "src_start_line": None,
            "src_end_line": None,
        }
    file_path = _require_field(call_loc, "file")
    line_no = _require_field(call_loc, "line")
    return {
        "src_file": file_path,
        "src_start_line": line_no,
        "src_end_line": line_no,
    }


def _timing_defaults() -> dict:
    return {
        "has_timing": False,
        "pct": 0,
        "dur_us": 0,
        "kernel_us": 0,
        "role": None,
    }


def _require_field(mapping: dict, field: str):
    if field not in mapping:
        raise RuntimeError(f"missing field: {field}")
    return mapping[field]


__all__ = ["adapt_serialized_dag"]
