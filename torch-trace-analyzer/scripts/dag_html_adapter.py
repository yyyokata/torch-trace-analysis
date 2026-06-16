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
    endpoint_by_id: dict[int, dict] = field(default_factory=dict)
    port_kind_by_id: dict[int, str] = field(default_factory=dict)
    all_serialized_node_ids: set[int] = field(default_factory=set)
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
    io_groups = _collapse_top_level_io_groups(serialized)
    root_group_ids = [group["id"] for group in root_groups]
    return {
        "input_node_ids": _extract_io_node_ids(serialized["input_nodes"]),
        "param_node_ids": _extract_io_node_ids(serialized["param_nodes"]),
        "const_node_ids": _extract_io_node_ids(serialized["const_nodes"]),
        "output_node_ids": _extract_io_node_ids(serialized["output_nodes"]),
        "io_groups": io_groups,
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
    _record_serialized_node_ids(serialized, state)
    state.raw_edges.extend(serialized["edges"])

    direct_groups: list[dict] = []
    direct_leaves: list[dict] = []
    call_order: list[dict] = []

    for section_name, io_subtype in _IO_SECTION_TO_SUBTYPE.items():
        for entry in serialized[section_name]:
            effective_subtype = entry.get("io_subtype", io_subtype)
            if effective_subtype in {"forward_arg", "return_val"}:
                continue
            leaf = _build_io_leaf(entry, io_subtype=effective_subtype, depth=depth)
            _register_leaf_node(state, leaf, entry)
            direct_leaves.append(leaf)
            call_order.append({"id": leaf["id"], "type": "node"})

    node_entries = serialized["nodes"]
    node_entry_by_id = _collect_node_entry_index(serialized)
    container_child_ids: set[int] = set()
    for entry in node_entries:
        node_id = _require_field(entry, "node_id")
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
        if node_id in state.group_by_id or node_id in state.leaf_nodes_by_id:
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


def _record_serialized_node_ids(serialized: dict, state: _AdapterState) -> None:
    for section_name in ("input_nodes", "param_nodes", "const_nodes", "output_nodes"):
        for entry in serialized[section_name]:
            state.all_serialized_node_ids.add(_require_field(entry, "node_id"))
    for entry in serialized["nodes"]:
        state.all_serialized_node_ids.add(_require_field(entry, "node_id"))


def _collect_node_entry_index(serialized: dict) -> dict[int, dict]:
    node_entry_by_id: dict[int, dict] = {}
    for entry in serialized["nodes"]:
        node_id = _require_field(entry, "node_id")
        if node_id in node_entry_by_id:
            raise RuntimeError(f"duplicate node id in serialized nodes: {node_id}")
        node_entry_by_id[node_id] = entry
    return node_entry_by_id


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
    _register_leaf_node(state, leaf, entry)
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
    class_name = entry.get("class_name") or ""
    attr_name = entry.get("attr_name") or ""
    location_fields = _call_loc_to_src_fields(entry.get("call_loc"))

    children_nodes: list[int] = []
    children_groups: list[dict] = []
    call_order: list[dict] = []
    in_ports: list[dict] = []
    out_ports: list[dict] = []
    if entry.get("children_nodes") is not None:
        child_entry_by_id = node_entry_by_id
        inner_dag = entry.get("inner_dag")
        if inner_dag is not None:
            child_entry_by_id = _collect_node_entry_index(inner_dag)
        for child_id in entry["children_nodes"]:
            if child_id not in child_entry_by_id:
                continue
            child_entry = child_entry_by_id[child_id]
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
        in_ports, out_ports = _collect_group_ports(inner_dag, state=state, parent_id=node_id)
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
        "class_name": class_name,
        "attr_name": attr_name,
        "def_loc": entry.get("def_loc"),
        "class_def_loc": entry.get("class_def_loc"),
        "depth": depth,
        "node_type": node_type,
        "children_nodes": children_nodes,
        "children_groups": children_groups,
        "call_order": call_order,
        "in_ports": in_ports,
        "out_ports": out_ports,
        "internal_edges": [],
        **location_fields,
        **_timing_defaults(),
    }
    _register_group_node(state, group, entry)
    return group


def _collect_group_ports(inner_dag: dict, state: _AdapterState, parent_id: int) -> tuple[list[dict], list[dict]]:
    in_ports: list[dict] = []
    out_ports: list[dict] = []
    for section_name in ("input_nodes", "output_nodes"):
        for entry in inner_dag[section_name]:
            io_subtype = entry.get("io_subtype")
            if io_subtype == "forward_arg":
                node_id = _require_field(entry, "node_id")
                label = _require_field(entry, "label")
                port = {"node_id": node_id, "arg_index": len(in_ports), "label": label}
                in_ports.append(port)
                _register_port_node(state, node_id=node_id, parent_id=parent_id, kind="in", entry=entry)
            elif io_subtype == "return_val":
                node_id = _require_field(entry, "node_id")
                label = _require_field(entry, "label")
                port = {"node_id": node_id, "ret_index": len(out_ports), "label": label}
                out_ports.append(port)
                _register_port_node(state, node_id=node_id, parent_id=parent_id, kind="out", entry=entry)
            elif io_subtype not in {None, _IO_SECTION_TO_SUBTYPE[section_name]}:
                raise RuntimeError(f"unsupported inner_dag io_subtype: {io_subtype}")
    return in_ports, out_ports


def _build_io_leaf(entry: dict, io_subtype: str, depth: int) -> dict:
    node_id = _require_field(entry, "node_id")
    label = _require_field(entry, "label")
    attr_name = _require_field(entry, "attr_name")
    class_name = _require_field(entry, "class_name")
    call_loc = entry.get("call_loc")
    return {
        "id": node_id,
        "label": label,
        "depth": depth,
        "node_type": io_subtype,
        "attr_name": attr_name,
        "class_name": class_name,
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
        "attr_name": _require_field(entry, "attr_name"),
        "class_name": _require_field(entry, "class_name"),
        **_call_loc_to_src_fields(entry.get("call_loc")),
        **_timing_defaults(),
    }


def _route_edges(state: _AdapterState) -> list[dict]:
    global_edges: list[dict] = []
    seen_global: set[tuple[int, int]] = set()
    seen_internal_by_group_id: dict[int, set[tuple[int, int]]] = {}
    known_ids = set(state.endpoint_by_id)
    for edge in state.raw_edges:
        if "src_id" not in edge:
            raise RuntimeError("edge missing src_id")
        if "dst_id" not in edge:
            raise RuntimeError("edge missing dst_id")
        src_id = edge["src_id"]
        dst_id = edge["dst_id"]
        if edge.get("is_containment"):
            continue
        if src_id not in known_ids:
            if src_id in state.all_serialized_node_ids or edge.get("flows"):
                continue
            raise RuntimeError(f"edge src_id {src_id} not found in node index")
        if dst_id not in known_ids:
            if dst_id in state.all_serialized_node_ids or edge.get("flows"):
                continue
            raise RuntimeError(f"edge dst_id {dst_id} not found in node index")
        edge_type = "dep"
        src_node = state.endpoint_by_id[src_id]
        dst_node = state.endpoint_by_id[dst_id]
        parent_src = state.parent_group_of_child.get(src_id)
        parent_dst = state.parent_group_of_child.get(dst_id)

        if parent_src is not None and parent_src == parent_dst:
            group = state.group_by_id[parent_src]
            internal_edge = {
                "type": edge_type,
                "from_node": src_node,
                "to_node": dst_node,
                "parent_class": group["class_name"],
                "flows": edge.get("flows"),
            }
            src_port_kind = state.port_kind_by_id.get(src_id)
            dst_port_kind = state.port_kind_by_id.get(dst_id)
            if src_port_kind is not None:
                if src_port_kind != "in":
                    raise RuntimeError(f"edge src_id {src_id} is not a valid group input port")
                internal_edge["from_port"] = "in"
            else:
                internal_edge["from_child"] = src_id
            if dst_port_kind is not None:
                if dst_port_kind != "out":
                    raise RuntimeError(f"edge dst_id {dst_id} is not a valid group output port")
                internal_edge["to_port"] = "out"
            else:
                internal_edge["to_child"] = dst_id
            edge_key = (src_id, dst_id)
            seen_internal = seen_internal_by_group_id.setdefault(parent_src, set())
            if edge_key in seen_internal:
                continue
            seen_internal.add(edge_key)
            group["internal_edges"].append(internal_edge)
            continue

        edge_key = (src_id, dst_id)
        if edge_key in seen_global:
            continue
        seen_global.add(edge_key)
        global_edges.append(
            {
                "from": src_id,
                "to": dst_id,
                "type": edge_type,
                "from_node": src_node,
                "to_node": dst_node,
                "parent_class": "",
                "flows": edge.get("flows"),
            }
        )
    return global_edges


def _collapse_top_level_io_groups(serialized: dict) -> list[dict]:
    buckets = (
        ("input", serialized["input_nodes"]),
        ("param", serialized["param_nodes"]),
        ("const", serialized["const_nodes"]),
    )
    member_ids_by_subtype: dict[str, list[int]] = {
        subtype: _extract_io_node_ids(entries)
        for subtype, entries in buckets
    }
    member_to_subtype: dict[int, str] = {}
    for subtype, member_ids in member_ids_by_subtype.items():
        for member_id in member_ids:
            if member_id in member_to_subtype:
                raise RuntimeError(f"duplicate top-level io group member id: {member_id}")
            member_to_subtype[member_id] = subtype

    io_groups: list[dict] = []
    for subtype, _entries in buckets:
        member_ids = member_ids_by_subtype[subtype]
        if not member_ids:
            continue
        io_groups.append(
            {
                "id": _top_level_io_group_id(subtype),
                "label": f"{_IO_SUBTYPE_TO_LABEL[subtype]} ({len(member_ids)})",
                "io_subtype": subtype,
                "collapsed": True,
                "member_ids": member_ids,
                "member_count": len(member_ids),
            }
        )
    return io_groups


def _top_level_io_group_id(io_subtype: str) -> str:
    return f"io_group:{io_subtype}"


def _assign_parent_group(state: _AdapterState, child_id: int, parent_id: int) -> None:
    existing_parent = state.parent_group_of_child.get(child_id)
    if existing_parent is not None and existing_parent != parent_id:
        raise RuntimeError(f"child {child_id} already belongs to group {existing_parent}")
    state.parent_group_of_child[child_id] = parent_id


def _register_leaf_node(state: _AdapterState, leaf: dict, entry: dict) -> None:
    node_id = leaf["id"]
    if node_id in state.leaf_nodes_by_id or node_id in state.group_by_id or node_id in state.port_kind_by_id:
        raise RuntimeError(f"duplicate node id in adapter output: {node_id}")
    state.leaf_nodes_by_id[node_id] = leaf
    state.endpoint_by_id[node_id] = _make_endpoint(
        attr_name=_require_field(entry, "attr_name"),
        class_name=_require_field(entry, "class_name"),
        call_loc=entry.get("call_loc"),
    )


def _register_port_node(state: _AdapterState, node_id: int, parent_id: int, kind: str, entry: dict) -> None:
    if kind not in {"in", "out"}:
        raise RuntimeError(f"unsupported port kind: {kind}")
    if node_id in state.leaf_nodes_by_id or node_id in state.group_by_id or node_id in state.port_kind_by_id:
        raise RuntimeError(f"duplicate node id in adapter output: {node_id}")
    state.endpoint_by_id[node_id] = _make_endpoint(
        attr_name=_require_field(entry, "attr_name"),
        class_name=_require_field(entry, "class_name"),
        call_loc=entry.get("call_loc"),
    )
    state.port_kind_by_id[node_id] = kind
    _assign_parent_group(state, node_id, parent_id)


def _register_group_node(state: _AdapterState, group: dict, entry: dict) -> None:
    node_id = group["id"]
    if node_id in state.group_by_id or node_id in state.leaf_nodes_by_id or node_id in state.port_kind_by_id:
        raise RuntimeError(f"duplicate node id in adapter output: {node_id}")
    state.group_by_id[node_id] = group
    state.group_list.append(group)
    state.endpoint_by_id[node_id] = _make_endpoint(
        attr_name=_require_field(entry, "attr_name"),
        class_name=_require_field(entry, "class_name"),
        call_loc=entry.get("call_loc"),
    )


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


def _make_endpoint(attr_name: str, class_name: str, call_loc: dict | None) -> dict:
    if not attr_name:
        raise RuntimeError(f"edge endpoint missing attr_name (class_name={class_name!r})")
    if not class_name:
        raise RuntimeError(f"edge endpoint missing class_name (attr_name={attr_name!r})")
    return {
        "attr_name": attr_name,
        "class_name": class_name,
        "call_loc": call_loc,
    }


def _require_field(mapping: dict, field: str):
    if field not in mapping:
        raise RuntimeError(f"missing field: {field}")
    return mapping[field]


__all__ = ["adapt_serialized_dag"]
