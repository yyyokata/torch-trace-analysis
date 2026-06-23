from __future__ import annotations

from attr_types import CallLoc, ContainerAttr, FunctionalAttr, ModuleAttr, _NATIVE_CONTAINER_KINDS
from dag_types import DAG, DagNode, DataFlowEdge, ModuleNode


def normalize_containers_recursive(
    dag: DAG,
    registry: dict[int, DagNode],
    container_attr_map: dict[str, ContainerAttr],
) -> None:
    attr_to_node_ids = _build_attr_to_node_ids(registry)
    next_node_id = max(registry.keys()) + 1 if registry else 0
    _ensure_missing_container_nodes(container_attr_map, registry, attr_to_node_ids, next_node_id)
    _normalize_containers_recursive(
        dag,
        registry,
        attr_to_node_ids=attr_to_node_ids,
        owner_node_id=None,
        visited_dag_ids=set(),
        scope_frames_prefix=("forward",),
        allow_function_grouping=True,
        allow_callloc_grouping=True,
    )


def _normalize_containers_recursive(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
    visited_dag_ids: set[int],
    scope_frames_prefix: tuple[str, ...],
    allow_function_grouping: bool,
    allow_callloc_grouping: bool,
) -> None:
    dag_object_id = id(dag)
    if dag_object_id in visited_dag_ids:
        return
    visited_dag_ids.add(dag_object_id)
    _normalize_single_dag(
        dag,
        registry,
        attr_to_node_ids,
        owner_node_id=owner_node_id,
        scope_frames_prefix=scope_frames_prefix,
        allow_function_grouping=allow_function_grouping,
        allow_callloc_grouping=allow_callloc_grouping,
    )
    _rebuild_adjacency(dag)
    for node_id in list(dag.nodes):
        if owner_node_id is not None and node_id == owner_node_id:
            continue
        node = registry[node_id]
        if not isinstance(node, ModuleNode) or node.inner_dag is None:
            continue
        if node.metadata.get("is_synthetic"):
            continue
        child_scope_frames_prefix = ("forward",)
        _skip_grouping = node.metadata.get("is_container", False) or node.is_native
        child_allow_function_grouping = not _skip_grouping
        child_allow_callloc_grouping = not _skip_grouping
        _normalize_containers_recursive(
            node.inner_dag,
            registry,
            attr_to_node_ids=attr_to_node_ids,
            owner_node_id=node_id,
            visited_dag_ids=visited_dag_ids,
            scope_frames_prefix=child_scope_frames_prefix,
            allow_function_grouping=child_allow_function_grouping,
            allow_callloc_grouping=child_allow_callloc_grouping,
        )


def _normalize_single_dag(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
    scope_frames_prefix: tuple[str, ...],
    allow_function_grouping: bool,
    allow_callloc_grouping: bool,
) -> None:
    container_node_ids = _collect_relevant_container_node_ids(
        dag=dag,
        registry=registry,
        attr_to_node_ids=attr_to_node_ids,
        owner_node_id=owner_node_id,
    )
    for container_node_id in container_node_ids:
        if container_node_id not in dag.nodes:
            dag.nodes.append(container_node_id)
        if container_node_id != owner_node_id and container_node_id not in dag.direct_nodes:
            dag.direct_nodes.append(container_node_id)
        registry[container_node_id].metadata["is_container"] = True

    for node_id in list(dag.nodes):
        node = registry.get(node_id)
        if node is not None and isinstance(node, ModuleNode) and isinstance(node.attr, ContainerAttr):
            node.metadata["is_container"] = True

    # 数据流边原样保留：不重写 boundary edge、不生成 containment 边。
    # 容器与成员的归属关系通过容器节点 inner_dag.direct_nodes 表达。
    for container_node_id in container_node_ids:
        container_node = registry[container_node_id]
        member_ids = _resolve_direct_member_ids(
            container_node=container_node,
            dag=dag,
            attr_to_node_ids=attr_to_node_ids,
        )
        inner_dag = container_node.inner_dag
        if inner_dag is None:
            continue

        if container_node.is_native:
            member_id_set = set(member_ids)
            inner_dag.nodes = sorted(member_ids)
            inner_dag.direct_nodes = sorted(member_ids)
            inner_dag.edges = [
                edge for edge in dag.edges if edge.src_id in member_id_set and edge.dst_id in member_id_set
            ]
            _rebuild_adjacency(inner_dag)
            if owner_node_id != container_node_id:
                for member_id in member_ids:
                    if member_id in dag.direct_nodes:
                        dag.direct_nodes.remove(member_id)
            continue

        for member_id in sorted(member_ids):
            if member_id not in inner_dag.direct_nodes:
                inner_dag.direct_nodes.append(member_id)
            if owner_node_id != container_node_id and member_id in dag.direct_nodes:
                dag.direct_nodes.remove(member_id)

    if allow_function_grouping:
        _apply_function_grouping_a(dag, registry, scope_frames_prefix=scope_frames_prefix)
    if allow_callloc_grouping:
        _apply_function_grouping_b(dag, registry, parent_func=scope_frames_prefix[-1])


def _apply_function_grouping_a(
    dag: DAG,
    registry: dict[int, DagNode],
    scope_frames_prefix: tuple[str, ...],
) -> None:
    buckets: dict[str, list[int]] = {}
    bucket_order: list[str] = []
    for node_id in list(dag.direct_nodes):
        node = registry[node_id]
        helper_frame = _find_a_group_helper_frame(node.call_loc.frames)
        if helper_frame is None:
            continue
        helper_name = helper_frame.function_name
        if helper_name == "__call__":
            continue  # 非 nn.Module 的第三方 class 调用边界，不是用户 helper
        if helper_name not in buckets:
            buckets[helper_name] = []
            bucket_order.append(helper_name)
        buckets[helper_name].append(node_id)

    for helper_name in bucket_order:
        member_ids = buckets[helper_name]
        if not all(
            isinstance(registry[node_id].attr, FunctionalAttr)
            or (isinstance(registry[node_id], ModuleNode) and registry[node_id].is_native)
            for node_id in member_ids
        ):
            continue
        new_node = _build_function_group_node(
            dag=dag,
            registry=registry,
            member_ids=member_ids,
            helper_name=helper_name,
        )
        dag.nodes.append(new_node.node_id)
        dag.direct_nodes = _replace_direct_nodes_with_group(
            dag.direct_nodes,
            member_ids=set(member_ids),
            group_node_id=new_node.node_id,
        )
        registry[new_node.node_id] = new_node


def _find_a_group_helper_frame(frames: tuple) -> object | None:
    last_forward_idx = None
    for idx in range(len(frames) - 1, -1, -1):
        if frames[idx].function_name == "forward":
            last_forward_idx = idx
            break
    if last_forward_idx is None:
        return None
    if last_forward_idx > 0 and frames[last_forward_idx - 1].function_name == "__call__":
        return None
    helper_idx = last_forward_idx + 1
    if helper_idx >= len(frames):
        return None
    return frames[helper_idx]


def _build_function_group_node(
    dag: DAG,
    registry: dict[int, DagNode],
    member_ids: list[int],
    helper_name: str,
) -> ModuleNode:
    representative_node = registry[member_ids[0]]
    frames = representative_node.call_loc.frames
    if not frames:
        raise RuntimeError(
            f"A-group {helper_name} representative node {representative_node.node_id} has empty call_loc.frames"
        )
    helper_frame = _find_a_group_helper_frame(frames)
    if helper_frame is None:
        raise RuntimeError(
            f"A-group {helper_name} representative node {representative_node.node_id} has no helper frame after last forward"
        )
    if not helper_frame.file:
        raise RuntimeError(
            f"A-group {helper_name} representative node {representative_node.node_id} has empty frame file"
        )
    if helper_frame.line <= 0:
        raise RuntimeError(
            f"A-group {helper_name} representative node {representative_node.node_id} has invalid frame line {helper_frame.line}"
        )
    if helper_frame.function_name != helper_name:
        raise RuntimeError(
            f"A-group {helper_name} representative node {representative_node.node_id} helper function {helper_frame.function_name} mismatches helper name"
        )
    member_id_set = set(member_ids)
    return ModuleNode(
        node_id=_next_node_id(registry),
        call_loc=representative_node.call_loc,
        attr=ModuleAttr(
            attr_name=helper_name,
            class_name=helper_name,
            def_loc=CallLoc(file=helper_frame.file, line=helper_frame.line, col=0),
        ),
        metadata={"is_synthetic": True, "synthetic_type": "function_group"},
        inner_dag=DAG(
            inputs=[],
            outputs=[],
            nodes=member_ids.copy(),
            edges=[
                edge for edge in dag.edges if edge.src_id in member_id_set and edge.dst_id in member_id_set
            ],
            direct_nodes=member_ids.copy(),
        ),
        is_native=False,
    )


def _apply_function_grouping_b(
    dag: DAG,
    registry: dict[int, DagNode],
    parent_func: str,
) -> None:
    segments: list[list[int]] = []
    current_seg: list[int] = []
    current_key: tuple[str, str] | None = None
    order = list(getattr(dag, "call_order", None) or [])
    has_functional_direct = any(
        isinstance(registry[node_id].attr, FunctionalAttr)
        or (isinstance(registry[node_id], ModuleNode) and registry[node_id].is_native)
        for node_id in dag.direct_nodes
    )
    has_nonfunctional_direct = any(
        not isinstance(registry[node_id].attr, FunctionalAttr)
        and not (isinstance(registry[node_id], ModuleNode) and registry[node_id].is_native)
        for node_id in dag.direct_nodes
    )
    if not order and has_functional_direct and has_nonfunctional_direct:
        local_node_ids: set[int] = set(dag.direct_nodes)
        seed_order: list[int] = list(dag.direct_nodes)
        for edge in dag.edges:
            if edge.src_id not in local_node_ids:
                local_node_ids.add(edge.src_id)
                seed_order.append(edge.src_id)
            if edge.dst_id not in local_node_ids:
                local_node_ids.add(edge.dst_id)
                seed_order.append(edge.dst_id)
        indegree = {node_id: 0 for node_id in local_node_ids}
        outgoing: dict[int, list[int]] = {node_id: [] for node_id in local_node_ids}
        for edge in dag.edges:
            if edge.src_id not in local_node_ids or edge.dst_id not in local_node_ids:
                continue
            outgoing[edge.src_id].append(edge.dst_id)
            indegree[edge.dst_id] += 1
        zero_indegree = [node_id for node_id in seed_order if indegree[node_id] == 0]
        topo_order: list[int] = []
        while zero_indegree:
            node_id = zero_indegree.pop(0)
            topo_order.append(node_id)
            for dst_id in outgoing[node_id]:
                indegree[dst_id] -= 1
                if indegree[dst_id] == 0:
                    zero_indegree.append(dst_id)
        order = topo_order or list(dag.direct_nodes)
    if not order:
        order = list(dag.direct_nodes)

    for node_id in order:
        node = registry[node_id]
        frames = node.call_loc.frames if node.call_loc is not None else ()
        if (
            (
                isinstance(node.attr, FunctionalAttr)
                or (isinstance(node, ModuleNode) and node.is_native)
            )
            and node.call_loc is not None
            and frames
        ):
            function_name = frames[-1].function_name
            key = (node.call_loc.file, function_name)
            if key == current_key:
                current_seg.append(node_id)
            else:
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [node_id]
                current_key = key
        else:
            if len(current_seg) >= 2:
                segments.append(current_seg)
            current_seg = []
            current_key = None

    if len(current_seg) >= 2:
        segments.append(current_seg)

    original_direct_node_count = len(dag.direct_nodes)
    for member_ids in segments:
        if len(member_ids) == original_direct_node_count:
            continue
        representative_node = registry[member_ids[0]]
        member_id_set = set(member_ids)
        function_name = representative_node.call_loc.frames[-1].function_name
        lines = [registry[node_id].call_loc.line for node_id in member_ids]
        min_line = min(lines)
        max_line = max(lines)
        group_name = f"{function_name}:{min_line}~{max_line}"
        group_node = ModuleNode(
            node_id=_next_node_id(registry),
            call_loc=representative_node.call_loc,
            attr=ModuleAttr(
                attr_name=group_name,
                class_name=group_name,
            ),
            metadata={"is_synthetic": True, "synthetic_type": "callloc_group"},
            inner_dag=DAG(
                inputs=[],
                outputs=[],
                nodes=member_ids.copy(),
                edges=[
                    edge for edge in dag.edges if edge.src_id in member_id_set and edge.dst_id in member_id_set
                ],
                direct_nodes=member_ids.copy(),
            ),
            is_native=False,
        )
        dag.nodes.append(group_node.node_id)
        dag.direct_nodes = _replace_direct_nodes_with_group(
            dag.direct_nodes,
            member_ids=member_id_set,
            group_node_id=group_node.node_id,
        )
        registry[group_node.node_id] = group_node


def _replace_direct_nodes_with_group(
    direct_nodes: list[int],
    member_ids: set[int],
    group_node_id: int,
) -> list[int]:
    new_direct_nodes: list[int] = []
    inserted = False
    for node_id in direct_nodes:
        if node_id in member_ids:
            if not inserted:
                new_direct_nodes.append(group_node_id)
                inserted = True
            continue
        new_direct_nodes.append(node_id)
    return new_direct_nodes


def _next_node_id(registry: dict[int, DagNode]) -> int:
    return max(registry.keys()) + 1 if registry else 0


def _ensure_missing_container_nodes(
    container_attr_map: dict[str, ContainerAttr],
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    next_node_id: int,
) -> int:
    container_items = sorted(
        container_attr_map.items(),
        key=lambda item: item[0].count(".") + 1 if item[0] else 0,
        reverse=True,
    )
    for path, container_attr in container_items:
        if container_attr.class_name not in _NATIVE_CONTAINER_KINDS:
            raise RuntimeError(
                f"Container attr {path or '<root>'} belongs to unsupported container class {container_attr.class_name}"
            )
        if id(container_attr) in attr_to_node_ids:
            continue

        child_ids: list[int] = []
        for child_attr in container_attr.items.values():
            matched_node_ids = attr_to_node_ids.get(id(child_attr))
            if not matched_node_ids:
                # 整个 child 子树已被 DCE 剪除（registry 中无对应节点），跳过。
                continue
            if len(matched_node_ids) != 1:
                raise RuntimeError(
                    f"Container attr {path or '<root>'} child attr {child_attr.attr_name} expected exactly one DagNode, got {matched_node_ids}"
                )
            child_ids.append(matched_node_ids[0])

        if not child_ids:
            continue

        node_id = next_node_id
        next_node_id += 1
        registry[node_id] = ModuleNode(
            node_id=node_id,
            call_loc=container_attr.def_loc or CallLoc(file="<container>", line=0, col=0),
            attr=container_attr,
            metadata={
                "module_path": path,
                "is_container": True,
            },
            inner_dag=DAG(
                inputs=[],
                outputs=[],
                nodes=child_ids.copy(),
                edges=[],
                direct_nodes=child_ids.copy(),
            ),
            is_native=True,
        )
        attr_to_node_ids.setdefault(id(container_attr), []).append(node_id)
    return next_node_id


def _collect_relevant_container_node_ids(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
) -> list[int]:
    owner_attr = registry[owner_node_id].attr if owner_node_id is not None else None
    container_node_ids: list[int] = []
    seen_container_ids: set[int] = set()

    for node_id in list(dag.nodes):
        node = registry[node_id]
        parent_attr = node.attr.parent
        while parent_attr is not None:
            if parent_attr.class_name not in _NATIVE_CONTAINER_KINDS:
                raise RuntimeError(
                    f"Node {node_id} belongs to unsupported container class {parent_attr.class_name}"
                )
            parent_node_ids = attr_to_node_ids.get(id(parent_attr))
            if not parent_node_ids:
                raise RuntimeError(
                    f"Container attr {parent_attr.attr_name} has no registered ModuleNode"
                )
            if len(parent_node_ids) != 1:
                raise RuntimeError(
                    f"Container attr {parent_attr.attr_name} expected exactly one ModuleNode, got {parent_node_ids}"
                )
            container_node_id = parent_node_ids[0]
            if container_node_id not in seen_container_ids:
                seen_container_ids.add(container_node_id)
                container_node_ids.append(container_node_id)
            if owner_attr is not None and parent_attr is owner_attr:
                break
            parent_attr = parent_attr.parent

    return container_node_ids


def _resolve_direct_member_ids(
    container_node: DagNode,
    dag: DAG,
    attr_to_node_ids: dict[int, list[int]],
) -> set[int]:
    if not isinstance(container_node.attr, ContainerAttr):
        raise RuntimeError(f"Container node {container_node.node_id} attr is not ContainerAttr")

    member_ids: set[int] = set()
    for child_attr in container_node.attr.items.values():
        matched_node_ids = attr_to_node_ids.get(id(child_attr))
        if not matched_node_ids:
            # 该子节点已被 DCE 剪除（registry 中无对应节点），跳过。
            continue
        current_level_ids = [node_id for node_id in matched_node_ids if node_id in dag.nodes]
        if not current_level_ids:
            continue
        if len(current_level_ids) != 1:
            raise RuntimeError(
                f"Container {container_node.node_id} child attr {child_attr.attr_name} expected exactly one direct child node in current DAG, got {current_level_ids}"
            )
        member_ids.add(current_level_ids[0])
    return member_ids


def _build_attr_to_node_ids(registry: dict[int, DagNode]) -> dict[int, list[int]]:
    attr_to_node_ids: dict[int, list[int]] = {}
    for node_id, node in registry.items():
        attr_to_node_ids.setdefault(id(node.attr), []).append(node_id)
    return attr_to_node_ids


def _rebuild_adjacency(dag: DAG) -> None:
    in_edges: dict[int, list[DataFlowEdge]] = {}
    out_edges: dict[int, list[DataFlowEdge]] = {}
    for edge in dag.edges:
        out_edges.setdefault(edge.src_id, []).append(edge)
        in_edges.setdefault(edge.dst_id, []).append(edge)
    dag.in_edges = in_edges
    dag.out_edges = out_edges


__all__ = ["normalize_containers_recursive"]
