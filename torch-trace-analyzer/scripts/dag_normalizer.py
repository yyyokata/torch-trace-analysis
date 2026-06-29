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
    # 模式 1 收缩法需要每个游离节点的全部真实数据流边（包括连到 scope 外节点，
    # 如 result/head 的边）。各 inner_dag.edges 只含自身节点内部边，不足以判定，
    # 因此在任何分组/重写发生前快照顶层全局边集合，向下层层传递。
    global_edges = list(dag.edges)
    pattern_registry: dict[tuple[tuple[str, ...], tuple[tuple[int, int], ...]], str] = {}
    pattern_counter: dict[str, int] = {}
    _normalize_containers_recursive(
        dag,
        registry,
        attr_to_node_ids=attr_to_node_ids,
        owner_node_id=None,
        visited_dag_ids=set(),
        scope_frames_prefix=("forward",),
        allow_function_grouping=True,
        allow_callloc_grouping=True,
        global_edges=global_edges,
        pattern_registry=pattern_registry,
        pattern_counter=pattern_counter,
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
    global_edges: list[DataFlowEdge],
    pattern_registry: dict[tuple[tuple[str, ...], tuple[tuple[int, int], ...]], str],
    pattern_counter: dict[str, int],
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
        global_edges=global_edges,
        pattern_registry=pattern_registry,
        pattern_counter=pattern_counter,
    )
    _rebuild_adjacency(dag)
    for node_id in list(dag.nodes):
        if owner_node_id is not None and node_id == owner_node_id:
            continue
        node = registry[node_id]
        if not isinstance(node, ModuleNode) or node.inner_dag is None:
            continue
        if node.metadata.get("synthetic_type") == "function_group":
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
            global_edges=global_edges,
            pattern_registry=pattern_registry,
            pattern_counter=pattern_counter,
        )


def _normalize_single_dag(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
    scope_frames_prefix: tuple[str, ...],
    allow_function_grouping: bool,
    allow_callloc_grouping: bool,
    global_edges: list[DataFlowEdge],
    pattern_registry: dict[tuple[tuple[str, ...], tuple[tuple[int, int], ...]], str],
    pattern_counter: dict[str, int],
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
            # 模式 1 容器扩展归入的游离节点持久化在 metadata 中，
            # 原生容器每次被访问都会从 attr.items 重建 inner_dag，
            # 因此必须把 absorbed 节点并回成员集合，否则会被重建逻辑抹掉。
            absorbed_ids = list(container_node.metadata.get("mode1_absorbed", []))
            effective_member_ids = sorted(set(member_ids) | set(absorbed_ids))
            member_id_set = set(effective_member_ids)
            inner_dag.nodes = list(effective_member_ids)
            inner_dag.direct_nodes = list(effective_member_ids)
            inner_dag.edges = [
                edge for edge in dag.edges if edge.src_id in member_id_set and edge.dst_id in member_id_set
            ]
            _rebuild_adjacency(inner_dag)
            if owner_node_id != container_node_id:
                for member_id in effective_member_ids:
                    if member_id in dag.direct_nodes:
                        dag.direct_nodes.remove(member_id)
            continue

        for member_id in sorted(member_ids):
            if member_id not in inner_dag.direct_nodes:
                inner_dag.direct_nodes.append(member_id)
            if owner_node_id != container_node_id and member_id in dag.direct_nodes:
                dag.direct_nodes.remove(member_id)

    # 模式 1 容器扩展：必须在 callloc_group（B-group）形成之前执行，
    # 否则游离 op 会先被 B-group 吃掉，无法再归入容器。
    apply_container_expansion_mode1(
        dag, registry, owner_node_id=owner_node_id, global_edges=global_edges
    )

    if allow_callloc_grouping:
        _apply_function_grouping_b(
            dag,
            registry,
            parent_func=scope_frames_prefix[-1],
            pattern_registry=pattern_registry,
            pattern_counter=pattern_counter,
        )
    if allow_function_grouping:
        _apply_function_grouping_a(dag, registry, scope_frames_prefix=scope_frames_prefix)


def _is_container_node(node: DagNode) -> bool:
    return (
        isinstance(node, ModuleNode)
        and node.inner_dag is not None
        and (bool(node.metadata.get("is_container")) or isinstance(node.attr, ContainerAttr))
    )


def apply_container_expansion_mode1(
    dag: DAG,
    registry: dict[int, DagNode],
    owner_node_id: int | None,
    global_edges: list[DataFlowEdge],
) -> None:
    """模式 1 容器扩展（收缩法）。

    把游离在容器外、但前后向仅连接容器子孙（或同批一起归入的游离节点）的 op，
    整体归入对应容器。算法见 design/grouping_mode1_container_expansion.md。

    - 多容器竞争（游离节点的边跨越互不嵌套的两个容器）：不归入任何容器（不做 LCA）。
    - 嵌套容器：归入最内层（descendant 最少的）满足条件的容器。
    - 模式 1 必须在 callloc_group 形成之前调用。
    - 必须使用全局边集合 ``global_edges``，因为各 scope 的 ``inner_dag.edges``
      不含连到 scope 外节点（如 result/head）的边，会导致收缩法误判。
    """
    container_ids: list[int] = []
    for node_id in dag.nodes:
        node = registry.get(node_id)
        if node is None:
            raise RuntimeError(f"mode1: dag node {node_id} missing from registry")
        if node_id == owner_node_id:
            continue
        if _is_container_node(node):
            container_ids.append(node_id)
    if not container_ids:
        return

    # desc[C] = C 的全部子孙节点（递归展开 inner_dag），用于成员判定。
    desc: dict[int, set[int]] = {}

    def _descendants(container_id: int) -> set[int]:
        cached = desc.get(container_id)
        if cached is not None:
            return cached
        container_node = registry[container_id]
        if container_node.inner_dag is None:
            raise RuntimeError(f"mode1: container {container_id} has no inner_dag")
        acc: set[int] = set()
        for member_id in container_node.inner_dag.nodes:
            acc.add(member_id)
            member_node = registry.get(member_id)
            if member_node is not None and _is_container_node(member_node):
                acc |= _descendants(member_id)
        desc[container_id] = acc
        return acc

    for container_id in container_ids:
        _descendants(container_id)

    def _min_container(node_id: int) -> int | None:
        """返回包含 node_id 的最内层容器（descendant 最少者）；不在任何容器内则 None。"""
        candidates = [cid for cid in container_ids if node_id in desc[cid]]
        if not candidates:
            return None
        return min(candidates, key=lambda cid: len(desc[cid]))

    container_id_set = set(container_ids)
    floating: set[int] = set()
    for node_id in dag.direct_nodes:
        if node_id == owner_node_id:
            continue
        node = registry.get(node_id)
        if node is None:
            raise RuntimeError(f"mode1: direct node {node_id} missing from registry")
        if node_id in container_id_set:
            continue
        if isinstance(node, ModuleNode) and node.metadata.get("is_synthetic"):
            # 模式 1 在分组之前运行，正常不应出现 synthetic group；出现即属异常。
            raise RuntimeError(
                f"mode1: unexpected synthetic node {node_id} present before grouping"
            )
        floating.add(node_id)
    if not floating:
        return

    in_neighbors: dict[int, list[int]] = {node_id: [] for node_id in floating}
    out_neighbors: dict[int, list[int]] = {node_id: [] for node_id in floating}
    for edge in global_edges:
        if edge.dst_id in floating:
            in_neighbors[edge.dst_id].append(edge.src_id)
        if edge.src_id in floating:
            out_neighbors[edge.src_id].append(edge.dst_id)

    # 收缩法（不动点）：若某游离节点存在一条边连到「既不在候选集合 T、又不属于任何容器」
    # 的外部节点，则该节点不可能整体归入容器，从 T 中删除；重复直到稳定。
    survivors = set(floating)
    changed = True
    while changed:
        changed = False
        for node_id in list(survivors):
            neighbors = in_neighbors[node_id] + out_neighbors[node_id]
            for neighbor_id in neighbors:
                if neighbor_id in survivors:
                    continue
                if _min_container(neighbor_id) is None:
                    survivors.discard(node_id)
                    changed = True
                    break
    if not survivors:
        return

    # 把 survivors 按内部边（T-T 边）划分为连通分量，整组归入同一容器。
    undirected: dict[int, set[int]] = {node_id: set() for node_id in survivors}
    for edge in global_edges:
        if edge.src_id in survivors and edge.dst_id in survivors:
            undirected[edge.src_id].add(edge.dst_id)
            undirected[edge.dst_id].add(edge.src_id)

    visited: set[int] = set()
    for seed in survivors:
        if seed in visited:
            continue
        component: set[int] = set()
        stack = [seed]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.add(cur)
            for nxt in undirected[cur]:
                if nxt not in visited:
                    stack.append(nxt)

        external_neighbors: set[int] = set()
        for node_id in component:
            for neighbor_id in in_neighbors[node_id] + out_neighbors[node_id]:
                if neighbor_id not in component:
                    external_neighbors.add(neighbor_id)
        if not external_neighbors:
            # 完全孤立的游离节点，没有归属目标，保持游离。
            continue

        targets = {_min_container(neighbor_id) for neighbor_id in external_neighbors}
        if None in targets:
            # 收缩法应已剔除此类节点；若仍出现说明状态不一致。
            raise RuntimeError(
                f"mode1: component {sorted(component)} has external neighbor outside any container"
            )
        if len(targets) != 1:
            # 多容器竞争：边跨越互不嵌套的容器，不归入任何容器。
            continue
        target_id = targets.pop()
        _absorb_into_container(
            dag=dag,
            registry=registry,
            component=component,
            target_id=target_id,
            target_descendants=desc[target_id],
            global_edges=global_edges,
        )


def _absorb_into_container(
    dag: DAG,
    registry: dict[int, DagNode],
    component: set[int],
    target_id: int,
    target_descendants: set[int],
    global_edges: list[DataFlowEdge],
) -> None:
    target_node = registry[target_id]
    inner_dag = target_node.inner_dag
    if inner_dag is None:
        raise RuntimeError(f"mode1: target container {target_id} has no inner_dag")

    for node_id in sorted(component):
        if node_id not in dag.direct_nodes:
            raise RuntimeError(
                f"mode1: absorbed node {node_id} not present in scope direct_nodes"
            )
        dag.direct_nodes.remove(node_id)
        if node_id not in inner_dag.nodes:
            inner_dag.nodes.append(node_id)
        if node_id not in inner_dag.direct_nodes:
            inner_dag.direct_nodes.append(node_id)

    # 记录到容器 metadata，使原生容器重建逻辑（_normalize_single_dag）保留这些节点。
    absorbed = target_node.metadata.setdefault("mode1_absorbed", [])
    for node_id in sorted(component):
        if node_id not in absorbed:
            absorbed.append(node_id)

    member_set = set(target_descendants) | set(component)
    existing_edges = {(edge.src_id, edge.dst_id) for edge in inner_dag.edges}
    for edge in global_edges:
        if edge.src_id in member_set and edge.dst_id in member_set:
            if (edge.src_id, edge.dst_id) not in existing_edges:
                inner_dag.edges.append(edge)
                existing_edges.add((edge.src_id, edge.dst_id))
    _rebuild_adjacency(inner_dag)


def _split_into_connected_components(
    member_ids: list[int],
    dag_edges,
) -> list[list[int]]:
    if not member_ids:
        raise RuntimeError("member_ids must not be empty")
    if len(set(member_ids)) != len(member_ids):
        raise RuntimeError(f"member_ids contains duplicates: {member_ids}")

    member_id_set = set(member_ids)
    adjacency: dict[int, list[int]] = {member_id: [] for member_id in member_ids}
    for edge in dag_edges:
        if edge.src_id is None or edge.dst_id is None:
            raise RuntimeError(f"edge endpoint must not be None: {edge}")
        if edge.src_id not in member_id_set or edge.dst_id not in member_id_set:
            continue
        adjacency[edge.src_id].append(edge.dst_id)
        adjacency[edge.dst_id].append(edge.src_id)

    visited: set[int] = set()
    components: list[list[int]] = []
    for seed_id in member_ids:
        if seed_id in visited:
            continue
        queue = [seed_id]
        visited.add(seed_id)
        component_set = {seed_id}
        while queue:
            current_id = queue.pop(0)
            for next_id in adjacency[current_id]:
                if next_id in visited:
                    continue
                visited.add(next_id)
                component_set.add(next_id)
                queue.append(next_id)
        components.append([member_id for member_id in member_ids if member_id in component_set])
    return components


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
        components = _split_into_connected_components(member_ids, dag.edges)
        for idx, component_member_ids in enumerate(components):
            group_name = helper_name if idx == 0 else f"{helper_name}#{idx}"
            new_node = _build_function_group_node(
                dag=dag,
                registry=registry,
                member_ids=component_member_ids,
                helper_name=helper_name,
                group_name=group_name,
            )
            dag.nodes.append(new_node.node_id)
            dag.direct_nodes = _replace_direct_nodes_with_group(
                dag.direct_nodes,
                member_ids=set(component_member_ids),
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
    group_name: str | None = None,
) -> ModuleNode:
    if group_name is None:
        group_name = helper_name
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
            attr_name=group_name,
            class_name=group_name,
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
    pattern_registry: dict[tuple[tuple[str, ...], tuple[tuple[int, int], ...]], str],
    pattern_counter: dict[str, int],
) -> None:
    del parent_func
    candidate_ids = [
        node_id
        for node_id in dag.direct_nodes
        if (
            isinstance(registry[node_id].attr, FunctionalAttr)
            or (isinstance(registry[node_id], ModuleNode) and registry[node_id].is_native is True)
            or registry[node_id].metadata.get("synthetic_type") == "function_group"
        )
    ]
    if len(candidate_ids) < 2:
        return

    components = _split_into_connected_components(candidate_ids, dag.edges)
    component_member_lists = [component for component in components if len(component) >= 2]
    if not component_member_lists:
        return

    original_direct_node_count = len(dag.direct_nodes)

    for component_member_ids in component_member_lists:
        if len(component_member_ids) == original_direct_node_count:
            continue

        component_member_id_set = set(component_member_ids)
        indegree = {node_id: 0 for node_id in component_member_ids}
        outgoing: dict[int, list[int]] = {node_id: [] for node_id in component_member_ids}
        internal_edges: list[DataFlowEdge] = []
        for edge in dag.edges:
            if edge.src_id not in component_member_id_set or edge.dst_id not in component_member_id_set:
                continue
            outgoing[edge.src_id].append(edge.dst_id)
            indegree[edge.dst_id] += 1
            internal_edges.append(edge)

        queue = [node_id for node_id in component_member_ids if indegree[node_id] == 0]
        topo_order: list[int] = []
        while queue:
            node_id = queue.pop(0)
            topo_order.append(node_id)
            for dst_id in outgoing[node_id]:
                indegree[dst_id] -= 1
                if indegree[dst_id] == 0:
                    queue.append(dst_id)
        if len(topo_order) != len(component_member_ids):
            raise RuntimeError(f"B-group 内发现环: component={component_member_ids}")

        topo_index = {node_id: idx for idx, node_id in enumerate(topo_order)}
        class_name_tuple = tuple(registry[node_id].attr.class_name for node_id in topo_order)
        edge_index_tuple = tuple(
            sorted((topo_index[edge.src_id], topo_index[edge.dst_id]) for edge in internal_edges)
        )
        pattern_key = (class_name_tuple, edge_index_tuple)
        if pattern_key not in pattern_registry:
            pattern_registry[pattern_key] = chr(ord("A") + len(pattern_registry))
        pattern_letter = pattern_registry[pattern_key]
        pattern_index = pattern_counter.get(pattern_letter, 0)
        group_name = f"Pattern{pattern_letter}" if pattern_index == 0 else f"Pattern{pattern_letter}#{pattern_index}"
        pattern_counter[pattern_letter] = pattern_index + 1

        group_node = _build_b_group_node(group_name, topo_order, dag, registry)
        dag.nodes.append(group_node.node_id)
        dag.direct_nodes = _replace_direct_nodes_with_group(
            dag.direct_nodes,
            member_ids=component_member_id_set,
            group_node_id=group_node.node_id,
        )
        registry[group_node.node_id] = group_node



def _build_b_group_node(
    group_name: str,
    member_ids: list[int],
    dag: DAG,
    registry: dict[int, DagNode],
) -> ModuleNode:
    representative_node = registry[member_ids[0]]
    member_id_set = set(member_ids)
    return ModuleNode(
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
            edges=[edge for edge in dag.edges if edge.src_id in member_id_set and edge.dst_id in member_id_set],
            direct_nodes=member_ids.copy(),
        ),
        is_native=False,
    )


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


__all__ = ["normalize_containers_recursive", "apply_container_expansion_mode1"]
