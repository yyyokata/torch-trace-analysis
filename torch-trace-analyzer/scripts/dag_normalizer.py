from __future__ import annotations

from attr_types import CallLoc, ContainerAttr, _NATIVE_CONTAINER_KINDS
from dag_types import DAG, DagNode, DataFlowEdge, ModuleNode


def normalize_containers_recursive(dag: DAG, registry: dict[int, DagNode]) -> None:
    _normalize_containers_recursive(
        dag,
        registry,
        owner_node_id=None,
        visited_dag_ids=set(),
    )


def _normalize_containers_recursive(
    dag: DAG,
    registry: dict[int, DagNode],
    owner_node_id: int | None,
    visited_dag_ids: set[int],
) -> None:
    dag_object_id = id(dag)
    if dag_object_id in visited_dag_ids:
        return
    visited_dag_ids.add(dag_object_id)
    attr_to_node_ids = _build_attr_to_node_ids(registry)
    _ensure_missing_container_nodes(dag, registry, attr_to_node_ids)
    attr_to_node_ids = _build_attr_to_node_ids(registry)
    _normalize_single_dag(dag, registry, attr_to_node_ids, owner_node_id=owner_node_id)
    _rebuild_adjacency(dag)
    for node_id in list(dag.nodes):
        if owner_node_id is not None and node_id == owner_node_id:
            continue
        node = registry[node_id]
        if isinstance(node, ModuleNode) and node.inner_dag is not None:
            _normalize_containers_recursive(
                node.inner_dag,
                registry,
                owner_node_id=node_id,
                visited_dag_ids=visited_dag_ids,
            )


def _normalize_single_dag(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
    owner_node_id: int | None,
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
        for member_id in sorted(member_ids):
            if member_id not in inner_dag.direct_nodes:
                inner_dag.direct_nodes.append(member_id)


def _ensure_missing_container_nodes(
    dag: DAG,
    registry: dict[int, DagNode],
    attr_to_node_ids: dict[int, list[int]],
) -> None:
    attr_to_node_ids = _build_attr_to_node_ids(registry)
    missing_container_attrs: list[ContainerAttr] = []
    seen_attr_ids: set[int] = set()
    for node_id in list(dag.nodes):
        node = registry[node_id]
        parent_attr = node.attr.parent
        while parent_attr is not None:
            if parent_attr.class_name not in _NATIVE_CONTAINER_KINDS:
                raise RuntimeError(
                    f"Node {node_id} belongs to unsupported container class {parent_attr.class_name}"
                )
            parent_attr_id = id(parent_attr)
            if parent_attr_id not in attr_to_node_ids and parent_attr_id not in seen_attr_ids:
                seen_attr_ids.add(parent_attr_id)
                missing_container_attrs.append(parent_attr)
            parent_attr = parent_attr.parent

    for container_attr in missing_container_attrs:
        if id(container_attr) in attr_to_node_ids:
            continue
        node_id = max(registry, default=0) + 1
        child_ids: list[int] = []
        for child_attr in container_attr.items.values():
            matched_node_ids = attr_to_node_ids.get(id(child_attr))
            if not matched_node_ids:
                # 该子节点已被 DCE 剪除（registry 中无对应节点），跳过。
                continue
            if len(matched_node_ids) != 1:
                raise RuntimeError(
                    f"Container attr {container_attr.attr_name} child attr {child_attr.attr_name} expected exactly one DagNode, got {matched_node_ids}"
                )
            child_ids.append(matched_node_ids[0])
        child_id_set = set(child_ids)
        inner_edges = [edge for edge in dag.edges if edge.src_id in child_id_set and edge.dst_id in child_id_set]
        registry[node_id] = ModuleNode(
            node_id=node_id,
            call_loc=container_attr.def_loc or CallLoc(file="<container>", line=0, col=0),
            attr=container_attr,
            metadata={
                "module_path": _container_attr_path(container_attr),
                "is_container": True,
            },
            inner_dag=DAG(
                inputs=[],
                outputs=[],
                nodes=child_ids.copy(),
                edges=inner_edges,
                direct_nodes=child_ids.copy(),
            ),
            is_native=True,
        )
        attr_to_node_ids.setdefault(id(container_attr), []).append(node_id)


def _container_attr_path(attr: ContainerAttr) -> str:
    parts: list[str] = []
    current = attr
    while current is not None:
        if current.attr_name:
            parts.append(current.attr_name)
        parent = current.parent
        current = parent if isinstance(parent, ContainerAttr) else None
    return ".".join(reversed(parts))


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


def _is_native_container_node(node: DagNode) -> bool:
    return (
        isinstance(node, ModuleNode)
        and isinstance(node.attr, ContainerAttr)
        and node.attr.class_name in _NATIVE_CONTAINER_KINDS
    )


def _rebuild_adjacency(dag: DAG) -> None:
    in_edges: dict[int, list[DataFlowEdge]] = {}
    out_edges: dict[int, list[DataFlowEdge]] = {}
    for edge in dag.edges:
        out_edges.setdefault(edge.src_id, []).append(edge)
        in_edges.setdefault(edge.dst_id, []).append(edge)
    dag.in_edges = in_edges
    dag.out_edges = out_edges


__all__ = ["normalize_containers_recursive"]
