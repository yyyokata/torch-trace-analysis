from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from attr_types import Attr, CallLoc


@dataclass
class EvidenceStep:
    loc: CallLoc
    role: str
    var: str


@dataclass
class VarEvidence:
    var: str
    path_id: int
    steps: list[EvidenceStep] = field(default_factory=list)


@dataclass
class DataFlowEdge:
    src_id: int
    dst_id: int
    is_containment: bool
    evidence: list[VarEvidence] = field(default_factory=list)
    tensor_info: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DagNode:
    node_id: int
    call_loc: CallLoc
    attr: Attr
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputNode(DagNode):
    pass


@dataclass
class ResultNode(DagNode):
    pass


@dataclass
class ModuleNode(DagNode):
    inner_dag: "DAG | None" = None
    arg_bindings: dict[int, int] = field(default_factory=dict)


@dataclass
class DAG:
    inputs: list[int]
    outputs: list[int]
    nodes: list[int]
    edges: list[DataFlowEdge]
    params: list = field(default_factory=list)
    consts: list = field(default_factory=list)
    in_edges: dict[int, list[DataFlowEdge]] = field(init=False, default_factory=dict)
    out_edges: dict[int, list[DataFlowEdge]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        in_edges: dict[int, list[DataFlowEdge]] = {}
        out_edges: dict[int, list[DataFlowEdge]] = {}
        for e in self.edges:
            out_edges.setdefault(e.src_id, []).append(e)
            in_edges.setdefault(e.dst_id, []).append(e)
        self.in_edges = in_edges
        self.out_edges = out_edges

    def walk_bottom_up(self, registry: dict[int, DagNode]) -> Iterator[int]:
        """Yield node ids in bottom-up order (post-order over nested dags)."""

        def walk(node_id: int) -> Iterator[int]:
            node = registry[node_id]
            if isinstance(node, ModuleNode) and node.inner_dag is not None:
                for child_id in node.inner_dag.nodes:
                    yield from walk(child_id)
                for child_id in node.inner_dag.outputs:
                    if child_id in registry:
                        yield from walk(child_id)
            yield node_id

        excluded = set(self.inputs) | set(self.outputs)
        for nid in self.nodes:
            if nid in excluded:
                continue
            yield from walk(nid)


__all__ = [
    "CallLoc",
    "Attr",
    "EvidenceStep",
    "VarEvidence",
    "DataFlowEdge",
    "DagNode",
    "InputNode",
    "ResultNode",
    "ModuleNode",
    "DAG",
]
