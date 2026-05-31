"""
DAG IR 数据结构定义。

包含：
  EvidenceStep / VarEvidence  —— 边上的数据流追踪链
  DagNode / InputNode / ResultNode / ModuleNode  —— 节点层级
  DataFlowEdge                —— 数据流边（含 is_containment 归属虚边）
  DAG                         —— 单层 DAG（nodes/edges/邻接表）
  DagContext                  —— 顶层容器（registry + loc_attr_index）
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ast_constants import ConstantTable

from attr_types import (
    CallLoc,
    Attr,
)


# ── Evidence ──────────────────────────────────────────────────────────────────

@dataclass
class EvidenceStep:
    """单步追踪记录。role: 'producer' / 'step' / 'consumer'"""
    loc  : CallLoc
    role : str
    var  : str


@dataclass
class VarEvidence:
    """src 输出视角的单条追踪链。(var, path_id) 联合唯一。"""
    var     : str
    path_id : int
    steps   : list  # list[EvidenceStep]


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class DagNode:
    """DAG 节点基类。"""
    node_id  : int
    call_loc : CallLoc
    attr     : Attr
    metadata : dict = field(default_factory=dict)


@dataclass
class InputNode(DagNode):
    """Root DAG 输入节点（InputAttr）或 inner_dag 形参节点（ForwardArgAttr）。"""
    pass


@dataclass
class ResultNode(DagNode):
    """Root DAG 结果节点（ResultAttr）或 inner_dag 返回值节点（ReturnValAttr）。"""
    pass


@dataclass
class ModuleNode(DagNode):
    """模块节点，可包含 inner_dag。"""
    inner_dag    : Optional["DAG"] = None
    arg_bindings : dict = field(default_factory=dict)
    # key: 外部实参 node_id(int) → value: inner_dag InputNode node_id(int)
    # ret_bindings 不需要：外层 edge src=ModuleNode，ret 靠 ReturnValAttr.ret_index 区分


# ── Edge ──────────────────────────────────────────────────────────────────────

@dataclass
class DataFlowEdge:
    """数据流边。is_containment=True 为归属虚边（evidence 为空列表）。"""
    src_id         : int
    dst_id         : int
    is_containment : bool
    evidence       : list  # list[VarEvidence]


# ── DAG ───────────────────────────────────────────────────────────────────────

@dataclass
class DAG:
    """单层 DAG，全部存 node_id，不持有 DagNode 对象。"""
    inputs  : list   # list[int] InputNode id
    outputs : list   # list[int] ResultNode id
    nodes   : list   # list[int] ModuleNode id
    edges   : list   # list[DataFlowEdge]
    in_edges  : dict = field(default_factory=dict)  # dict[int, list[DataFlowEdge]] 派生
    out_edges : dict = field(default_factory=dict)  # dict[int, list[DataFlowEdge]] 派生

    def __post_init__(self):
        for e in self.edges:
            self.out_edges.setdefault(e.src_id, []).append(e)
            self.in_edges.setdefault(e.dst_id, []).append(e)

    def walk_bottom_up(self, registry: dict):
        """后序遍历（叶子优先）当前层及所有嵌套 inner_dag 的 ModuleNode node_id。"""
        for node_id in self.nodes:
            node = registry[node_id]
            if isinstance(node, ModuleNode) and node.inner_dag:
                yield from node.inner_dag.walk_bottom_up(registry)
            yield node_id


# ── DagContext ─────────────────────────────────────────────────────────────────

@dataclass
class DagContext:
    """顶层容器：全量节点注册表 + 根 DAG + loc_attr 索引。"""
    registry       : dict   # dict[int, DagNode]
    root           : DAG
    loc_attr_index : dict = field(default_factory=dict)
    const_table    : Optional["ConstantTable"] = None
    attr_registry  : dict = field(default_factory=dict)
    # key: (file: str, line: int, attr_id: int) → node_id: int
    # col 不参与 key（timeline event 不携带 col）

    def __post_init__(self):
        for node_id, node in self.registry.items():
            key = (node.call_loc.file, node.call_loc.line, node.attr.attr_id)
            self.loc_attr_index[key] = node_id
        self.callsite_index: dict = {}
        for (file, line, _attr_id), node_id in self.loc_attr_index.items():
            self.callsite_index[(os.path.basename(file), line)] = node_id
