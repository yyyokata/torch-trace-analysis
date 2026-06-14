# DAG → HTML 可视化管线设计

## 一、动态 IR 数据结构总览

### dag_types.py

```
DAG
  inputs: list[int]          # InputNode / param / const 的 node_id 列表
  outputs: list[int]         # ResultNode 的 node_id 列表
  nodes: list[int]           # ModuleNode 的 node_id 列表（不含 inputs/outputs）
  edges: list[DataFlowEdge]  # 所有数据流边
  params: list               # ParamAttr 节点 id 列表
  consts: list               # ConstantAttr 节点 id 列表
  in_edges / out_edges       # dict，__post_init__ 建立

DataFlowEdge
  src_id: int
  dst_id: int
  is_containment: bool       # containment 边（parent-child 关系，非数据流）
  evidence: list[VarEvidence]
  tensor_info: dict[int, dict]  # {tensor_id: {shape, dtype}}

DagNode (base)
  node_id: int
  call_loc: CallLoc          # (file, line, col)
  attr: Attr
  metadata: dict

InputNode(DagNode)    # attr: InputAttr / ForwardArgAttr
ResultNode(DagNode)   # attr: ResultAttr（head_name / classifier_type）
ModuleNode(DagNode)   # attr: ModuleAttr / FunctionalAttr，inner_dag: DAG | None
```

### attr_types.py

```
Attr.attr_name / class_name / def_loc / attr_id
ModuleAttr.is_native
InputAttr.kind
ResultAttr.head_name / classifier_type
FunctionalAttr.is_native
ConstantAttr.op_name
ParamAttr.param_name
```

dag_session_types.py 中的 _TensorSlot / _PendingCall / _ResultHeadRecord / FunctionalRecord 均为构建中间态，不进最终 DAG 对象。

---

## 二、前端 DATA schema 字段全表

### flowchart_data 顶层字段

| 字段 | 类型 | 必填 | JS 消费 |
|------|------|------|---------|
| nodes | list[dict] | ✅ | nodeMap[n.id] = n |
| groups | list[dict] | ✅ | groupMap[g.id] = g |
| edges | list[dict] | ✅ | 顶层边 |
| root_groups | list[int] | ✅ | DATA.root_groups.map(rid => groupMap[rid]) |
| top_level_ids | list | ✅ | 顶层布局顺序 |
| input_node_id | int | ✅ | isIONode() 判断 |
| loss_node_id | int | ✅ | isIONode() 判断 |
| result_node_id | int | ✅ | 同 loss_node_id |
| has_timing | bool | ✅ | timing bar 开关 |
| has_timing_data | bool | ✅ | 同上 |
| has_trace_step | bool | 否 | 传 False |
| meta | dict | ✅ | 顶部 metadata 展示 |

### node 对象字段

| 字段 | 必填 | 动态线填值 |
|------|------|-----------|
| id | ✅ | node.node_id |
| attr_name | ✅ | node.attr.attr_name |
| class_name | ✅ | node.attr.class_name |
| depth | ✅ | 递归深度 |
| pct / exc_pct | ✅ | 0.0 |
| kernel_us 等 timing 字段 | ✅ | 0.0 |
| dur_us | ✅ | 0.0 |
| has_timing | ✅ | False |
| has_phase_timing | ✅ | False |
| calls | ✅ | 0 |
| role | ✅ | "main" |
| kind | 否 | "io" 仅 Input/Result/param/const |
| src_file | 否 | node.attr.def_loc.file |
| src_start_line / src_end_line | 否 | node.attr.def_loc.line |
| src_snippet 等 | 否 | None（第一期）|

### group 对象字段

| 字段 | 必填 | 动态线填值 |
|------|------|-----------|
| id | ✅ | node.node_id |
| label | ✅ | node.attr.class_name |
| attr_name | ✅ | node.attr.attr_name |
| depth | ✅ | 递归深度 |
| children_nodes | ✅ | inner_dag.nodes 里无 inner_dag 的 id |
| children_groups | ✅ | inner_dag.nodes 里有 inner_dag 的递归结果 |
| call_order | ✅ | 按 inner_dag.nodes 顺序 |
| internal_edges | ✅ | inner_dag 内非 containment 边 |
| timing 字段 | ✅ | 0.0 |
| has_timing / has_phase_timing | ✅ | False |

### edge / internal_edge 对象字段

顶层 dag_edge：from / to / type / from_attr / to_attr / evidence

group 内 internal_edge：from_child / to_child / type / from_attr / to_attr / parent_class / evidence

---

## 三、动态 IR → 前端 schema 映射表

### InputNode → 合成 Input 顶层节点（kind="io"）
- dag.inputs 中所有普通 InputNode（InputAttr / ForwardArgAttr）合并为一个合成节点
- attr_name="Input", class_name="Input", kind="io"
- 这是第一个顶层节点，id 为合成新 id

### ResultNode → 合成 Result 顶层节点（kind="io"）
- dag.outputs → attr_name="Result", class_name="LG.result()", kind="io"
- loss_node_id = result_node_id = 合成新 id

### ModuleNode（inner_dag is None）→ leaf node
- 直接填 id / attr_name / class_name，depth 由递归层

### ModuleNode（inner_dag is not None）→ group
- 递归序列化 inner_dag.nodes
- children_nodes / children_groups / call_order / internal_edges 从 inner_dag 生成

### DataFlowEdge（is_containment=False）→ edge
- src/dst 均为同一 group 直接子节点 → internal_edge（from_child/to_child）
- 否则 → 顶层 dag_edge（from/to）
- is_containment=True：不进 edge 列表

---

## 四、缺口字段一览

| 字段 | 动态线填法 |
|------|-----------|
| pct / exc_pct | 0.0 |
| kernel_us 等 | 0.0 |
| has_timing | False |
| calls | 0 |
| role | "main" |
| evidence | 第一期 None（type="boundary"），后续见第三节 |
| src_snippet | None |
| ctor_src_* | 不填 |
| container_kind / is_container | 不填 |

---

## 五、顶层拓扑

```
top_level_ids = [input_node_id, root_group_id, loss_node_id]
root_groups = [root_group_id]
dag_edges = [
  {from: input_node_id, to: root_group_id, type: "boundary", ...},
  {from: root_group_id, to: loss_node_id,  type: "boundary", ...},
]
```

root_group_id 对应 registry 里 is_root=True 的 ModuleNode。

---

## 六、序列化入口

文件：torch-trace-analyzer/scripts/dag_to_html.py

```python
def dag_to_flowchart_data(dag: DAG, registry: dict[int, DagNode], meta: dict | None) -> dict: ...
def _serialize_node(node_id, registry, depth) -> (dict, str): ...  # kind: "node" | "group"
def _serialize_dag_level(dag, registry, depth) -> (...): ...
```

调用链：run_dag_session.py finalize → dag_to_flowchart_data → _generate_flowchart_html → 写 HTML 文件
