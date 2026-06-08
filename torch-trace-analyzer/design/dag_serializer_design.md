# DAG 序列化设计（dag_serializer.py）

## 1. 职责与边界
- 输入：`DAG` + `registry: dict[int, DagNode]`（经过 `normalize_containers_recursive` 后处理）
- 输出：可直接 JSON 序列化的 Python `dict`（对应前端 DATA schema）
- 不做：任何 trace 数据计算、分类、聚合；只做结构映射

补充约束：
- 序列化模块只消费已经归一化后的 IR，不负责补边、改写 boundary edge、推断容器成员关系。
- `dag.edges` 要完整输出，包括 `is_containment=True` 的 containment 边；前端按需过滤。
- 所有输出字段都来自现有 IR 字段或稳定映射规则，不引入兼容层、别名字段或静默兜底路径。

## 2. 输出 Schema

### 顶层 SerializedDAG
```python
{
  "input_nodes":  list[IONode],    # ForwardArgAttr（dag.inputs）
  "param_nodes":  list[IONode],    # ParamAttr（dag.params）
  "const_nodes":  list[IONode],    # ConstantAttr（dag.consts）
  "output_nodes": list[IONode],    # ResultNode（dag.outputs）
  "nodes":        list[Node | GroupNode],  # dag.nodes（ModuleNode）
  "edges":        list[Edge]       # dag.edges 全部，含 is_containment=True 的边
}
```

顶层字段来源：
- `input_nodes`：按 `dag.inputs` 顺序遍历，对应节点必须是 `InputNode` 且 `attr` 为 `ForwardArgAttr`
- `param_nodes`：按 `dag.params` 顺序遍历，对应节点必须是 `InputNode` 且 `attr` 为 `ParamAttr`
- `const_nodes`：按 `dag.consts` 顺序遍历，对应节点必须是 `InputNode` 且 `attr` 为 `ConstantAttr`
- `output_nodes`：按 `dag.outputs` 顺序遍历，对应节点必须是 `ResultNode`
- `nodes`：按 `dag.nodes` 顺序遍历，对应节点必须是 `ModuleNode`
- `edges`：按 `dag.edges` 顺序遍历，不做过滤

若节点类型与所在列表语义不匹配，应直接 `raise RuntimeError`，不允许静默跳过。

### IONode
```python
{
  "node_id":    int,
  "io_subtype": "input" | "param" | "const" | "output",
  "label":      str,   # 见字段映射规则
  "call_loc":   {"file": str, "line": int, "col": int} | null
}
```

label 映射规则：
- `ForwardArgAttr` → `f"arg_{attr.arg_index}"`
- `ParamAttr` → `attr.param_name`（为空则 `f"param_{node_id}"`）
- `ConstantAttr` → `f"{attr.attr_name}({attr.op_name})"`（`attr_name` 为空则只用 `op_name`）
- `ResultAttr` → `attr.head_name`（为空则 `f"output_{node_id}"`）

call_loc 映射规则：
- `node.call_loc is None` → `null`
- 否则输出 `{"file": loc.file, "line": loc.line, "col": loc.col}`

类型约束：
- `io_subtype="input"` 仅接受 `ForwardArgAttr`
- `io_subtype="param"` 仅接受 `ParamAttr`
- `io_subtype="const"` 仅接受 `ConstantAttr`
- `io_subtype="output"` 仅接受 `ResultAttr`

### Node（普通 ModuleNode，metadata.get("is_container") 不为 True）
```python
{
  "node_id":   int,
  "label":     str,   # f"{attr.attr_name}({attr.class_name})"，见规则
  "call_loc":  {"file": str, "line": int, "col": int} | null,
  "attr_type": "module" | "functional",   # ModuleAttr → "module"，FunctionalAttr → "functional"
  "inner_dag": SerializedDAG | null       # 递归序列化 ModuleNode.inner_dag
}
```

label 规则：
- `attr.attr_name` 和 `attr.class_name` 都非空：`f"{attr.attr_name}({attr.class_name})"`
- `attr.attr_name` 为空、`attr.class_name` 非空：只输出 `attr.class_name`
- `attr.class_name` 为空、`attr.attr_name` 非空：只输出 `attr.attr_name`
- 二者都为空：直接 `raise RuntimeError`

attr_type 规则：
- `ModuleAttr` → `"module"`
- `FunctionalAttr` → `"functional"`
- 其他 attr 类型进入普通 `ModuleNode` 路径时直接 `raise RuntimeError`

### GroupNode（ModuleNode，metadata["is_container"]=True）
```python
{
  "node_id":        int,
  "label":          str,    # f"{attr.attr_name}({container_kind})"
  "call_loc":       {"file": str, "line": int, "col": int} | null,
  "is_container":   true,
  "container_kind": str,    # ContainerAttr.container_kind
  "children_nodes": list[int]  # containment 边（is_containment=True）中 src_id==此节点的 dst_id 列表
}
```

注：`GroupNode` 没有 `inner_dag`，也没有 `internal_edges` 字段。containment 边已在顶层 `edges` 里，前端按需过滤。

字段规则：
- 仅当 `node.metadata.get("is_container") is True` 时进入该分支
- `node.attr` 必须是 `ContainerAttr`，否则直接 `raise RuntimeError`
- `label` 规则：
  - `attr.attr_name` 非空：`f"{attr.attr_name}({attr.container_kind})"`
  - `attr.attr_name` 为空：只输出 `attr.container_kind`
- `children_nodes` 通过扫描当前 `dag.edges` 中满足 `edge.is_containment is True and edge.src_id == node.node_id` 的边得到，保持 `dag.edges` 原始顺序

### Edge
```python
{
  "src_id":         int,
  "dst_id":         int,
  "is_containment": bool,
  "tensor_info":    {str(tensor_id): {"shape": list[int], "dtype": str}},  # 空则 {}
  "evidence":       list[EvidenceStep] | null
    # null 表示 add_edge() 时 evidence=[]（未填），不是"应该是 None"
    # 非 null 时每条：{"loc": {"file":str,"line":int,"col":int}, "role": str, "var": str}
}
```

evidence 序列化规则：
- `edge.evidence == []` → 序列化为 `null`（如实反映“未填”，不是默认值）
- `edge.evidence` 非空 → 依次展开每条 `VarEvidence.steps`
- 输出顺序保持原始 `edge.evidence` 顺序，再保持每条 `VarEvidence.steps` 内部顺序
- 第一期不额外输出 `path_id`、`VarEvidence.var` 等上层容器信息；只输出前端当前需要消费的 `EvidenceStep`

tensor_info 序列化规则：
- 键：`int tensor_id` 转为 `str(tensor_id)`
- 值：保留 `shape` / `dtype` 两个字段
- `edge.tensor_info == {}` → 输出空对象 `{}`，不省略字段

## 3. 序列化入口函数签名

```python
def serialize_dag(dag: DAG, registry: dict[int, DagNode]) -> dict:
    """
    序列化 DAG 为前端 DATA schema。
    前置条件：dag 已经过 normalize_containers_recursive 后处理。
    返回值可直接 json.dumps()。
    """
```

内部拆分：
- `_serialize_io_node(node: DagNode) -> dict`
- `_serialize_module_node(node: ModuleNode, dag: DAG, registry: dict) -> dict`（返回 Node 或 GroupNode）
- `_serialize_edge(edge: DataFlowEdge) -> dict`
- `_serialize_call_loc(loc: CallLoc | None) -> dict | None`

建议实现流程：
1. 遍历 `dag.inputs`，仅序列化 `ForwardArgAttr` 节点到 `input_nodes`
2. 遍历 `dag.params`，序列化 `ParamAttr` 节点到 `param_nodes`
3. 遍历 `dag.consts`，序列化 `ConstantAttr` 节点到 `const_nodes`
4. 遍历 `dag.outputs`，序列化 `ResultNode` 到 `output_nodes`
5. 遍历 `dag.nodes`，根据 `metadata["is_container"]` 选择普通 `Node` 或 `GroupNode`
6. 遍历 `dag.edges`，统一序列化为 `Edge`
7. 返回顶层 `dict`

校验原则：
- 如果 `dag` 中的 node id 不在 `registry`，直接 `raise RuntimeError`
- 如果 `dag.nodes` 中出现非 `ModuleNode`，直接 `raise RuntimeError`
- 如果 `GroupNode` 同时还挂了 `inner_dag`，序列化阶段不消费该 `inner_dag`，只输出 group 结构；这不是兜底，而是明确的 schema 边界

## 4. 递归终止条件
- `ModuleNode.inner_dag is None` → leaf node，`inner_dag` 序列化为 `null`
- `GroupNode` 不递归（没有 `inner_dag`）

递归策略说明：
- 普通 `ModuleNode` 若 `inner_dag` 非空，则对该 `inner_dag` 再次调用 `serialize_dag()`，产出嵌套 `SerializedDAG`
- containment 关系不通过 `inner_dag` 表达，而是通过顶层 `edges` + `GroupNode.children_nodes` 表达
- 因此递归只发生在普通 `ModuleNode.inner_dag` 上，不发生在容器节点上

## 5. 已知限制（第一期）
- evidence 第一期 `add_edge()` 均为 `[]`，序列化后全为 `null`；evidence 打通后自动生效，无需修改序列化逻辑
- `tensor_info` 为 `{}` 时序列化为空对象，不省略字段
- 当前 schema 不输出 `VarEvidence.path_id`，也不区分不同 `VarEvidence` 分组；前端若后续需要更高保真 evidence，再单独扩 schema
- 当前 schema 不为 `GroupNode` 输出 `inner_dag` / `internal_edges` 等旧版 `dag_to_html` 字段，避免与已确认的新容器方案冲突
