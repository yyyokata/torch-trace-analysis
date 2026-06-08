# DAG 序列化 UT 设计

## 测试文件位置
`testset/unit/test_dag_serializer.py`（在 storage 仓库）

## 测试策略
纯单元测试：手动构造 `DAG + registry`，调用 `serialize_dag()`，断言输出 `dict` 结构。不依赖真实模型执行。

约束：
- 所有测例都直接构造最小 IR，不经过真实 tracer / mock run。
- 每个测例只覆盖一个主决策点，避免多个失败原因耦合。
- 断言以结构和字段值为主，不依赖 JSON 字符串文本比较。
- `normalize_containers_recursive` 不在本 UT 内重复验证；涉及容器的测例直接构造“已归一化后”的 DAG 形态。

## 测例列表（S1-S8）

### S1 test_serialize_io_nodes_three_subtypes
- mock：一个 DAG，`inputs` 包含 `ForwardArgAttr` 节点，`params` 包含 `ParamAttr` 节点，`consts` 包含 `ConstantAttr` 节点
- 断言：`input_nodes[0].io_subtype=="input"`；`param_nodes[0].io_subtype=="param"`；`const_nodes[0].io_subtype=="const"`；`label` 分别符合映射规则
- 额外断言：三个列表长度都为 1；`call_loc` 都按 `{file,line,col}` 结构输出

### S2 test_serialize_output_node
- mock：`outputs` 包含 `ResultAttr` 节点，`head_name="ctr_head"`
- 断言：`output_nodes[0].io_subtype=="output"`；`label=="ctr_head"`
- 额外断言：当 `call_loc` 存在时字段完整；`output_nodes` 长度为 1

### S3 test_serialize_plain_module_node_no_inner_dag
- mock：`dag.nodes` 包含普通 `ModuleNode`（`ModuleAttr`，`inner_dag=None`）
- 断言：`nodes[0].attr_type=="module"`；`inner_dag==null`；`label` 格式正确
- 额外断言：节点对象不包含 `is_container` 字段

### S4 test_serialize_functional_node
- mock：`ModuleNode`（`FunctionalAttr`）
- 断言：`nodes[0].attr_type=="functional"`
- 额外断言：`label` 仍遵循 `attr_name(class_name)` 规则

### S5 test_serialize_module_node_with_inner_dag
- mock：`ModuleNode.inner_dag` 是一个嵌套 DAG（含 1 个 `InputNode` + 1 个 `ModuleNode`）
- 断言：`nodes[0].inner_dag` 不为 `null`；`inner_dag.nodes` 长度正确；递归结构正确
- 额外断言：嵌套 DAG 内 `input_nodes`、`nodes`、`edges` 都存在且类型正确

### S6 test_serialize_group_node
- mock：`ModuleNode`（`metadata["is_container"]=True`，`ContainerAttr`，`container_kind="ModuleList"`）；`dag.edges` 包含 `is_containment=True` 的边指向子节点
- 断言：`nodes[0].is_container==true`；`container_kind=="ModuleList"`；`children_nodes` 正确；节点没有 `inner_dag` 字段
- 额外断言：`children_nodes` 顺序与 `dag.edges` 中 containment 边顺序一致

### S7 test_serialize_edge_with_empty_evidence
- mock：`DataFlowEdge`，`evidence=[]`，`tensor_info={123: {"shape":[2,3],"dtype":"float32"}}`
- 断言：`edge.evidence==null`（空列表序列化为 `null`）；`tensor_info` 正确
- 额外断言：`tensor_info` 的 key 从 `123` 转成字符串 `"123"`

### S8 test_serialize_edge_with_evidence
- mock：`DataFlowEdge`，`evidence=[VarEvidence(var="x", path_id=0, steps=[EvidenceStep(loc=CallLoc(...), role="src", var="x")])]`
- 断言：`edge.evidence` 不为 `null`；`evidence[0].role=="src"`；`loc` 字段正确
- 额外断言：只展开 `steps`，不输出 `path_id`

## 构造辅助约定
- 测试内可定义最小 helper：
  - `make_call_loc(file="model.py", line=1, col=0)`
  - `make_input_node(...) / make_result_node(...) / make_module_node(...)`
  - `make_edge(...)`
- helper 仅用于减少样板代码，不改变断言语义。
- `registry` 一律显式列出所有 node_id，避免依赖隐式共享状态。

## 覆盖目标
- IO 四种 subtype 全覆盖：input / param / const / output
- 普通 ModuleNode / FunctionalNode / GroupNode 三条节点分支全覆盖
- 递归 `inner_dag` 路径覆盖
- edge.evidence 的 `[] -> null` 与 `非空 -> step 列表` 两条路径全覆盖
- `tensor_info` 键类型转换覆盖

## 非本期覆盖范围
- 不在本 UT 中覆盖异常分支（如 registry 缺 node、attr 类型不匹配）；这些属于实现时可追加的 defensive case，但不在本轮用户确认范围内
- 不覆盖 `normalize_containers_recursive` 自身逻辑；该模块已有独立职责
- 不覆盖前端消费逻辑；这里只验证后端序列化输出 schema
