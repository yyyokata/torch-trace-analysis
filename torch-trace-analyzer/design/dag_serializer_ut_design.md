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

## 测例列表（S1-S14）

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
- mock：`ModuleNode.inner_dag` 是一个仅包含 leaf inner_dag 的嵌套 DAG（含 1 个 `InputNode` + 1 个 `ModuleNode`，无更深嵌套）
- 断言：`nodes[0].inner_dag` 不为 `null`；`inner_dag.nodes` 长度正确；递归结构正确
- 额外断言：嵌套 DAG 内 `input_nodes`、`nodes`、`edges` 都存在且类型正确；顶层 `edges` 不包含 inner_dag 内的边

### S6 test_serialize_group_node
- mock：`ModuleNode`（`metadata["is_container"]=True`，`ContainerAttr`，`container_kind="ModuleList"`）；`dag.edges` 包含 1 条 `is_containment=True` 的边指向唯一子节点
- 断言：`nodes[0].is_container==true`；`container_kind=="ModuleList"`；`children_nodes` 正确；节点没有 `inner_dag` 字段
- 额外断言：`children_nodes` 顺序与 `dag.edges` 中 containment 边顺序一致

### S7 test_serialize_edge_with_empty_evidence
- mock：`DataFlowEdge`，`evidence=[]`，`tensor_info={123: {"shape":[2,3],"dtype":"float32"}}`
- 断言：`edge.evidence==null`（空列表序列化为 `null`）；`tensor_info` 正确
- 额外断言：`tensor_info` 的 key 从 `123` 转成字符串 `"123"`

### S8 test_serialize_edge_with_evidence
- mock：`DataFlowEdge`，同一条 edge 上有 2 个 `VarEvidence`，每个 `VarEvidence.steps` 各含 1 个以上 `EvidenceStep`
- 断言：`edge.evidence` 不为 `null`；展开结果按 `edge.evidence` 顺序，再按每条 `steps` 内顺序合并；`loc` 字段正确
- 额外断言：只展开 `steps`，不输出 `path_id`

### S9 test_serialize_multiple_inputs_mixed_io
- mock：2 个 `ForwardArg`（`arg_0`/`arg_1`）、2 个 `ParamAttr`（`"w1"` + 空 `param_name`）、1 个 `ConstantAttr`（`attr_name` 为空，`op_name="zeros"`）、2 个 `ResultAttr`（`"out"` + 空 `head_name`）
- 断言：`input_nodes` 长度 2，label `"arg_0"`/`"arg_1"`；`param_nodes[1].label==f"param_{node_id}"`；`const_nodes[0].label=="zeros"`；`output_nodes[1].label==f"output_{node_id}"`；四类列表 `node_id` 互不重叠

### S10 test_serialize_two_level_inner_dag
- mock：顶层 `ModuleNode(Encoder, inner_dag=dag_encoder)`；`dag_encoder` 含 `ModuleNode(Layer, inner_dag=dag_layer)`；`dag_layer` 含 0 个 `ModuleNode` + 2 条 edge（1 普通 + 1 containment）
- 断言：`result["nodes"][0]["inner_dag"]` 非 null；内层 `["nodes"][0]["inner_dag"]` 非 null；`dag_layer` 层 `edges` 长度 2；三层 `edges` 互相独立

### S11 test_serialize_group_node_with_multiple_children
- mock：1 个 `GroupNode(ModuleList，3 子节点 id=[10,20,30])`；`dag.edges` 含 3 条 `is_containment=True(src=group_id, dst=10/20/30)` + 1 条普通数据流边
- 断言：`children_nodes==[10,20,30]`；`edges` 总长度 4；containment/非containment 各自 `is_containment` 值正确；`GroupNode` 无 `inner_dag` 字段

### S12 test_serialize_nested_container_inside_module
- mock：顶层 `ModuleNode(Transformer, inner_dag=dag_inner)`；`dag_inner` 含 1 个 `GroupNode(ModuleList，2 子节点)` + 对应 containment 边
- 断言：顶层 `nodes[0].inner_dag` 非 null；`inner_dag.nodes[0].is_container==true`；`children_nodes` 长度 2；`inner_dag.edges` 含 containment 边

### S13 test_serialize_edge_multiple_tensor_infos
- mock：`DataFlowEdge`，`tensor_info={100: {"shape":[4,8],"dtype":"float16"}, 200: {"shape":[1],"dtype":"int64"}}`，`evidence=[]`
- 断言：`output["100"]["shape"]==[4,8]`；`output["200"]["dtype"]=="int64"`；`len==2`；`evidence is None`；int key 100/200 不在输出

### S14 test_serialize_raises_on_node_not_in_registry
- mock：`dag.nodes` 含 `node_id=99` 的 `ModuleNode`，`registry` 不含 key 99
- 断言：`pytest.raises(RuntimeError)`

## 构造辅助约定
- 测试内可定义最小 helper：
  - `make_call_loc(file="model.py", line=1, col=0)`
  - `make_input_node(...) / make_result_node(...) / make_module_node(...) / make_group_node(...)`
  - `make_edge(...)`
- helper 仅用于减少样板代码，不改变断言语义。
- `registry` 一律显式列出所有 node_id，避免依赖隐式共享状态。

## 覆盖目标
- IO 四种 subtype 全覆盖：input / param / const / output
- 普通 `ModuleNode` / `FunctionalNode` / `GroupNode` 三条节点分支全覆盖
- 递归 `inner_dag` 路径覆盖（单层与两层嵌套）
- edge.evidence 的 `[] -> null` 与 `非空 -> step 列表` 两条路径全覆盖
- 同一条 edge 上多个 `VarEvidence` 的顺序展开覆盖
- `tensor_info` 键类型转换覆盖
- registry 缺失节点的异常路径覆盖

## 非本期覆盖范围
- 不覆盖 `normalize_containers_recursive` 自身逻辑；该模块已有独立职责
- 不覆盖前端消费逻辑；这里只验证后端序列化输出 schema
- 不覆盖设计文档未定义的派生字段或兼容层行为
