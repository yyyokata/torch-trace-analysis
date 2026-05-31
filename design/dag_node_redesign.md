# DAG 节点/边显式类型重构设计

## 0. 背景与目标

NaN 当前要求是：不要再让 `frontend_html.py` 从 `tree` 临时拼装 DAG，也不要继续保留 `_scan_root_result_edges()`、`_input_edge_var_history()` 这类“前端补后端洞”的扫描路径。新的方向是：

1. 在 `scripts/attr_types.py` 的 Attr 体系之上，新建独立文件 `scripts/dag_types.py`
2. 后端在 `tree` 构造完成后，**统一生成完整 `DagGraph`**
3. 前端只消费 `DagGraph`，不再自己扫 `tree`、不再自己补 result/input edge
4. **不留 fallback，不拷旧路径，不在前端做 trace 数据处理**
5. JS 层现有 `dag_nodes` / `dag_edges` 渲染协议保持不变，重构只发生在 Python 层

这份文档先梳理现状，再给出 `DagNode` / `DagEdge` / `DagGraph` 的完整设计和落地顺序。

---

## 1. 现状梳理

### 1.1 `attr_types.py` 当前 Attr 体系

`attr_types.py` 目前已经定义了后续 DAG typed graph 可以直接复用的基础对象：

- `CallLoc(file, line, col)`
- `Attr(attr_name, class_name, call_loc, parent, source_expr)`
- `ModuleAttr`
- `ContainerAttr(container_kind, items)`
- `InputAttr(forward_use_loc, kind/lg_source_kind, owner_expr, slot_expr)`
- `ResultAttr(head_name, classifier_type, prediction_expr, label_expr, sample_rate_expr, loss_expr)`

关键信号：

1. **Attr 已经是“源码语义对象”**，不是前端 JSON
2. `InputAttr` / `ResultAttr` 已经把 Input / Result 相关语义放在后端类型里了
3. 现在真正缺的是一个**显式 DAG 图类型层**，把 Attr 组织成 graph，而不是继续依赖 `tree + frontend 拼装`

### 1.2 `analyze_trace.py` 当前 `tree[cname]` 结构

`build_static_module_tree()` 在组装完 `class_attrs` 和 sibling data dependency 后，当前会为每个 `cname` 写入一份 `tree[cname]`。

核心字段（以当前 `feat/ast-timing-fix` 为准）包括：

| key | 当前含义 | 备注 |
|---|---|---|
| `children` | 子 class 名列表 | 按 forward call order 优先排序 |
| `attrs` | `attr_name -> class_ref` | 仍是字符串映射，不是 Attr 对象 |
| `dep_edges` | `[(from_attr, to_attr), ...]` | sibling-level 数据依赖 |
| `dep_edge_locs` | `{(from_attr, to_attr): evidence_dict}` | edge evidence |
| `containers` | 容器元信息（当前基本未成为 typed graph 主入口） | 历史残留 |
| `is_torch_native` | 是否 torch native module | 展示辅助 |
| `dynamic_setattr_attrs` | 动态 setattr attr 名集合 | 历史 fallback 元数据 |
| `container_kw` | container 实例长度裁剪信息 | group 展开辅助 |
| `attr_def_loc` | `attr -> (file, line)` | `__init__` 定义行 |
| `first_call_loc` | `attr -> (file, line)` | 第一次 forward 调用行 |
| `input_source_attrs` | `set(attr_name)` | 仅有集合，没有 InputAttr 对象 |
| `result_attrs` | `{result_var: [ResultAttr, ...]}` | 这里已经是 typed object |
| `static_diagnostics` | 静态分析诊断 | 可复用为 DAG 构图诊断 |
| `container_kinds` | `container_attr -> kind` | `ModuleDict` / `ModuleList` / `Sequential` |
| `conditional_attrs` | 条件 attr 信息 | 现阶段不直接进入 DagNode 类型 |
| `unresolved_moduledict` | 未解析告警 | 诊断信息 |
| `instance_kw_list_lens` | per-instance 裁剪信息 | 递归展开辅助 |
| `_ast_*` | AST ground truth 辅助信息 | 当前仍是对旧路径的补充 |

问题不在于 `tree` 信息不够，而在于：**这些信息还没有被收口成后端唯一 graph model**。

### 1.3 `_build_data_dependency_edges()` 当前返回值结构

`_build_data_dependency_edges()` 当前返回：

```python
(edges_dict, edge_locs_dict, split_info)
```

其中：

- `edges_dict: {class_name: [(from_attr, to_attr), ...]}`
- `edge_locs_dict: {class_name: {(from_attr, to_attr): evidence_dict}}`
- `split_info: {class_name: {orig_attr: [orig_attr#0, orig_attr#1, ...]}}`

`dep_edge_locs[(from_attr, to_attr)]` 的 evidence 结构当前已经很接近目标态：

| 字段 | 当前含义 |
|---|---|
| `file` | evidence 文件 |
| `var` | carrying variable 名 |
| `from_line` | producer 产出行 |
| `to_line` | consumer 消费行 |
| `lineage` | 变量链 steps，供后续转成前端可展示 evidence |

也就是说，**module→module edge 的后端证据已经基本成熟**；真正缺的是 Input / Result edge 也收口到同一后端 graph 层。

### 1.4 `result_attrs` 当前存储方式

`ast_frontend.py:get_result_attrs()` 当前返回：

```python
{result_var_name: [ResultAttr, ...]}
```

每个 `ResultAttr` 当前包含：

- `attr_name`：result var 名
- `class_name`
- `call_loc`：`result.head(...)` 调用行
- `source_expr`
- `head_name`
- `classifier_type`
- `prediction_expr`
- `label_expr`
- `sample_rate_expr`
- `loss_expr`

这意味着：

1. `ResultAttr` 已经是**后端语义对象**
2. Result edge 本应由后端从 `ResultAttr` 直接生成
3. 现在前端 `_build_root_result_edges_from_result_attrs()` 仍在二次扫描源码，本质上是错位

### 1.5 `frontend_html.py` 当前 DAG 组装方式

当前 `generate_html_flowchart()` 的问题是：**typed data 在后端，graph 组装却在前端**。

#### 1.5.1 `_root_attr_id_map` / `attr_id_map`

当前 `build_dag_recursive()` 在递归构造 group / leaf node 时，临时维护：

```python
attr_id_map[attr_name] = ("node" | "group", id)
```

用途：

- 把 `tree[cname]["attrs"]` 的 attr 名映射到当前 DAG node/group id
- 后续 root-level Input / Result edge 再通过这个 map 找到真实 node id
- container wrapping 时还要重写一次 `attr_id_map`

这说明现在前端承担了两件不该由前端承担的事：

1. **实例化 DagNode id**
2. **维护 attr 到 node id 的唯一映射**

#### 1.5.2 `_root_result_edges` / `_root_edge_locs`

当前 root 级连接是这样做的：

- `_root_edge_locs = tree[root]["dep_edge_locs"]`
- `_root_result_edges = _build_root_result_edges_from_result_attrs(root_cname)`
- 然后前端用 `_root_attr_id_map` 把 attr 名解析成 node id，再拼 `dag_edges`

这里最大的问题是：

- `module→module` edge 来自后端 `dep_edge_locs`
- `module→Result` edge 却由前端临时从 `ResultAttr.prediction_expr` + forward 扫描重新推断
- `Input→module` edge 又由 `_input_edge_var_history()` 和若干 fallback 逻辑补

同一张图，三种来源，语义层次不一致。

#### 1.5.3 当前 `dag_nodes` / `dag_edges` 产物格式

当前前端最后给 JS 的结构大致是：

- `dag_nodes`: leaf node 列表
- `dag_groups`: group/root/container 列表
- `dag_edges`: edge 列表

leaf node 当前 shape（摘要）：

| 字段 | 当前含义 |
|---|---|
| `id` | node id |
| `attr_name` | attr 名 |
| `class_name` | class 名 |
| `depth` | 层级 |
| timing 字段 | `kernel_us` / `inclusive_us` / `pct` 等 |
| 源码字段 | `src_file` / `src_start_line` / `src_snippet` |
| ctor 字段 | `ctor_src_file` / `ctor_src_line` / `ctor_src_excerpt` |

edge 当前 shape（摘要）：

| 字段 | 当前含义 |
|---|---|
| `from` / `to` | node/group id |
| `type` | 现阶段大多是 `dep`，少量 `boundary` |
| `from_attr` / `to_attr` | attr 名 |
| `evidence` | file/line/excerpt/var_history 等 |

结论：**JS 真正依赖的是最终 flowchart JSON，而不是 tree。**
所以重构目标不是改 JS，而是把 Python 侧的“DAG 组装权”从 `frontend_html.py` 收回到 `dag_types.py + analyze_trace.py`。

### 1.6 `edge_highlight_redesign.md` 已经确认的设计约定

已有设计里已经定稿的约束，需要继续保持：

1. `ModuleAttr` / `ContainerAttr` / `InputAttr` / `ResultAttr` 是一等公民
2. `InputAttr` 保留 `call_loc` + `forward_use_loc` 双 loc
3. Input edge evidence 链首插 `role="input_source"`
4. `ResultAttr` 复用 `var_lineage`
5. `first_call_loc` 在新构图下最终应被消除
6. `_scan_root_result_edges()` / `_input_edge_var_history()` 要废弃

这与本设计完全一致，因此 `dag_types.py` 不是另起炉灶，而是**把 edge_highlight 设计真正落到图模型层**。

---

## 2. DagNode 类设计

### 2.1 文件位置

新文件：

```text
scripts/dag_types.py
```

与 `attr_types.py` 同目录，职责明确分层：

- `attr_types.py`：源码语义对象
- `dag_types.py`：图语义对象

### 2.2 `DagNode` 设计目标

`DagNode` 表示“已经进入 DAG 语义层的节点”，它**不继承 `Attr`**，而是由 Attr 或 root-level synthetic concept 组装得到。

关键原则：

1. **Attr ≠ DagNode**
   - Attr 描述“源码对象”
   - DagNode 描述“图中的节点”
2. **DagNode 允许 synthetic node**
   - `Input`
   - `Root`
   - `Result`
3. **Root / Container 也是 node，不再需要前端额外发明 group 概念**
   - group 只是前端现有 JSON 的一种投影
   - 后端 graph 内部统一用 `DagNode`

### 2.3 `node_type` 枚举

`DagNode.node_type` 统一枚举为：

| 枚举值 | 含义 | 来源 |
|---|---|---|
| `Input` | 顶层输入聚合节点 | synthetic |
| `Root` | 顶层 root 容器节点 | synthetic |
| `ModuleAttr` | 普通用户 module attr 节点 | 由 `ModuleAttr` 组装 |
| `ContainerAttr` | `ModuleList` / `ModuleDict` / `Sequential` 容器节点 | 由 `ContainerAttr` 组装 |
| `ResultAttr` | 单个 `result.head(...)` 语义节点 | 由 `ResultAttr` 组装 |
| `Result` | 顶层输出聚合节点 | synthetic |

说明：

- 这里不单独再定义 `InputAttr` node_type
- `InputAttr` 仍然存在于 Attr 层，用来构建 **Input edge evidence**
- 图层里仍保留单一 `Input` 节点，以兼容现有 `Input -> Root -> Result` 顶层结构

### 2.4 建议字段列表

`DagNode` 建议字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `node_id` | `str` | 稳定唯一 id，建议 path-based，而不是 `node_1` 这种递增号 |
| `node_type` | `str` | 上述枚举值 |
| `label` | `str` | 前端展示标签 |
| `attr_name` | `str | None` | 对应 attr 名；synthetic node 可为空或固定值 |
| `class_name` | `str | None` | 叶子/容器对应 class 名 |
| `call_loc` | `CallLoc | None` | 节点主定位行 |
| `parent_id` | `str | None` | 父节点 id |
| `owner_cname` | `str | None` | 所属 class 名（构图时用于诊断/调试） |
| `depth` | `int` | 逻辑深度 |
| `container_kind` | `str | None` | `ModuleList` / `ModuleDict` / `Sequential` |
| `source_expr` | `str | None` | Attr 原始表达式 |
| `forward_use_loc` | `CallLoc | None` | 仅 Input 相关节点/语义需要 |
| `head_name` | `str | None` | 仅 `ResultAttr` |
| `classifier_type` | `str | None` | 仅 `ResultAttr` |
| `children_ids` | `list[str]` | 仅 `Root` / `ContainerAttr` / 可展开 `ModuleAttr` |
| `call_order` | `list[str]` | 子节点 id 顺序，替代前端临时 `attr_id_map + call_order` |
| `is_group_like` | `bool` | 是否投影成现有 `dag_groups` |
| `metadata` | `dict[str, Any]` | 预留诊断字段，不承载核心语义 |

### 2.5 `call_loc` 取值规则

不同 node_type 的 `call_loc` 约定：

| node_type | `call_loc` 取值 |
|---|---|
| `Input` | `None`（Input 是聚合节点，具体 source 在 edge evidence 中） |
| `Root` | Root class 源码定义位置或 `None` |
| `ModuleAttr` | 该 attr 的真实调用点；过渡阶段可先用 `first_call_loc`，最终由 Attr 实例自带 `call_loc` 替代 |
| `ContainerAttr` | 容器在 forward 中被引用/展开的位置；无真实调用时为容器定义行 |
| `ResultAttr` | `result.head(...)` 调用行 |
| `Result` | `None` |

### 2.6 与现有 Attr 类的关系

这里必须严格区分三层：

1. **Attr 层**
   - `ModuleAttr`
   - `ContainerAttr`
   - `InputAttr`
   - `ResultAttr`

2. **DagNode 层**
   - 从 Attr 组装 graph node
   - synthetic node（`Input` / `Root` / `Result`）也在这一层

3. **前端 JSON 层**
   - `DagNode -> dag_nodes / dag_groups` 的投影

结论：

- `DagNode` **不继承** `Attr`
- `DagNode` 通过组合引用 Attr 信息
- `ModuleAttr` / `ContainerAttr` / `ResultAttr` 节点由对应 Attr 组装
- `Input` / `Root` / `Result` 是 synthetic DagNode，不对应单个 Attr

---

## 3. DagEdge 类设计

### 3.1 设计目标

`DagEdge` 统一表达 graph 中所有边，彻底替代当前前端分散构造：

- sibling `dep`
- `Input -> module`
- `module -> Result`

### 3.2 `edge_type` 枚举

内部语义枚举固定为：

| 枚举值 | 含义 |
|---|---|
| `dep` | 普通 module/container 之间的数据依赖 |
| `result` | producer -> Result / ResultAttr 相关边 |
| `input` | Input -> consumer 相关边 |

说明：

- 这是 **后端 graph 语义**
- 前端序列化到旧 JS 协议时，可统一映射回当前 `type="dep"`，同时保留一个额外 `edge_kind=edge_type` 供新逻辑使用
- 这样可以做到“JS 不动，但后端语义变清晰”

### 3.3 建议字段列表

| 字段 | 类型 | 含义 |
|---|---|---|
| `edge_id` | `str` | 稳定唯一 id，可由 `from_id + to_id + edge_type` 生成 |
| `from_id` | `str` | 起点节点 id |
| `to_id` | `str` | 终点节点 id |
| `edge_type` | `str` | `dep` / `result` / `input` |
| `from_attr` | `str | None` | 兼容旧前端 evidence panel |
| `to_attr` | `str | None` | 兼容旧前端 evidence panel |
| `evidence` | `EvidenceInfo | None` | 统一证据结构 |
| `metadata` | `dict[str, Any]` | 预留聚合/诊断字段 |

### 3.4 `EvidenceInfo` 结构

建议在 `dag_types.py` 同文件定义 `EvidenceInfo`，字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `file` | `str` | 证据文件 |
| `from_line` | `int | None` | producer 行 |
| `to_line` | `int | None` | consumer 行 |
| `var` | `str | None` | carry var |
| `from_excerpt` | `ExcerptInfo | None` | producer 片段 |
| `to_excerpt` | `ExcerptInfo | None` | consumer 片段 |
| `var_history` | `list[EvidenceStep]` | 兼容现有前端 source panel |
| `lineage_by_carrier` | `dict[str, list[EvidenceStep]]` | 新 edge_highlight 设计要求的 carrier 视图 |
| `diagnostics` | `list[str]` | 无 fallback 场景下的未解析说明 |

其中：

- `ExcerptInfo = {start, end, text, highlight}`
- `EvidenceStep = {var, file, line, excerpt, role, carriers, arg_carriers}`

### 3.5 三类边的构造来源

#### 3.5.1 `dep` 边

直接来自：

- `tree[cname]["dep_edges"]`
- `tree[cname]["dep_edge_locs"]`

这是当前最成熟的路径，原则上只做 typed 化，不改语义。

#### 3.5.2 `input` 边

不再由前端 `_input_edge_var_history()` 临时合成，改为后端统一构造：

- Input 起点固定为 synthetic `Input` node
- 终点为 root 下真实 consumer 节点
- evidence 来源于 `InputAttr.call_loc + forward_use_loc + var_lineage`
- `lineage_by_carrier` 链首插入 `role="input_source"` step

#### 3.5.3 `result` 边

不再由前端 `_build_root_result_edges_from_result_attrs()` 扫 forward 得到。

改为：

1. 从 `tree[root]["result_attrs"]` 读取 `ResultAttr`
2. 直接对 `prediction_expr` 做 AST parse
3. 提取其中引用到的 producer attr
4. 生成 `producer -> Result`（或中间 `producer -> ResultAttr -> Result`）语义边

**关键约束：不扫 forward，不做 regex fallback。**

如果 `prediction_expr` AST parse 之后仍只得到普通局部变量而不是 module attr：

- 不再退回前端源码扫描
- 直接把未解析信息记入 `EvidenceInfo.diagnostics`
- 由后续专门的后端别名求值能力解决，而不是让前端兜底

---

## 4. DagGraph 类设计

### 4.1 顶层结构

`DagGraph` 顶层字段按要求保持最小集合：

```python
nodes: list[DagNode]
edges: list[DagEdge]
```

不在顶层继续暴露 `groups` / `attr_id_map` / `_root_result_edges` 这类前端组装期临时数据。

### 4.2 静态构造入口

按 NaN 当前要求，统一入口定义为：

```python
DagGraph.from_tree(tree, root_cname, source_files)
```

建议职责拆成 4 个明确阶段：

1. `_materialize_attr_objects(tree, root_cname)`
   - 从当前 `tree` 元数据补齐 `ModuleAttr` / `ContainerAttr` / `InputAttr` / `ResultAttr` 视图
2. `_build_nodes(...)`
   - 生成 `Input` / `Root` / `ModuleAttr` / `ContainerAttr` / `ResultAttr` / `Result`
3. `_build_edges(...)`
   - 统一生成 `dep` / `input` / `result`
4. `_validate_graph(...)`
   - 检查 dangling id / duplicate edge / unresolved result producer 等诊断

### 4.3 建图原则

#### 4.3.1 统一 node id

建议 `node_id` 改为**稳定 path-based id**，而不是 `node_17` 这种运行期递增：

| 节点 | 建议 id |
|---|---|
| Input | `input` |
| Root | `root:<root_cname>` |
| 普通 attr | `attr:<parent_path>.<attr_name>` |
| 容器 | `container:<parent_path>.<attr_name>` |
| ResultAttr | `result_attr:<result_var>@<line>` |
| Result | `result` |

好处：

1. 单测稳定
2. 前后端比对容易
3. 删除 `attr_id_map` 后仍可直接通过 id 定位节点

#### 4.3.2 Root / Container 统一当作 group-like node

当前前端同时维护：

- `dag_nodes`
- `dag_groups`

新设计里，后端 `DagGraph` 内部统一只有 `DagNode`；其中：

- `Root`
- `ContainerAttr`
- 需要展开的 `ModuleAttr`

都携带 `children_ids` / `call_order`，在前端序列化层再投影成现有 `dag_groups` shape。

#### 4.3.3 Result 边构造规则

Result 相关分两层语义：

- graph 内部保留 `ResultAttr` node，便于调试和后续扩展
- 但为兼容当前 JS，前端序列化时仍可直接输出 `producer -> Result`

也就是说：

- **图层可以更精确**
- **展示层继续兼容当前 Result 单节点协议**

### 4.4 `from_tree()` 的输入依赖

`DagGraph.from_tree()` 只允许依赖以下后端输入：

- `tree[cname]["attrs"]`
- `tree[cname]["dep_edges"]`
- `tree[cname]["dep_edge_locs"]`
- `tree[cname]["result_attrs"]`
- `tree[cname]["input_source_attrs"]`（过渡期）或未来 `input_attrs`
- `tree[cname]["container_kinds"]`
- `tree[cname]["attr_def_loc"]`
- `tree[cname]["static_diagnostics"]`
- `source_files`

**明确排除：**

- 前端扫描函数
- root 级二次 forward 扫描
- `_infer_input_edges_from_call_args()` 之类临时补洞逻辑

### 4.5 `Result` 边构造细则

NaN 已明确：**直接从 `tree[root]["result_attrs"]` 的 `prediction_expr` 提取 attr 名（AST parse），不扫 forward。**

因此建议：

1. 对每个 `ResultAttr.prediction_expr` 执行 `ast.parse(..., mode="eval")`
2. 只识别以下 producer 形态：
   - `self.xxx(...)`
   - `self.container[idx](...)`
   - `getattr(self, "xxx")(...)`
3. AST 解析出 producer attr 后，直接落 `producer -> Result`（或 `producer -> ResultAttr -> Result`）
4. 如果表达式是局部变量 / alias，而 AST 无法直接映射到 attr：
   - 标记 unresolved
   - 记录 diagnostics
   - **不 fallback 到 forward 扫描**

这是本次设计里最重要的边界：**把“没有解析出来”显式暴露，而不是偷偷走旧路径。**

---

## 5. 后端集成点

### 5.1 集成位置

建议把 `DagGraph.from_tree()` 放在 `build_static_module_tree()` 尾部、所有 `tree[cname]` 元数据补齐之后调用。

也就是在以下元数据都完成后再构图：

- `attr_def_loc`
- `first_call_loc`（过渡期仍可读，最终逐步消除）
- `input_source_attrs`
- `result_attrs`
- `container_kinds`
- `static_diagnostics`
- `_ast_*`

### 5.2 推荐时机

推荐时机：

1. `tree` 已完成 root 过滤
2. `roots` 已确定主 root
3. `result_attrs` 已经挂到 root class
4. 然后执行：

```text
primary_root = roots[0] if roots else None
if primary_root:
    tree[primary_root]["dag_graph"] = DagGraph.from_tree(tree, primary_root, source_files)
```

### 5.3 为什么只挂在 root 上

`DagGraph` 表示的是**整张 DAG**，不是某个 class 局部视图。

因此不建议把完整 graph 复制到每个 `tree[cname]` 上；推荐规则：

- 仅主 root `tree[root_cname]["dag_graph"]` 挂整图
- 其他 `tree[cname]` 不挂，避免重复和歧义

### 5.4 analyze_trace.py 中需要新增/调整的内容

后端侧建议新增明确的 3 层职责：

1. `attr_types.py`
   - 继续承载 Attr 类型
2. `dag_types.py`
   - 新增 `DagNode` / `DagEdge` / `EvidenceInfo` / `DagGraph`
3. `analyze_trace.py`
   - 只负责：`tree` 元数据补齐 + `DagGraph.from_tree()` 调用 + 挂入 root tree entry

这样 `analyze_trace.py` 不需要再理解前端 `attr_id_map`、`_root_result_edges` 等展示实现细节。

---

## 6. 前端适配

### 6.1 目标

`frontend_html.py` 的职责应收敛为：

1. 读取 `tree[root]["dag_graph"]`
2. 把 `DagGraph` 序列化成现有 flowchart JSON
3. 保持 JS 协议不变
4. 不再做任何源码扫描 / DAG 边推断

### 6.2 新的前端输入方式

当前：

```text
tree -> build_dag_recursive() -> attr_id_map / root_result_edges / input fallback -> dag_nodes/dag_edges
```

目标：

```text
dag_graph -> serialize_for_flowchart() -> dag_nodes/dag_groups/dag_edges
```

也就是说，前端只做**格式投影**，不再做**图语义构造**。

### 6.3 `frontend_html.py` 需要替换的核心点

#### 6.3.1 `_root_attr_id_map`

当前 `_root_attr_id_map = _root_group.get("attr_id_map", {})` 的整条链路都可以去掉。

原因：

- `DagGraph` 里每个节点已经有稳定 `node_id`
- edge 已经直接存 `from_id` / `to_id`
- 前端不应再通过 `attr_name -> id` 二次解析

#### 6.3.2 `attr_id_map`

`build_dag_recursive()` 里所有 `attr_id_map` 构造与重写逻辑都应移除。

包括但不限于：

- leaf/group 创建时的 `attr_id_map[attr_name] = ...`
- container wrapping 之后对 `attr_id_map` 的覆盖
- root 级 `_root_attr_id_map` 的二次消费

这些都是 graph 生成阶段逻辑，不属于前端。

#### 6.3.3 `_root_result_edges` / `_root_edge_locs`

两者都不应再由前端读取或构造：

- `_root_edge_locs` 的 `dep_edge_locs` 已在后端变成 `DagEdge.evidence`
- `_root_result_edges` 已在后端变成 `edge_type="result"` 的 `DagEdge`

#### 6.3.4 Input / Result 边前端补洞逻辑

下面这些路径都应删除：

- `_infer_input_edges_from_call_args()`
- `_input_edge_var_history()`
- “没有 inbound 的 root child 自动补 Input 边”
- `Input -> group` / `Input -> child` 的多轮补边逻辑
- `_build_root_result_edges_from_result_attrs()`
- `_scan_root_result_edges()`

这些逻辑的共同问题是：**前端在推断边，而不是渲染边。**

### 6.4 `dag_nodes` / `dag_edges` JSON 格式兼容策略

JS 层不动，因此 Python 序列化层要负责兼容：

#### 6.4.1 Leaf node

`DagNode(node_type=ModuleAttr)` 继续序列化成当前 `dag_nodes` shape，保留：

- `id`
- `attr_name`
- `class_name`
- `depth`
- timing 字段
- source 字段
- ctor 字段

#### 6.4.2 Group node

`DagNode(node_type in {Root, ContainerAttr})` 序列化成当前 `dag_groups` shape，保留：

- `id`
- `label`
- `attr_name`
- `depth`
- `children_nodes`
- `children_groups`
- `call_order`
- `internal_edges`
- timing/source 字段

#### 6.4.3 Edge

`DagEdge` 序列化成当前 `dag_edges` shape：

- `from`
- `to`
- `type`
- `from_attr`
- `to_attr`
- `evidence`

兼容策略：

- `DagEdge.edge_type` 内部保留 `dep` / `input` / `result`
- 对 JS 输出时仍写 `type="dep"`
- 可额外透传 `edge_kind=edge_type`，JS 当前忽略也不影响

这样可以做到：

- 旧 JS 不改
- 新 Python graph 语义更清楚

---

## 7. 删除范围

### 7.1 后端可删除/收缩的代码范围

本次 typed graph 落地后，后端应删除或收缩的主要是“给前端扫描路径喂辅助元数据”的逻辑。

#### 必删

1. `frontend_html.py:_build_root_result_edges_from_result_attrs()`
2. `frontend_html.py:_scan_root_result_edges()`
3. `frontend_html.py:_input_edge_var_history()`
4. `frontend_html.py:_infer_input_edges_from_call_args()`

#### 可收缩为仅诊断用途

1. `tree[cname]["first_call_loc"]`
   - 过渡期可继续保留供 DagGraph materialization 使用
   - 一旦 `ModuleAttr.call_loc` 全面落到 Attr 层，可逐步移除
2. `tree[cname]["dynamic_setattr_attrs"]`
   - 不再允许前端消费作为 fallback
   - 若仍有价值，只保留在 diagnostics 中

### 7.2 前端可删除的代码块

以下几类代码块，在 `DagGraph` 上线后都应删除：

#### Root 级 Input / Result 拼边整段逻辑

大致覆盖当前：

- `_root_attr_id_map` 读取
- `_root_edge_locs` 读取
- `_root_result_edges` 构造
- `Input -> consumer` 多轮补边
- `producer -> Result` 拼边

这部分本质都属于“从 tree 临时重建 graph”。

#### `attr_id_map` 全链路

包括：

- leaf/group 创建时写 `attr_id_map`
- container wrapping 时重写 `attr_id_map`
- root 级再读 `attr_id_map`

#### container wrapping 中只服务前端映射的部分

container 本身仍需要，但下面这些“为了兼容前端临时 id map 的改写”不再需要：

- parent `attr_id_map` 重写
- container 自己的 `_cont_attr_id_map`
- 依赖 `attr_id_map` 的 root-level endpoint 解析

### 7.3 可以整体替换的函数级入口

从设计上看，当前 `build_dag_recursive()` 最终可以整体被以下新流程替代：

```text
build_static_module_tree()
  -> tree[root]["dag_graph"]
  -> serialize_dag_graph_for_frontend(dag_graph, timing_data, source_info)
```

也就是说，当前前端递归构图函数不应继续承担 graph build 职责，最多保留一个“DagGraph -> flowchart JSON”的纯序列化函数。

---

## 8. 分阶段实现建议

### Phase 3.1：定义 `DagNode` / `DagEdge` / `DagGraph`，写单测

目标：先把类型层站稳，不碰现有前端渲染。

建议单测覆盖：

1. `DagNode` 字段完整性
2. `DagEdge` evidence 序列化完整性
3. `DagGraph.from_tree()` 对 module/container 节点的基础构图
4. `ResultAttr.prediction_expr` AST parse 结果是否稳定
5. 无 fallback 约束：未解析 result producer 时必须出 diagnostics，而不是偷偷补边

### Phase 3.2：后端集成 `DagGraph.from_tree()`，Result 边在后端完成

目标：让 `analyze_trace.py` 真正产出 root-level `dag_graph`。

落地点：

1. `build_static_module_tree()` 尾部生成 `tree[root]["dag_graph"]`
2. 把 `dep_edge_locs` 统一转成 `DagEdge.evidence`
3. 把 `result_attrs` 直接转成 `result` edge
4. Input edge evidence 后移到后端
5. 保留旧前端读取路径仅作为短暂兼容层，但**不再新增任何 fallback**

### Phase 3.3：前端适配，删除旧扫描路径

目标：`frontend_html.py` 只做序列化，不再做 graph 推断。

具体顺序：

1. 新增 `serialize_dag_graph_for_frontend()`
2. 从 `tree[root]["dag_graph"]` 生成 `dag_nodes` / `dag_groups` / `dag_edges`
3. 删除 `_build_root_result_edges_from_result_attrs()`
4. 删除 `_scan_root_result_edges()`
5. 删除 `_input_edge_var_history()`
6. 删除 root 级 Input / Result 多轮补边逻辑
7. 删除 `attr_id_map` 相关解析链

---

## 9. 最终结论

这次重构的核心不是“把旧前端逻辑搬到另一个文件”，而是把职责真正收口：

- `attr_types.py`：表达源码对象
- `dag_types.py`：表达图对象
- `analyze_trace.py`：从源码对象构图
- `frontend_html.py`：只消费图对象并序列化为旧 JS 协议

只要 `DagGraph.from_tree()` 成为唯一 graph 构造入口，就能同时解决当前几个根问题：

1. `Input / Result / module` 三类边来源不统一
2. 前端仍在扫描源码和补洞
3. `attr_id_map` / `_root_result_edges` / `_root_edge_locs` 这类临时结构过多
4. 新 Attr 体系没有真正成为 DAG 唯一数据源

建议严格按 Phase 3.1 → 3.2 → 3.3 推进，中间不要再新增任何 tree-to-frontend 的过渡 fallback。