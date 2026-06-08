# DAG-to-HTML 接入设计

## 1. 目标
前端做 **最小必要改动**，仅修改 IO 字段识别逻辑（旧单值 → 四分类集合）。`has_timing` / `pct` / `dur_us` / `kernel_us` / `role` 等 timing 字段保留供后续补实现，不在本次删除；本次通过后端 adapter 显式填默认值。动态 DAG 构图链路增加一层后端 adapter，把 `serialize_dag(dag, registry)` 产出的 `SerializedDAG` 转成 `render_flowchart_to_file()` 可直接消费的 `flowchart_data`，从而让 `run_dag_session.py` 在输出文本报告之外同步产出 HTML DAG。

## 2. 总体方案
采用 **adapter 层** 方案，不直接改写现有前端模板，也不把旧 schema 兼容逻辑散落到 `run_dag_session.py` 或 `frontend_html.py` 中。

整条链路如下：

```text
DAG + registry
  → serialize_dag()        # dag_serializer.py（已完成）
  → SerializedDAG dict
  → adapt_serialized_dag() # 新增 adapter 函数（位置待定）
  → flowchart_data dict
  → render_flowchart_to_file()  # frontend_html.py（已有）
  → HTML 文件
```

adapter 放在新文件：

- `scripts/dag_html_adapter.py`

这样拆分的原因：

1. `dag_serializer.py` 当前职责很清晰：把 DAG IR 递归序列化成稳定 dict，不应该直接耦合旧前端 schema（`dag_serializer.py:131-147`）。
2. `frontend_html.py` 当前唯一仍可用的入口就是 `render_flowchart_to_file(flowchart_data, output_path)`（`frontend_html.py:1700-1704`）；旧的 `generate_html_flowchart*` 路径已经全部 `NotImplementedError`（`1687-1699`）。
3. `run_dag_session.py` 在 `_run_single_mode()` 中已经完成 `session.finalize()`、DCE、统计和 validate，但目前完全没有 HTML 产出逻辑（`run_dag_session.py:427-455`），因此最小接入点是在 DCE 之后插入 `serialize_dag → adapter → render`。
4. adapter 单独成文件后，后续 UT 可以只针对字段转换做纯函数测试，不需要把前端模板或 mock run 绑进测试。

## 3. SerializedDAG → flowchart_data 字段映射

### 3.1 顶层字段映射表

`serialize_dag()` 当前输出的顶层结构为：
- `input_nodes`
- `param_nodes`
- `const_nodes`
- `output_nodes`
- `nodes`
- `edges`

见 `dag_serializer.py:131-147`。

适配后的 `flowchart_data` 顶层字段建议如下：

| SerializedDAG 字段 | flowchart_data 字段 | 映射规则 |
|---|---|---|
| `input_nodes` | `input_node_ids` | 提取每个 entry 的 `node_id`，直接生成 list[int] |
| `param_nodes` | `param_node_ids` | 提取每个 entry 的 `node_id`，直接生成 list[int] |
| `const_nodes` | `const_node_ids` | 提取每个 entry 的 `node_id`，直接生成 list[int] |
| `output_nodes` | `output_node_ids` | 提取每个 entry 的 `node_id`，直接生成 list[int] |
| `nodes` | `nodes` | 转成前端 leaf node 列表，规则见 3.2 |
| `edges` | `edges` | 转成前端 edge dict，规则见 3.3 |
| （新增） | `root_groups` | 从所有 group node 中提取 `depth == 0` 的 group id |
| （新增） | `groups` | 从所有 module/group node 递归展开后生成前端 group 结构，规则见 3.2 |
| （新增） | `meta` | adapter 注入 `model_id`、`mode`、`time`、`num_modules`、`roots` |
| （新增） | `has_timing` | 固定 `False` |

注意点：

1. 旧模板目前仍写死使用单值字段 `DATA.input_node_id` / `DATA.loss_node_id`：
   - `isIONodeId()`：`frontend_html.py:335-337`
   - 顶层 IO pill 渲染：`frontend_html.py:1254-1268`
2. 本次接入严格采用 IO 四分类，旧字段 `input_node_id` / `loss_node_id` 从 `frontend_html.py` 里删除；timing 字段保留（adapter 填默认值），evidence/src 字段保留有实际语义。
3. 因此前端必须同步修改为按四个集合识别 IO；adapter 不应再造 `input_node_id` / `loss_node_id` 这种 shim。
4. `meta.time` 由 adapter 用当前时间写入；`meta.model_id` / `meta.mode` 由 `run_dag_session.py` 调用 adapter 时显式传参。
5. `meta.num_modules` 建议定义为 `len(groups) + len(nodes)`，直接复用 header / summary 展示逻辑（`frontend_html.py:1282-1324`）。
6. `meta.roots` 建议写为顶层 group 的 `label` 列表，供 `meta-info` 中 `Root:` 文本直接展示（`frontend_html.py:1287-1289`）。

### 3.2 Node 映射规则

#### 3.2.1 SerializedDAG 当前 node 结构

`serialize_dag()` 中 `nodes` 只来自 `dag.nodes`，且强约束这些 id 必须对应 `ModuleNode`（`dag_serializer.py:145,150-154`）。`_serialize_module_node()` 目前会输出两类结构：

1. **容器节点**（`dag_serializer.py:51-70`）
   - 字段：`node_id`、`label`、`call_loc`、`is_container`、`container_kind`、`children_nodes`
2. **普通 module / functional 节点**（`dag_serializer.py:72-96`）
   - 字段：`node_id`、`label`、`call_loc`、`attr_type`、`inner_dag`

因此 adapter 不能只把 `serialized["nodes"]` 平铺给前端；必须递归遍历 `inner_dag`，把整个树拆成：
- `groups`：可展开节点
- `nodes`：叶子节点

#### 3.2.2 group / leaf 判定

建议判定规则如下：

1. **group node**：满足以下任一条件
   - 存在 `inner_dag != None`
   - 或存在 `children_nodes`（容器节点）

2. **leaf node**：满足以下任一条件
   - module / functional 节点且 `inner_dag is None`
   - 四类 IO 节点（input / param / const / output）

IO 节点也必须进入 `flowchart_data.nodes`，原因有二：

1. `renderIOPill()` 内部会执行 `const n = nodeMap[nid]`，再绑定 tooltip / hover（`frontend_html.py:1213-1230`）；如果 IO 节点不在 `nodeMap`，pill 只能显示静态圆角框，无法显示节点详情。
2. `applyEdgeFocusState()` 把 IO 节点也当作普通 `nodeDomRegistry` 成员处理（`frontend_html.py:339-401`），因此 IO 节点最好和其他 leaf node 使用统一 node 结构。

#### 3.2.3 group 字段映射

前端 group 渲染与布局依赖这些字段：
- `id`、`label`、`depth`：map 初始化、默认折叠、颜色（`frontend_html.py:456-472`, `480-503`）
- `children_nodes` / `children_groups`：祖先索引与递归布局（`456-470`, `755-885`）
- `internal_edges`：group 内部边布局（`971-985`）
- `src_file` / `src_start_line` / `src_end_line`：信息按钮与源码面板（`808-831`, `872-879`）
- `has_timing` / `pct` / `dur_us`：模板中有条件访问（`861-869`）

建议 group 输出结构：

| flowchart_data group 字段 | 来源 | 映射规则 |
|---|---|---|
| `id` | `node_id` | 直接赋值 |
| `label` | `label` | 直接赋值 |
| `depth` | adapter 递归深度 | 根 DAG 第一层 group 为 0，向下逐层 +1 |
| `node_type` | adapter 固定 | module 节点写 `group`；容器节点写 `container_group` |
| `children_nodes` | 递归遍历结果 | 仅保留直属 leaf child 的 id 列表 |
| `children_groups` | 递归遍历结果 | 仅保留直属 group child 的完整 dict 列表 |
| `internal_edges` | 从当前层 `inner_dag.edges` 或容器的边集合生成 | 结构见下文 |
| `src_file` | `call_loc.file` | 无则填 `null` |
| `src_start_line` | `call_loc.line` | 无则填 `null` |
| `src_end_line` | `call_loc.line` | 第一阶段无源码范围信息，先与 start 相同 |
| `has_timing` | adapter 固定 | `False` |
| `pct` | adapter 固定 | `0` |
| `dur_us` | adapter 固定 | `0` |
| `kernel_us` | adapter 固定 | `0` |
| `role` | adapter 固定 | `null` |

`internal_edges` 推荐字段：

| internal_edges 字段 | 来源 | 说明 |
|---|---|---|
| `from_child` | edge 源节点 id | 必须是当前 group 直属 child id |
| `to_child` | edge 目标节点 id | 必须是当前 group 直属 child id |
| `type` | containment/data 映射结果 | 当前设计只会是 `data` / `containment` |
| `from_attr` | edge 源节点 label | edge panel 标题会读取（`frontend_html.py:1517-1524`） |
| `to_attr` | edge 目标节点 label | 同上 |
| `parent_class` | 当前 group.label | 供 side panel subtitle 使用 |
| `evidence` | edge.evidence | 第一阶段直接透传 |
| `tensor_info` | edge.tensor_info | 直接透传 |

#### 3.2.4 leaf node 字段映射

前端 node 侧主要依赖：
- `id`、`label`、`depth`
- `class_name` / `node_type`
- `src_file` / `src_start_line` / `src_end_line`
- `has_timing` / `pct` / `dur_us` / `kernel_us` / `role`

因此建议统一输出：

| flowchart_data leaf 字段 | 来源 | 映射规则 |
|---|---|---|
| `id` | `node_id` | 直接赋值 |
| `label` | `label` | 直接赋值 |
| `depth` | adapter 递归深度 | 该 leaf 所在 DAG 层级 |
| `node_type` | `io_subtype` / `attr_type` | IO 节点分别写 `input` / `param` / `const` / `output`；普通节点写 `module` / `functional` |
| `class_name` | `label` 或 subtype 名称 | 供顶层 output pill sublabel 使用（`frontend_html.py:1259-1267`） |
| `src_file` | `call_loc.file` | 无则 `null` |
| `src_start_line` | `call_loc.line` | 无则 `null` |
| `src_end_line` | `call_loc.line` | 第一阶段与 start 相同 |
| `has_timing` | adapter 固定 | `False` |
| `pct` | adapter 固定 | `0` |
| `dur_us` | adapter 固定 | `0` |
| `kernel_us` | adapter 固定 | `0` |
| `role` | adapter 固定 | `null` |

**严格约束**：
- 这些旧字段虽然当前动态 DAG 没有 timing 数据，但模板中会直接读取；因此 adapter 必须显式填默认值。
- 不允许依赖前端 `undefined` 或“字段没给就当没有”的隐式路径；缺字段就是 adapter bug，应直接 raise。

### 3.3 Edge 映射规则

`_serialize_edge()` 当前输出字段为（`dag_serializer.py:99-122`）：
- `src_id`
- `dst_id`
- `is_containment`
- `tensor_info`
- `evidence`

建议映射到前端 edge 结构如下：

| SerializedDAG edge 字段 | flowchart_data edge 字段 | 映射规则 |
|---|---|---|
| `src_id` | `from` | 直接重命名 |
| `dst_id` | `to` | 直接重命名 |
| `is_containment` | `type` | `true -> "containment"`；`false -> "data"` |
| `tensor_info` | `tensor_info` | 直接透传 |
| `evidence` | `evidence` | `null` 保持 `null`；非空列表第一期直接透传 |
| （新增） | `from_attr` | 取源节点 `label` |
| （新增） | `to_attr` | 取目标节点 `label` |
| （新增） | `parent_class` | 若属于某 group 的内部边，则写该 group.label；否则 `""` |

这里要区分两类边：

1. **group 内部边**
   - 应放进对应 group 的 `internal_edges`
   - 供 `renderGroupAt()` 中 `for (const ed of g.internal_edges)` 使用（`frontend_html.py:971-985`）

2. **全局边**
   - 放进顶层 `flowchart_data.edges`
   - 供 `render()` 的全局 edge pass 使用（`frontend_html.py:1272-1279`）

建议划分原则：
- 如果一条 edge 的两端都属于同一个 group 的直属 child，则放入该 group 的 `internal_edges`
- 否则放到顶层 `edges`

**禁止静默跳过**：
- 遇到缺少 `src_id` / `dst_id` / `is_containment`：直接 `raise RuntimeError`
- 遇到未知 `is_containment` 类型：直接 `raise RuntimeError`
- 映射后若 `from` 或 `to` 在 node/group 索引中不存在：直接 `raise RuntimeError`
- 不允许返回 `None`、空 dict、或 `continue` 掩盖坏数据

关于 evidence：
- 目前 `_serialize_edge()` 的 `evidence` 不是旧前端完整结构，而是把 `var_evidence.steps` 展平成 `{loc, role, var}` 列表（`dag_serializer.py:100-111`）
- `showEdgePanel()` 旧逻辑更偏向读取 `ev.file / ev.var_history / ev.from_excerpt / ev.to_excerpt` 等字段（`frontend_html.py:1517-1614`）
- 因此第一阶段策略是：adapter 不做伪造补全，只做透传；前端后续单独兼容“list 型简化 evidence”显示即可
- 这里同样不能为了“先跑起来”去静默置空；有值就保留原值，没值就显式 `null`

## 4. adapter 函数设计

函数签名：

```python
def adapt_serialized_dag(serialized: dict, model_id: str, mode: str) -> dict:
    ...
```

输入：
- `serialize_dag()` 返回的 dict
- `model_id`: 当前模型 id
- `mode`: `training` 或 `inference`

输出：
- `render_flowchart_to_file()` 可以直接消费的 `flowchart_data` dict

建议实现要点：

1. **先做输入校验**
   - 校验 `serialized` 顶层必须包含：
     - `input_nodes`
     - `param_nodes`
     - `const_nodes`
     - `output_nodes`
     - `nodes`
     - `edges`
   - 任一缺失直接 `raise RuntimeError`

2. **建立节点索引**
   - 把四类 IO 节点按 `node_id` 建索引
   - 递归遍历 `serialized["nodes"]`，构建：
     - `all_groups_by_id`
     - `all_leaf_nodes_by_id`
     - `parent_group_of_child`
     - `label_by_node_id`
   - 遍历中顺带计算 `depth`

3. **建立 `root_groups`**
   - 从根层递归遍历结果中提取 `depth == 0` 的 group id 列表
   - 若根层没有 group，直接 `raise RuntimeError`；因为当前前端主布局是 `DATA.root_groups.map(...)`（`frontend_html.py:766-769`），空根组不是合法输入

4. **建立 `groups`**
   - 输出所有 group 的完整 dict
   - 每个 group 内同时填好：
     - `children_nodes`
     - `children_groups`
     - `internal_edges`
   - `children_groups` 必须是嵌套 group dict，而不是仅 id；因为 `indexGroupAncestors()` 直接递归读 `g.children_groups`（`frontend_html.py:456-464`）

5. **转换 `nodes`**
   - 输出所有 leaf module / functional 节点
   - 再把 input / param / const / output 四类 IO 节点补入 `nodes`
   - 保证 `DATA.nodes.forEach(n => nodeMap[n.id] = n)` 后，所有 IO pill 都能查到 node（`frontend_html.py:468-469`, `1213-1230`）

6. **转换 `edges`**
   - 字段重命名：`src_id -> from`, `dst_id -> to`
   - 类型转换：`is_containment -> type`
   - 補充 `from_attr` / `to_attr` / `parent_class`
   - 按“是否同属某 group 直属 child”决定归入 `group.internal_edges` 还是全局 `edges`
   - 任一 edge 校验失败直接 `raise RuntimeError`

7. **填 IO 集合字段**
   - `input_node_ids`
   - `param_node_ids`
   - `const_node_ids`
   - `output_node_ids`

8. **填 `meta` 与 `has_timing`**
   - `has_timing=False`
   - `meta` 至少包括：
     - `model_id`
     - `mode`
     - `time`
     - `num_modules`
     - `roots`

伪代码：

```python
def adapt_serialized_dag(serialized: dict, model_id: str, mode: str) -> dict:
    _validate_serialized_top_level(serialized)

    io_nodes = _collect_io_leaf_nodes(serialized)
    group_tree, leaf_nodes, indexes = _walk_serialized_nodes(serialized["nodes"], depth=0)
    root_groups = [g["id"] for g in group_tree]
    if not root_groups:
        raise RuntimeError("serialized DAG has no root groups")

    groups = _finalize_groups(group_tree, indexes)
    edges, groups = _route_edges_into_global_or_internal(
        serialized_edges=serialized["edges"],
        groups=groups,
        indexes=indexes,
        io_nodes=io_nodes,
    )

    nodes = list(leaf_nodes) + list(io_nodes)
    return {
        "input_node_ids": [...],
        "param_node_ids": [...],
        "const_node_ids": [...],
        "output_node_ids": [...],
        "root_groups": root_groups,
        "groups": groups,
        "nodes": nodes,
        "edges": edges,
        "meta": {...},
        "has_timing": False,
    }
```

## 5. run_dag_session.py 改动点

接入点在 `_run_single_mode()` 内，位置是：
- `registry, dag = session.finalize()` 之后（`run_dag_session.py:427`）
- DCE 完成之后（`429-430`）
- 统计 / validate 之前或之后都可，但建议 **在 DCE 之后、统计之前** 生成 HTML，这样 HTML 和后续 stats 使用的是同一份 DCE 后 DAG

建议伪代码：

```python
registry, dag = session.finalize()

# Step 5.5: DCE
_dce_warnings = _run_dce_and_warn(dag, registry)

# Step 5.6: HTML DAG
serialized = serialize_dag(dag, registry)
flowchart_data = adapt_serialized_dag(serialized, model_id=model_id, mode=mode_label)
html_latest_path = ast_dir / f"dag_{model_id}_{mode_label}_latest.html"
render_flowchart_to_file(flowchart_data, str(html_latest_path))
emit(f"[HTML] {html_latest_path}")

# Step 6: stats
...
```

说明：

1. **调用顺序必须固定**：
   - `serialize_dag`
   - `adapt_serialized_dag`
   - `render_flowchart_to_file`
2. **插入位置**：DCE 之后，避免把被裁掉的死节点再渲染进 HTML。
3. **输出路径**：
   - `develop/ast/dag_{model_id}_{mode}_latest.html`
   - 与现有 txt 报告 latest 命名风格保持一致；现有文本报告 latest 路径生成位于 `run_dag_session.py:488-492`
4. **文本报告同步记录 HTML 路径**：
   - 使用现有 `emit()` 输出一行 `[HTML] <path>`
   - 这样查看 txt 报告时可直接定位 HTML 产物
5. 如果后续需要保留时间戳版本，也可以仿照 `_get_report_path()` 再加：
   - `dag_{model_id}_{mode}_{run_ts}.html`
   - 但本设计第一期先只要求 latest 文件即可

## 6. frontend_html.py 改动点

本次不是重写模板，而是做 **最小必要改动**，让旧模板理解新的 IO 四分类顶层字段。前端改动范围只包括 6.1 ~ 6.3；其余模板结构、evidence panel、timing 展示字段均不在本次范围内。

### 6.1 `isIONodeId()`：`335-337`（本次做）
当前实现：
- 只识别 `DATA.input_node_id`
- 只识别 `DATA.loss_node_id`

需要改为识别四个集合：
- `DATA.input_node_ids`
- `DATA.param_node_ids`
- `DATA.const_node_ids`
- `DATA.output_node_ids`

建议改法：初始化四个 `Set`，`isIONodeId(nodeId)` 判断是否属于任一集合。旧的 `DATA.input_node_id` / `DATA.loss_node_id` 单值读取必须删除，不保留 shim / alias。

### 6.2 IO hover 逻辑：`335-401`（本次做）
`applyEdgeFocusState()` 当前通过 `isIONodeId()` 决定：
- hovered 节点是否进入 `ioFocusedNodes`
- active edge 两端是否有 IO 端点

因此这里不需要额外重写算法，只随 6.1 更新 IO 判定来源，并确保：
1. `isIONodeId()` 已经基于四集合实现
2. IO 相关节点都能在 `nodeDomRegistry` / `nodeMap` 中找到
3. output 不再专门绑定到旧的 `loss_node_id`

### 6.3 顶层 render 里的 IO pill：`1212-1268`（本次做）
当前只有两颗 pill：
- `Input`
- `Result`

且读取的是单值字段：
- `DATA.input_node_id`
- `DATA.loss_node_id`

需要改成：
1. 顶部按 `input_node_ids / param_node_ids / const_node_ids` 三类分别渲染
   - 可横向并排，或纵向堆叠后居中
2. 底部按 `output_node_ids` 渲染一组 output pills
3. label 建议：
   - input: `Input`
   - param: `Param`
   - const: `Const`
   - output: `Result`
4. sublabel 从对应 `nodeMap[nid]` 读取 `class_name` 或 `label`
5. 删除旧的 `DATA.input_node_id` / `DATA.loss_node_id` 单值读取

### 6.4 group/node map 初始化：`456-503`（not in scope）
这里现有逻辑仍然可复用：
- `DATA.groups.forEach(g => groupMap[g.id] = g)`
- `DATA.nodes.forEach(n => nodeMap[n.id] = n)`
- `indexGroupAncestors(DATA.root_groups.map(...))`

只要 adapter 输出满足：
- `groups` 是完整嵌套结构
- `root_groups` 是顶层 group id 列表
- `nodes` 含 IO 节点与普通 leaf
这里就不需要结构性重写。

### 6.5 group 渲染主逻辑：`755-879`（not in scope）
这里不需要大改模板算法，但 adapter 输出必须满足它的约束：
- root layout 基于 `DATA.root_groups`
- 每个 group 需要 `children_nodes` / `children_groups`
- timing 字段虽然当前无值，但字段必须存在

### 6.6 edge 映射：`971-983`（not in scope）
当前 group 内部 edge 读取字段：
- `from_child`
- `to_child`
- `type`
- `from_attr`
- `to_attr`
- `parent_class`
- `evidence`

因此 adapter 的 `internal_edges` 必须严格提供这套字段，不能只给 `from` / `to`。

### 6.7 顶层 render / meta / has_timing：`1212-1325`（not in scope）
这里的 header、legend、summary 基本可复用；本次不删除 timing 字段，adapter 必须补齐默认值：
- `meta.num_modules`
- `meta.roots`
- `has_timing=False`
- node/group 上的 `pct=0`、`dur_us=0`、`kernel_us=0`、`role=null`

### 6.8 edge panel / evidence：`1517-1614`（not in scope）
evidence panel 重写不在本次范围内。本次保持现有 panel 结构，adapter 只保留 evidence 的实际语义，不伪造旧 evidence 对象，也不在前端引入结构性重写。

### 6.9 JSON 注入：`1885-1986`（not in scope）
这里不需要改 Python 端注入机制；只要 adapter 生成的 `flowchart_data` 字段完整，`_generate_flowchart_html()` 仍可直接用 `const DATA = ...` 注入。

## 7. dag_serializer.py 改动点（如有）
本次第一阶段 **可以不改** `dag_serializer.py`，adapter 自己补字段即可。

原因：

1. `dag_serializer.py` 当前已经提供了 adapter 必需的最核心信息：
   - IO 节点的 `node_id / io_subtype / label / call_loc`（`dag_serializer.py:22-47`）
   - module 节点的 `node_id / label / call_loc / attr_type / inner_dag`（`50-96`）
   - edge 的 `src_id / dst_id / is_containment / tensor_info / evidence`（`99-122`）
2. `depth`、前端旧 timing 缺省字段、`node_type`、`from_attr`、`to_attr` 这些都属于 **前端适配视图字段**，更适合在 adapter 层补。
3. 若后续发现 adapter 无法可靠区分某些节点类型，再考虑给 serializer 增量补字段；但第一期不应为了“可能以后要用”提前扩字段。

可选增强项（非本次必须）：
- 若后续希望 adapter 更少依赖上下文，可考虑在 serializer 里显式加：
  - `node_kind`
  - `depth`（如果 DAG 本身后续带层级信息）
- 但当前设计不依赖这些字段，因此先不动 serializer。

## 8. 新增文件清单

| 文件 | 说明 |
|---|---|
| `scripts/dag_html_adapter.py` | `SerializedDAG -> flowchart_data` 转换逻辑 |
| `testset/unit/test_dag_html_adapter.py` | adapter 对应 UT，放在 storage 仓库 |

UT 建议覆盖的最小场景：
1. 只有单 root group + input/output 的最小 DAG
2. 含 param / const / output 四类 IO 的顶层字段映射
3. 含嵌套 `inner_dag` 的 group / leaf 拆分
4. containment edge / data edge 类型映射
5. edge 缺字段时报 `RuntimeError`
6. 无 root group 时报 `RuntimeError`

## 9. 不在本次范围内
本次设计明确 **不做** 以下内容：

1. evidence 完整字段补全
   - 第一阶段 evidence 只透传，不把简化 list 强行补造成旧对象结构
2. `analyze_trace.py` CLI 重新接入
   - 本次只打通 `run_dag_session.py -> HTML` 动态 DAG 链路
3. 前端模板重写
   - 仅做最小兼容改动，不改整体布局框架
4. timing 数据接入
   - `has_timing` 固定 `False`，所有 timing 相关字段用显式默认值
