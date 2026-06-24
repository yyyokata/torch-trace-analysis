# Stage 1.2 — Node / Group / Port 静态绘制 + 前端 UT infra 收敛（AS-BUILT 回顾 + 等价性说明）

> 状态：**已实现并合入 `feat/canvas-renderer`**
> 研发分支 commit：`b066d13 refactor(canvas): migrate node/group/port draw to render_canvas.js and remove render_group.js`
> 测试分支 commit：`300abd6 test(canvas): centralize front-end probe helpers and migrate A-class tests to snapshot`
>
> 上位文档：`design/frontend_canvas_phase1.md`、`design/frontend_canvas_phase1_stages.md`
>
> 本文是 Stage 1.2 的 **as-built** 复盘。重点：节点/容器/端口绘制如何从 SVG 迁到 Canvas，如何证明坐标/端口 bit-for-bit 等价，以及 `render_group.js` 为何可以直接删除。

---

## 1. 设计目标（本阶段范围）

把 `renderNodeAt` 与 `render_group.js` 的图形职责迁到 `render_canvas.js`，并把前端 probe helper 收敛到 `frontend_test_infra.py`：

- `NodeView`：leaf node 矩形 + 主/副 label
- `GroupView`：collapsed / expanded box、header、timing、info 圆点占位
- `PortRenderer`：端口写入共享 `nodePortMap`
- `window.__renderSnapshot()` 开始返回 nodes / groups / ports / layers
- 删除 `render_group.js` 与 `test_render_group_helpers.js`

**关键约束**：坐标全部来自原布局结果（`layoutGroup` 产出的 `groupLayout` / `childPositions`），绘制阶段**绝不重算坐标**。

---

## 2. 现状基线（源码证据）

| 事实 | 位置 |
| --- | --- |
| `drawNode()`（=旧 `renderNodeAt`）：写 L3 box+label[+sublabel]，push `engine.nodes`，注册端口 | `scripts/render_canvas.js:219-239` |
| `drawCollapsedGroup()`：L2 box + `▶ class_name` label + timing + info，push `engine.groups`(`has_header:false`) | `scripts/render_canvas.js:242-261` |
| `drawExpandedGroupShell()`：L2 box + `▼` header + info + timing，`has_header:true` | `scripts/render_canvas.js:263-281` |
| `walkGroup()`：按 `pos.collapsed` 分流，递归 `childPositions`（type=node/group），未知 type 抛错 | `scripts/render_canvas.js:284-312` |
| `registerNodePorts / registerCollapsedGroupPorts / registerExpandedGroupPorts`：写 `nodePortMap`（含 `__in/__out/__center` 与 port.node_id） | `scripts/render_canvas.js:187-216` |
| `layoutAndDrawRoots()`：调用**未改动**的 `layoutGroup(rid)`，复用其 `w/h/childPositions` | `scripts/render_canvas.js:333-350` |
| snapshot 输出 nodes/groups/ports/layers（edges/io_pills 仍空） | `scripts/render_canvas.js:384-412` |
| `render_group.js` 已从 `scripts/` 删除 | 文件不存在 |

---

## 3. 旧功能替换映射

| 旧函数 / 文件 | 新函数（render_canvas.js） | 等价定义 |
| --- | --- | --- |
| `renderNodeAt()` | `drawNode()` / NodeView | 相同 `(x,y,w,h)`；颜色走 `getNodeColor`；label=`class_name`，sublabel=`pct%`（有 timing 时） |
| `renderCollapsedGroupBox/Label/Timing/Info` | `drawCollapsedGroup()` | collapsed group 仍只表现为单个 box；timing/info 显隐条件不变 |
| `renderExpandedGroupBox/Header/Timing/Info` | `drawExpandedGroupShell()` | header/info/timing 语义不丢；`has_header=true` |
| `registerCollapsedGroupPorts/registerExpandedGroupPorts` | 同名函数（写 `nodePortMap`） | `nodePortMap` key/value 与旧实现完全相同 |
| `renderGroupAt` 递归 | `walkGroup()` | 递归顺序与 collapsed 分流一致 |
| `render_group.js`（整文件） | 已删除 | 所有 group 绘制只剩 `render_canvas.js` 一份 |
| `test_render_group_helpers.js` | P1-4/P1-5/P1-7/P1-14/P1-15 | 从 `<rect>/<text>` DOM 断言改为 snapshot 语义断言 |

---

## 4. 功能等价性保证（本阶段验证什么）

| 等价维度 | 保证方式 | 对应 UT |
| --- | --- | --- |
| 结构等价：节点数 | snapshot.nodes 数 == 可见叶子数；折叠 group 内 leaf 不出现 | P1-4 / P1-14 |
| 结构等价：group 数与折叠态 | snapshot.groups 数 == group 数；`collapsed` 与 inline `collapsedState` 对齐 | P1-5 |
| 坐标等价：端口 | `snapshot.ports` 的 key 集合与值（cx/cy）与 `window.nodePortMap` **逐项相等** | P1-7 |
| 分层等价 | 只有 L2（group）/L3（node）有子节点；L1（edge）/L4（overlay）为空 | P1-9 |
| 视觉参数等价 | `nodeW/nodeH`、圆角、颜色、label 截断规则不变（沿用 LAYOUT 与 getNodeColor） | P1-4/P1-15 间接覆盖 |
| 布局不重算 | 坐标来自未改动的 `layoutGroup`；绘制只消费 `childPositions` | 设计约束 + P1-13（Stage 1.4） |

> 「端口 bit-for-bit 相等」是 Stage 1.2 最强的等价证据：它直接保证 Stage 1.3 的全局边端点解析（`nodePortMap[id+'__out']` 等）几何不变。

---

## 5. UT 详细设计（已落地，全部 PASS）

| ID | 测例 | 输入 fixture | 断言 |
| --- | --- | --- | --- |
| P1-4 | `test_render_snapshot_node_count_matches_data` | 3 expanded group + 5 leaf | `snapshot.nodes.length == 5`，id 集合 == [21,22,31,32,33] |
| P1-5 | `test_render_snapshot_group_count_matches_data` | 同上 | `snapshot.groups.length == 3`；每个 group `collapsed` == inline `collapsedState`；depth 0/1 全展开 |
| P1-7 | `test_render_snapshot_ports_equal_node_port_map` | 1 展开 group + 1 折叠子 group(带 in/out port) + 2 leaf | `snapshot.ports` keys == `nodePortMap` keys；值逐项相等；含 `2__in/2__out/2__center/210/211` |
| P1-9 | `test_render_snapshot_layer_children_only_l2_l3_populated` | 同 P1-7 fixture | `l2>0`、`l3>0`、`l1==0`、`l4==0` |
| P1-14 | `test_collapsed_group_renders_as_single_box` | 1 折叠 root group(depth≥2) 含 3 leaf | `snapshot.groups == [{id:1,collapsed:true}]`；3 个内部 leaf 不在 `snapshot.nodes` |
| P1-15 | `test_expanded_group_renders_header_and_info_circle` | 1 expanded group 有源码定位 | `collapsed=false`、`has_header=true`、`has_info=true` |

### A 类旧测试改写（已执行）

以下保留语义，探针从 SVG DOM 改为 `__renderSnapshot()` / `nodePortMap`，均迁移到从 `frontend_test_infra` import 的统一 helper：

- `test_expanded_group_registers_in_port_node_id_in_nodemap`
- `test_expanded_group_registers_out_port_node_id_in_nodemap`
- `test_global_edge_from_port_node_id_resolves_without_missing_error`
- `test_global_edge_to_collapsed_group_redirects_correctly`
- `test_global_edge_endpoint_from_collapsed_group_port_node_id_resolves_without_missing_error`
- `test_global_edge_endpoint_from_port_node_id_inside_collapsed_ancestor_resolves_without_missing_error`
- `test_port_node_ids_indexed_in_ancestor_groups`

---

## 6. 旧 UT / 旧实现清理动作（已执行）

1. 新增 `testset/unit/frontend_test_infra.py`，承载全部前端 probe helper（`_make_node` / `_make_group` / `minimal_flowchart_data` / `render_minimal_flowchart_to_string` / `run_canvas_render_probe` / `run_canvas_snapshot_probe` / DOM stub）。
2. `test_frontend_html.py` 删除重复 helper，改为 import；无双份维护。
3. 删除 `testset/unit/frontend/test_render_group_helpers.js`。
4. 删除 `scripts/render_group.js`。

---

## 7. 验收门槛（已满足）

- `render_group.js`、`test_render_group_helpers.js` 已删除。
- `frontend_test_infra.py` 成为唯一前端 helper 来源。
- P1-4 / P1-5 / P1-7 / P1-9 / P1-14 / P1-15 PASS。
- `nodePortMap` 与 snapshot 端口逐值相等。
