# Canvas Phase 1 — Stage 1.1 ~ 1.4 实施设计（含 UT 详细设计 / 旧功能替换 / 清理计划）

> 状态：**设计待 NaN review**，review 通过后才开始实现。
>
> 上位文档：`design/frontend_canvas_phase1.md`
>
> 本文只回答四件事：
> 1. 每个 Stage 到底改哪些文件、删哪些旧路径；
> 2. 每个 Stage 旧功能如何替换，哪些保持等价、哪些明确下线；
> 3. 每个 Stage 对应哪些新增 / 改写 / skip / 删除的 UT；
> 4. 每个 Stage 如何证明“功能等价而不是视觉碰运气”。

---

## 0. 当前基线（源码证据）

### 0.1 前端实现基线

- `scripts/frontend_html.py:234-236`
  ```html
  <div class="dag-container" id="dag-container">
      <svg class="dag-svg" id="dag-svg"></svg>
  </div>
  ```
- `scripts/frontend_html.py:3064-3071`
  ```python
  render_group_js = _RENDER_GROUP_JS_PATH.read_text(...)
  if _RENDER_GROUP_JS_PLACEHOLDER not in html_template:
      raise RuntimeError(...)
  html_template = html_template.replace(_RENDER_GROUP_JS_PLACEHOLDER, render_group_js, 1)
  ```
- `scripts/render_group.js:21-172` 只负责 group 绘制 helper：
  - `renderCollapsedGroupBox`
  - `renderCollapsedGroupLabel`
  - `renderCollapsedGroupTiming`
  - `renderCollapsedGroupInfoButton`
  - `renderExpandedGroupBox`
  - `renderExpandedGroupHeaderLabel`
  - `renderExpandedGroupInfoButton`
  - `renderExpandedGroupTiming`
  - `renderGroupChildren`
  - `renderGroupAt`
- `scripts/frontend_html.py` 仍承担：
  - 布局：`computeRanks` / `orderRanks` / `layoutGroup`
  - 节点绘制：`renderNodeAt`
  - edge 绘制：`buildDirectEdgePath` / `buildIntraGroupEdgePath` / `renderEdge`
  - 交互：hover/click/toggle/focus/overlay 逻辑

### 0.2 UT 基线

#### Python UT（`test_frontend_html.py`）

- A 类：10 个，保留并改 probe
- D 类：1 个，重写
- B/C 类：30 个
  - B：16 个 SVG/CSS/DOM 强绑定测试
  - C：14 个 hover/click/active 行为测试

#### 其他前端 UT

- `test_frontend_redirect.py`：8 个，全部保留
- `test_edge_routing.js`：3 个，保留
- `test_render_group_helpers.js`：6 个，Phase 1 内删除或改写后删除
- `test_hover_index.js`：11 个，Phase 1 临时 skip，Phase 3 重写恢复

---

## 1. Stage 总览

| Stage | 核心目标 | 主要文件 | 旧实现清理动作 | 核心 UT |
| --- | --- | --- | --- | --- |
| 1.1 | 模板切到 Canvas + 引擎骨架 | `frontend_html.py`、新增 `render_canvas.js` | 停止运行时注入 `render_group.js`；B/C 30 个 Python UT 先分组 skip | P1-1 / P1-2 / P1-3 / P1-10 |
| 1.2 | 节点/group/port 静态绘制 + helper 基建迁移 | `render_canvas.js`、`frontend_test_infra.py`、`test_frontend_html.py` | 删除 `test_render_group_helpers.js`；删除 `render_group.js`；清掉 `test_frontend_html.py` 里重复 helper | P1-4 / P1-5 / P1-7 / P1-9 / P1-14 / P1-15 |
| 1.3 | edge/arrow 批绘 | `render_canvas.js`、`frontend_html.py`、`test_frontend_redirect.py` | 删除 SVG edge path / hitbox 生成路径；D 类 UT 重写 | P1-6 / P1-16 / P1-E1 / P1-E2 / P1-E3 |
| 1.4 | IO/culling/progress/pan-zoom-fit + Phase 1 收尾 | `render_canvas.js`、`frontend_html.py` | 删除 B 类 SVG 强绑定测试；保留 C 类但统一 skip 到 Phase 3 | P1-8 / P1-11 / P1-12 / P1-13 / P1-17 / P1-18 |

---

## 2. 测试总策略

## 2.1 新 probe 统一入口

新增：`window.__renderSnapshot()`，只读返回：

```javascript
{
  nodes: [{ id, x, y, w, h, color, label, sublabel, visible }],
  groups: [{ id, collapsed, x, y, w, h, has_header, has_info, has_timing }],
  edges: [{ key, from, to, type, points, has_arrow, bucket }],
  ports: { [nodeId]: { cx, cy } },
  io_pills: [{ id, collapsed, x, y, w, h, label }],
  viewport: { scale, x, y, worldWidth, worldHeight },
  layers: { l0, l1, l2, l3, l4, l5 },
  flags: { noInteractionMode, cullingEnabled }
}
```

### 设计要求

- 只读快照，不得反向修改运行时状态
- 正常浏览器路径与 headless mock 路径都返回同结构
- 任何字段缺失都算 bug，不做 optional fallback

## 2.2 新 helper 基建

新增 `testset/unit/frontend_test_infra.py`，只承载前端 UT helper，不混入 `dag_session_test_infra.py`。

### 从 `test_frontend_html.py` 迁入的现有 helper

- `_make_node`
- `_make_group`
- `render_minimal_flowchart_to_string`
- `_extract_script`
- `_patch_script_for_node`
- `_run_render_probe`

### 新增 helper

- `_build_canvas_dom_stub()`
- `_run_canvas_render_probe(flowchart, probe_js)`
- `_make_standard_flowchart_data(...)`
- `_assert_snapshot_layers(snapshot, expected)`

### 迁移原则

- `test_frontend_html.py` 中不再保留任何重复 helper 定义
- 所有前端 probe 测试都从 `frontend_test_infra.py` import
- helper 迁移后，原文件里若还有副本，必须删除，不允许双份维护

## 2.3 旧测试处理矩阵

| 类别 | 当前来源 | Phase 1 处理方式 |
| --- | --- | --- |
| 纯数据/布局 A 类 | `test_frontend_html.py` | 保留，SVG DOM 探针改为 snapshot |
| redirect 语义 | `test_frontend_redirect.py`、`test_edge_routing.js` | 保留 |
| SVG helper 结构测试 | `test_render_group_helpers.js` | 用 P1-4/P1-5/P1-7/P1-14/P1-15 覆盖后删除 |
| SVG/CSS/DOM 强绑定 B 类 | `test_frontend_html.py` | Stage 1.1 临时 skip，Stage 1.4 删除 |
| 交互 C 类 | `test_frontend_html.py`、`test_hover_index.js` | 整体 skip 到 Phase 3，保留语义，不删除 |
| D 类 edge index globals | `test_frontend_html.py` | Stage 1.3 改写为 snapshot edge/layer 断言 |

---

## 3. Stage 1.1 — 模板 & 引擎基建

## 3.1 设计目标

只做“渲染容器和脚本注入链路切换”，不画任何真实图元。

## 3.2 改动文件

- `scripts/frontend_html.py`
- 新增 `scripts/render_canvas.js`
- `testset/unit/test_frontend_canvas_phase1.py`

## 3.3 旧功能替换

| 旧实现 | 新实现 | 备注 |
| --- | --- | --- |
| `<svg id="dag-svg">` | `<div id="dag-stage">` | 必须唯一；缺失即抛错 |
| `_RENDER_GROUP_JS_PLACEHOLDER` 注入 | `__ENGINE_BUNDLE_PLACEHOLDER__` + `__RENDER_CANVAS_JS_PLACEHOLDER__` | 新旧占位符同时存在时，旧的只能替空串 |
| `.dag-svg ...` CSS | 删除 | DOM 浮层 CSS 保留 |
| 无 Canvas 根对象 | `PIXI.Application` + `Viewport` + L0~L5 | 此阶段 layers 只建空容器 |

## 3.4 旧实现清理动作

1. 运行时不再注入 `render_group.js`
2. `render_group.js` 文件暂时还在仓库里，但**没有任何运行时入口**
3. `test_frontend_html.py` 的 B/C 30 个测试先加 skip，避免模板切换后无意义红灯
4. 不改 A 类 / redirect 测试

## 3.5 功能等价性保证

- 数据等价：`const DATA = ...` 注入逻辑不变
- 布局等价：布局函数源码不改
- DOM 浮层等价：`tooltip` / `side-panel` / `summary` / `legend` / `progress overlay` 节点仍存在
- 行为约束：此阶段 `window.__phase1NoInteractionMode === true`

## 3.6 UT 详细设计

### 新增测例

| ID | 测例 | 输入 | 断言 |
| --- | --- | --- | --- |
| P1-1 | `test_canvas_stage_div_replaces_svg` | 最小 flowchart | HTML 不含 `<svg id="dag-svg"`；含 `<div id="dag-stage"` |
| P1-2 | `test_engine_and_canvas_js_placeholders_replaced` | 最小 flowchart | HTML 不含两个新占位符字面量；包含 Pixi bundle / `render_canvas.js` 标识 |
| P1-3 | `test_engine_placeholder_missing_raises` | 人工构造模板缺占位符 | `_generate_flowchart_html()` 直接抛 `RuntimeError` |
| P1-10 | `test_phase1_no_hover_listeners_attached` | 最小 flowchart | `__renderSnapshot().flags.noInteractionMode === true`；L4 overlay 为空 |

### 现有测试处理

#### B 类 16 个（Stage 1.1 开始 skip）

- `test_html_emits_local_hover_css`
- `test_hover_no_svg_has_hover_global_rule`
- `test_hover_local_dim_css_present`
- `test_edge_hover_no_svg_has_hover_toggle`
- `test_edge_has_hitbox_path_with_data_edge_key`
- `test_edge_hover_uses_elements_from_point_not_isPointInStroke`
- `test_edge_active_classlist_present`
- `test_node_active_classlist_present`
- `test_edge_active_css_class_rules_present`
- `test_edge_active_css_colors_present`
- `test_inline_style_focus_mutation_removed`
- `test_hover_active_container_toggle_present`
- `test_mousemove_does_not_call_elements_from_point`
- `test_mouseenter_calls_elements_from_point_once`
- `test_edge_hover_does_not_use_append_child_for_zorder`
- `test_active_edge_overlay_layer_exists`

#### C 类 14 个（Stage 1.1 开始 skip）

- `test_html_applyEdgeFocusState_no_full_registry_scan`
- `test_overlap_scroll_updates_active_edge_overlay`
- `test_overlap_click_hits_active_path_after_scroll`
- `test_edge_hover_leave_clears_highlight_when_group_hovered`
- `test_p0_group_hover_uses_edge_by_group_index`
- `test_first_group_hover_no_full_scan`
- `test_edge_hover_path_no_full_registry_iteration`
- `test_p1_group_hover_uses_delta_update`
- `test_p2_same_gid_noop_guard`
- `test_group_hover_uses_edge_by_group_id`
- `test_register_edge_dom_fills_edge_by_group_id_via_ancestor_groups`
- `test_edge_gid_field_not_used_for_group_index`
- `test_edge_by_group_id_filled_for_both_endpoints`
- `test_edge_by_group_id_cleared_on_re_render`

## 3.7 Stage 1.1 验收门槛

- 只允许模板 / 注入 / Canvas 骨架相关改动
- 布局函数 diff = 0
- P1-1 / P1-2 / P1-3 / P1-10 PASS

---

## 4. Stage 1.2 — Node / Group / Port 静态绘制

## 4.1 设计目标

把 `renderNodeAt` 与 `render_group.js` 的图形职责迁到 Canvas，并把前端 probe helper 收敛到 `frontend_test_infra.py`。

## 4.2 改动文件

- `scripts/render_canvas.js`
- `testset/unit/frontend_test_infra.py`
- `testset/unit/test_frontend_canvas_phase1.py`
- `testset/unit/test_frontend_html.py`
- 删除：`testset/unit/frontend/test_render_group_helpers.js`
- 删除：`scripts/render_group.js`

## 4.3 旧功能替换

| 旧函数 / 文件 | 新函数 / 文件 | 等价定义 |
| --- | --- | --- |
| `renderNodeAt()` | `NodeView.draw()` | 输出相同 `(x,y,w,h)` 和颜色/label 语义 |
| `renderCollapsedGroupBox/Label/Timing/Info` | `GroupView.drawCollapsed()` | collapsed group 仍只表现为一个 box |
| `renderExpandedGroupBox/Header/Timing/Info` | `GroupView.drawExpanded()` | header/info/timing 语义不丢 |
| `registerCollapsedGroupPorts/registerExpandedGroupPorts` | `PortRenderer.drawPorts()` | `nodePortMap` key/value 完全相同 |
| `test_render_group_helpers.js` | P1-4/P1-5/P1-7/P1-14/P1-15 | 从 DOM 结构断言改成 snapshot 语义断言 |

## 4.4 旧实现清理动作

1. 新增 `frontend_test_infra.py`
2. `test_frontend_html.py` 删除重复 helper 定义，全部改为 import
3. `test_render_group_helpers.js` 删除
4. `scripts/render_group.js` 删除
5. 所有 group 绘制逻辑只剩 `render_canvas.js` 一份实现

## 4.5 功能等价性保证

- Group / node 坐标全部来自原布局结果，不在绘制阶段重算
- `snapshot.ports` 与 `window.nodePortMap` 逐项相等
- collapsed group 内部 leaf 不应出现在 `snapshot.nodes`
- expanded group 的 `has_header` / `has_info` / `has_timing` 与数据条件一致

## 4.6 UT 详细设计

| ID | 测例 | 输入 | 断言 |
| --- | --- | --- | --- |
| P1-4 | `test_render_snapshot_node_count_matches_data` | 3 group + 5 leaf | `snapshot.nodes` 长度 == 5 |
| P1-5 | `test_render_snapshot_group_count_matches_data` | 同上 | `snapshot.groups` 长度 == 3；collapsed 字段对齐 |
| P1-7 | `test_render_snapshot_ports_equal_node_port_map` | 1 展开 group + 1 折叠 group + 2 leaf | `snapshot.ports` keys 与 `nodePortMap` keys 完全一致；值逐项相等 |
| P1-9 | `test_render_snapshot_layer_children_only_l0_l1_l2_l3_l5_populated` | 同上 | `l2>0`、`l3>0`、`l1==0`、`l4==0` |
| P1-14 | `test_collapsed_group_renders_as_single_box` | 1 collapsed group 含 3 leaf | group 存在；3 个内部 leaf 不在 `snapshot.nodes` |
| P1-15 | `test_expanded_group_renders_header_and_info_circle` | 1 expanded group 且有源码定位 | `has_header==true`、`has_info==true` |

### A 类旧测试改写（Stage 1.2）

以下测试保留，但 probe 从 SVG DOM 改为 snapshot：

- `test_expanded_group_registers_in_port_node_id_in_nodemap`
- `test_expanded_group_registers_out_port_node_id_in_nodemap`
- `test_global_edge_from_port_node_id_resolves_without_missing_error`
- `test_global_edge_to_collapsed_group_redirects_correctly`
- `test_global_edge_endpoint_from_collapsed_group_port_node_id_resolves_without_missing_error`
- `test_global_edge_endpoint_from_port_node_id_inside_collapsed_ancestor_resolves_without_missing_error`
- `test_port_node_ids_indexed_in_ancestor_groups`

## 4.7 Stage 1.2 验收门槛

- `render_group.js` 已删除
- `test_render_group_helpers.js` 已删除
- `frontend_test_infra.py` 成为唯一前端 helper 来源
- P1-4 / P1-5 / P1-7 / P1-9 / P1-14 / P1-15 PASS

---

## 5. Stage 1.3 — Edge / Arrow / 批绘

## 5.1 设计目标

把 SVG edge path 路径和 hitbox 路径从生产代码中清掉，用点列 + bucket Graphics 替换。

## 5.2 改动文件

- `scripts/render_canvas.js`
- `scripts/frontend_html.py`
- `testset/unit/test_frontend_canvas_phase1.py`
- `testset/unit/test_frontend_html.py`
- `testset/unit/test_frontend_redirect.py`（仅当其探针要配合 snapshot）

## 5.3 旧功能替换

| 旧实现 | 新实现 | 等价定义 |
| --- | --- | --- |
| `buildDirectEdgePath()` | `EdgeRoute.buildDirect()` | 几何控制点等价 |
| `buildIntraGroupEdgePath()` | `EdgeRoute.buildIntraGroup()` | lane/gutter 路由等价 |
| 每边一个 `<path>` | 每 type 一个 bucket Graphics | 最终可见边数等价 |
| `<marker>` 箭头 | 终点三角形 | 每条边 `has_arrow` 为 true |
| hitbox path | Phase 1 不存在 | Phase 3 再引入空间索引命中 |

## 5.4 旧实现清理动作

1. 删除 SVG edge path 生成与 DOM append 路径
2. 删除 hitbox path 生成路径
3. `test_html_emits_edge_index_globals` 改写为 snapshot edge/layer 断言
4. `test_edge_routing.js` 保留，不受 SVG 清理影响

## 5.5 功能等价性保证

- `snapshot.edges.length === DATA.edges.length`
- 直接边 / 跨层边几何与旧 SVG route 一致（容差 0.5px）
- bucket 划分只影响绘制批次，不影响可见边集合
- redirect 后边端点解析仍遵循原 `resolveCollapsedAncestor` 语义

## 5.6 UT 详细设计

| ID | 测例 | 输入 | 断言 |
| --- | --- | --- | --- |
| P1-6 | `test_render_snapshot_edge_count_matches_data` | 含 redirect 边的 mock | `snapshot.edges.length == len(DATA.edges)` |
| P1-16 | `test_edge_arrow_terminal_present` | 1 直接边 + 1 跨层边 | 两条边 `has_arrow == true` |
| P1-E1 | `test_edge_batch_groups_by_type` | 2 dep + 1 flow + 1 internal | `snapshot.layers.l1 == 3`，每 bucket 数正确 |
| P1-E2 | `test_edge_route_direct_bezier_matches_svg_geometry` | 固定 port 坐标直接边 | 点列与旧 route 一致，容差 0.5px |
| P1-E3 | `test_edge_route_intra_group_polyline_matches_svg_geometry` | 固定 port 坐标跨层边 | 点列与旧 route 一致，容差 0.5px |

### D 类旧测试改写

- `test_html_emits_edge_index_globals`
  - 旧断言：检查旧全局变量注入
  - 新断言：`__renderSnapshot().edges` 可用，且 `l1>0`

## 5.7 Stage 1.3 验收门槛

- 生产代码里不再创建 `edge-path` / hitbox DOM
- P1-6 / P1-16 / P1-E1 / P1-E2 / P1-E3 PASS
- redirect 相关旧测试无回退

---

## 6. Stage 1.4 — IO / Culling / Progress / Pan-Zoom-Fit / Phase 1 收尾

## 6.1 设计目标

补齐 Phase 1 最后一个“可看不可动”版本：IO pill、视口裁剪、Fit、progress overlay 三阶段，并把 SVG 专属测试做最终清理。

## 6.2 改动文件

- `scripts/render_canvas.js`
- `scripts/frontend_html.py`
- `testset/unit/test_frontend_canvas_phase1.py`
- `testset/unit/test_frontend_html.py`
- `testset/unit/frontend/test_hover_index.js`（加 suite skip）

## 6.3 旧功能替换

| 旧实现 | 新实现 | 等价定义 |
| --- | --- | --- |
| `renderIOGroupPill()` | `IOLayer.drawPill()` | 数量与 collapsed 状态等价 |
| `btn-fit` + CSS transform | Viewport fit | 几何不重算，只变 viewport transform |
| SVG label 常驻 | Culling 后视口外 label 不创建 | 节点不丢，只是 `label=null` |
| 旧 progress 文案 | 三段式中文 progress 文案 | 顺序、单调性可断言 |

## 6.4 旧实现清理动作

1. 删除 B 类 16 个 SVG 强绑定 Python 测试
2. C 类 14 个交互 Python 测试统一保留并 skip 到 Phase 3
3. `test_hover_index.js` 整个 suite skip 到 Phase 3
4. Phase 1 收尾时检查：运行时 SVG 渲染代码已完全不存在

## 6.5 功能等价性保证

- IO pill 数和 collapsed 状态与 `DATA.io_groups` 等价
- Culling 只影响文本创建，不影响 node/group/edge 数量
- Pan/Zoom/Fit 只改 viewport transform，不改布局产物
- `groupLayout` / `nodePortMap` 在“只跑布局”和“跑 Canvas 渲染管线”两条路径下深度相等

## 6.6 UT 详细设计

| ID | 测例 | 输入 | 断言 |
| --- | --- | --- | --- |
| P1-8 | `test_render_snapshot_io_pills_match_io_groups` | 2 个 IO group | 数量一致；collapsed 对齐 |
| P1-11 | `test_phase1_pan_zoom_updates_viewport_transform_not_geometry` | 标准 mock | zoom 后 nodes/edges 坐标不变；viewport.scale 变化 |
| P1-12 | `test_phase1_fit_to_view_button_invokes_viewport_fit` | 标准 mock | fit 后 viewport scale/position 符合整图入框 |
| P1-13 | `test_layout_outputs_unchanged_by_canvas_switch` | 同一 DATA 跑两遍 | `groupLayout` / `nodePortMap` 深度相等 |
| P1-17 | `test_culling_offscreen_text_not_created` | 视口外 1 节点 | 该节点 `label==null`；视口内节点 label 非空 |
| P1-18 | `test_progress_callback_phases_canvas_pipeline` | 标准 mock | progress 文案包含三阶段，百分比单调递增到 100 |

## 6.7 Stage 1.4 验收门槛

- B 类 SVG 测试已删除
- C 类交互测试已成体系 skip，并带明确 phase3 reason
- `test_hover_index.js` 已 skip
- 全部 P1-* + P1-E* PASS

---

## 7. Phase 1 结束时的文件状态

### 7.1 研发仓库

| 文件 | 状态 |
| --- | --- |
| `scripts/frontend_html.py` | 保留，生产渲染路径改为 Canvas |
| `scripts/render_canvas.js` | 新增并成为唯一渲染实现 |
| `scripts/render_group.js` | 删除 |
| `scripts/dag_html_adapter.py` | 0 改动 |

### 7.2 测试仓库

| 文件 | 状态 |
| --- | --- |
| `testset/unit/frontend_test_infra.py` | 新增 |
| `testset/unit/test_frontend_canvas_phase1.py` | 新增 |
| `testset/unit/test_frontend_html.py` | 保留并瘦身 |
| `testset/unit/test_frontend_redirect.py` | 保留 |
| `testset/unit/frontend/test_edge_routing.js` | 保留 |
| `testset/unit/frontend/test_render_group_helpers.js` | 删除 |
| `testset/unit/frontend/test_hover_index.js` | 保留但 skip 到 Phase 3 |

---

## 8. 每阶段覆盖率要求

### 8.1 阶段内规则

每个 Stage 完成后都要满足：

1. 当阶段新增/改写 UT 全 PASS
2. 当阶段新增代码 line coverage ≥ 90%
3. 当阶段新增代码 branch coverage ≥ 50%
4. 前序 Stage UT 不回退

### 8.2 Phase 1 回归命令

Python：

```bash
python3 -m pytest \
  testset/unit/test_frontend_canvas_phase1.py \
  testset/unit/test_frontend_html.py \
  testset/unit/test_frontend_redirect.py -v
```

JS：

- 保留并跑：`test_edge_routing.js`
- skip：`test_hover_index.js`
- 删除后不再跑：`test_render_group_helpers.js`

---

## 9. 待 NaN review

1. **Stage 1.2 直接删 `render_group.js`** 是否接受？从代码职责看它只负责 group 绘制，迁完后没有保留价值。
2. **B 类测试删除、C 类测试 skip** 的清理策略是否接受？
3. **`test_edge_routing.js` 保留** 是否接受？它测的是 routing 语义，不应和 SVG DOM 一起删除。
4. **以当前源码为准修正测试数量**：B/C 共 30 个，而不是旧文档里的 27 个，是否按这个清单执行？
5. **Stage 1.1 先 skip、Stage 1.4 再删 B 类测试** 的节奏是否接受？这样可以避免 Stage 1.1 刚切模板时同时大量改测试，降低首个提交风险。

> 以上五点确认后，我下一步就按这个 Stage 方案进入实现，不再重复改设计口径。