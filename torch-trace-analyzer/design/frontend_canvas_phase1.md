# Canvas/WebGL 渲染 — Phase 1 详细设计（静态渲染 + 旧实现/旧 UT 清理）

> 状态：**设计待 NaN review，不含任何实现代码**。
> 
> 上位设计：`design/frontend_canvas_renderer.md`
> 
> 本文只覆盖 Phase 1（Stage 1.1 ~ 1.4）。重点不是“把 Canvas 先糊出来”，而是：
> 1. 在**不动布局算法 / 不动后端 JSON** 的前提下切换渲染后端；
> 2. 每个 Stage 都有明确的 **UT 入口、覆盖率门槛、旧 UT 清理动作、旧实现替换边界**；
> 3. Phase 1 结束时，**运行时不再依赖任何 SVG 渲染路径**，只保留后续 Phase 2/3/4 仍会复用的纯逻辑代码。

---

## 0. 设计依据（源码基线，不靠猜）

### 0.1 当前前端运行时事实

基于当前源码：

- `scripts/frontend_html.py:234-236`：DAG 容器仍是
  ```html
  <div class="dag-container" id="dag-container">
      <svg class="dag-svg" id="dag-svg"></svg>
  </div>
  ```
- `scripts/frontend_html.py:3064-3071`：`_generate_flowchart_html()` 当前只注入 `render_group.js`，且模板缺占位符会直接 `raise RuntimeError`。
- `scripts/render_group.js:21-172`：只负责 group 相关 SVG 绘制（折叠/展开 box、label、timing、info、children 递归），不负责 edge hover/active。
- `scripts/dag_html_adapter.py`：只产出 `groups / nodes / edges / io_groups / root_groups / meta` 等前端消费数据；从字段职责看，**Phase 1 无需改它**。

### 0.2 当前 UT 基线事实

基于当前 `develop/frontend/storage/testset/unit/test_frontend_html.py`：

- **A 类：纯数据 / 布局 / redirect 语义** 10 个：
  - `test_collect_source_files_deduplicates_src_call_def_and_class_def_files`
  - `test_container_group_has_ports_when_whole_forward_called`
  - `test_build_source_map_skips_missing_files`
  - `test_expanded_group_registers_in_port_node_id_in_nodemap`
  - `test_expanded_group_registers_out_port_node_id_in_nodemap`
  - `test_global_edge_from_port_node_id_resolves_without_missing_error`
  - `test_global_edge_to_collapsed_group_redirects_correctly`
  - `test_global_edge_endpoint_from_collapsed_group_port_node_id_resolves_without_missing_error`
  - `test_global_edge_endpoint_from_port_node_id_inside_collapsed_ancestor_resolves_without_missing_error`
  - `test_port_node_ids_indexed_in_ancestor_groups`
- **D 类：旧 edge index 全局变量探针** 1 个：
  - `test_html_emits_edge_index_globals`
- **B/C 类：SVG 强绑定 + hover/click 行为** 共 30 个：
  - B 类（SVG/CSS/DOM 强绑定）16 个
  - C 类（hover/click/active 行为）14 个
- `testset/unit/test_frontend_redirect.py` 8 个测试，都是纯 redirect 逻辑，可保留。
- `testset/unit/frontend/test_edge_routing.js` 是纯 routing 语义，不应跟 SVG DOM 一起一锅端掉。
- `testset/unit/frontend/test_render_group_helpers.js` 是 `render_group.js` 的 SVG helper 测试，语义要迁移，但断言形式必须替换。
- `testset/unit/frontend/test_hover_index.js` 是旧 hover index/active class 逻辑，语义仍有价值，但 Phase 1 不实现交互，因此应临时 skip，等 Phase 3 改写后恢复。

> 结论：之前文档里把“旧前端 Jest 全部 skip”和“B/C 27 个测试”写死，都不够准确。**本次设计以当前源码为准**：Python B/C 是 30 个；Jest 需要拆分处理，`test_edge_routing.js` 保留。

---

## 1. Phase 1 目标与边界

### 1.1 Phase 1 最终目标

Phase 1 完成后得到的是一个**可看不可动**的 Canvas 版 DAG：

- 能生成 smoke / 5698781 HTML
- 能静态显示：group、leaf node、port、edge、arrow、IO pill
- 能 pan / zoom / fit to view
- 有三段式 progress overlay
- 没有 hover / click / toggle / side-panel 触发能力
- 运行时不再走 SVG 渲染路径

### 1.2 必须保持不变的边界

| 边界 | 约束 |
| --- | --- |
| 布局算法 | `computeRanks` / `orderRanks` / `layoutGroup` / `indexGroupAncestors` 原样保留，不改一行 |
| 后端 JSON | `dag_html_adapter.py` 0 改动 |
| SOURCE_MAP / 源码采集 | `_collect_source_files` / `_build_source_map` 保留 |
| DOM 浮层 | `tooltip` / `side-panel` / `summary` / `legend` / `progress overlay` DOM 结构保留 |
| 错误处理 | 占位符缺失、容器缺失、引擎初始化失败都必须明确 `raise`，不做 fallback / no-op |

### 1.3 Phase 1 明确不做的事

| 不做项 | 推迟到 |
| --- | --- |
| Group toggle / Expand All / Collapse All | Phase 2 |
| Hover / click / active overlay / overlap wheel | Phase 3 |
| info button → source panel、edge click → evidence panel、tooltip | Phase 4 |
| 任何 trace 数据处理前移到前端 | 永不做 |

---

## 2. Phase 1 的“替换 + 清理”总原则

### 2.1 运行时路径：立即切走，不留 fallback

一旦 Stage 1.1 开始，运行时 HTML 模板就只允许两种脚本注入：

1. `__ENGINE_BUNDLE_PLACEHOLDER__`
2. `__RENDER_CANVAS_JS_PLACEHOLDER__`

旧 `__RENDER_GROUP_JS_PLACEHOLDER__` 可以在 Stage 1.1 保留模板槽位，但只能替换为空字符串；**不允许出现“优先 Canvas，失败就回退 SVG”** 这种兼容层。

### 2.2 旧实现清理分两层

| 层级 | 处理原则 |
| --- | --- |
| 运行时死路径 | 对应 Stage 完成后立即从生产路径移除 |
| 仍有迁移价值的语义 | 先迁成 Canvas 版 snapshot UT，再删旧文件 / 旧测试 |

### 2.3 旧 UT 清理分三类

| 类别 | 处理原则 |
| --- | --- |
| 纯数据/布局语义（A 类 + redirect） | 保留；探针从 SVG DOM 改成 `window.__renderSnapshot()` |
| SVG 结构强绑定（B 类 + `test_render_group_helpers.js`） | 在有 Canvas 等价 UT 后删除或改写，**不长期 skip** |
| 交互语义（C 类 + `test_hover_index.js`） | Phase 1 期间临时 skip；Phase 3 必须恢复成 Canvas 版 UT |

---

## 3. Stage 拆分与每阶段交付

## 3.1 Stage 1.1 — 模板切换 + Canvas 引擎基建

### 产出

- `<svg id="dag-svg">` → `<div id="dag-stage">`
- 注入链路从 `render_group.js` 改为 `Pixi bundle + render_canvas.js`
- `render_canvas.js` 建立 `PIXI.Application + Viewport + L0~L5`
- 暴露 `window.__phase1NoInteractionMode = true`
- 暴露空骨架版 `window.__renderSnapshot()`

### 旧功能替换

| 旧 | 新 |
| --- | --- |
| `dag-svg` 容器 | `dag-stage` |
| `render_group.js` 注入 | `render_canvas.js` 注入 |
| SVG 样式规则 | 仅保留 DOM 浮层样式；SVG 样式删掉 |

### 旧 UT / 旧实现清理

- `test_frontend_html.py` 中 B/C 30 个测试统一先加 skip，reason 分两类：
  - `canvas phase1: svg-specific assertions removed`
  - `canvas phase1: interaction deferred to phase3`
- `test_edge_routing.js` 不动
- `test_render_group_helpers.js` 暂不动，等 Stage 1.2 有等价 UT 后删除
- 运行时不再执行 `render_group.js`

### 本阶段硬门槛

- P1-1 / P1-2 / P1-3 / P1-10 通过
- 模板缺占位符时明确报错
- 任何布局函数源码 diff 必须为 0

## 3.2 Stage 1.2 — Node / Group / Port 静态绘制 + 前端 UT infra 清理

### 产出

- `NodeView`：leaf node 矩形 + 主/副 label
- `GroupView`：collapsed / expanded box、header、timing、info 圆点占位
- `PortRenderer`：端口圆点绘制
- `frontend_test_infra.py`：迁入前端渲染 probe helper
- `window.__renderSnapshot()` 开始返回 nodes / groups / ports / layers

### 旧功能替换

| 旧 | 新 |
| --- | --- |
| `renderNodeAt()` | `NodeView.draw()` |
| `renderCollapsedGroupBox()` / `renderExpandedGroupBox()` 等 | `GroupView.drawCollapsed()` / `drawExpanded()` |
| `registerCollapsedGroupPorts()` / `registerExpandedGroupPorts()` | `PortRenderer.drawPorts()` + 仍写 `nodePortMap` |
| `test_render_group_helpers.js` 对 `<rect>` / `<text>` 的断言 | P1-4 / P1-5 / P1-7 / P1-14 / P1-15 对 snapshot 断言 |

### 旧 UT / 旧实现清理

- 新建 `testset/unit/frontend_test_infra.py`
- 从 `test_frontend_html.py` 迁出并去重：
  - `_make_node`
  - `_make_group`
  - `render_minimal_flowchart_to_string`
  - `_extract_script`
  - `_patch_script_for_node`
  - `_run_render_probe`
- A 类测试里凡是依赖 SVG DOM 的探针，统一改成 `window.__renderSnapshot()`
- `test_render_group_helpers.js` 在等价 Python UT 落地后删除
- `scripts/render_group.js` 在 Stage 1.2 完成后可直接删除：它只负责 group 绘制，迁移完成后不再有存在价值

### 本阶段硬门槛

- P1-4 / P1-5 / P1-7 / P1-9 / P1-14 / P1-15 通过
- `nodePortMap` 与 snapshot 端口坐标逐值相等
- `frontend_test_infra.py` 成为唯一前端 probe helper 来源

## 3.3 Stage 1.3 — EdgeBatch / Arrow / 旧 SVG edge 绘制清理

### 产出

- `EdgeRoute.buildDirect()`：三次贝塞尔点列
- `EdgeRoute.buildIntraGroup()`：跨层折线点列
- `EdgeBatch.draw()`：按 type 分桶批绘到 L1
- `window.__renderSnapshot()` 补齐 edges

### 旧功能替换

| 旧 | 新 |
| --- | --- |
| `buildDirectEdgePath()` | `EdgeRoute.buildDirect()` |
| `buildIntraGroupEdgePath()` | `EdgeRoute.buildIntraGroup()` |
| 每条边一条 `<path class="edge-path">` | 每个 type 一个 Graphics bucket |
| `<marker>` 箭头 | 终点三角形 |
| hitbox path | Phase 1 不存在；Phase 3 再用空间索引接回命中 |

### 旧 UT / 旧实现清理

- `test_html_emits_edge_index_globals` 改写为 snapshot edge-count / edge-layer 断言
- `test_frontend_redirect.py` 保留，但若探针依赖 HTML，则改为 snapshot 语义
- 旧 SVG edge 绘制函数、hitbox path 生成代码、`edge-path` DOM 相关渲染路径在 Stage 1.3 结束后从生产代码删除
- `test_edge_routing.js` 保留，继续验证 ancestor / collapsed redirect 语义；不和 SVG 清理绑定

### 本阶段硬门槛

- P1-6 / P1-16 / P1-E1 / P1-E2 / P1-E3 通过
- redirect 边数量与 `DATA.edges` 一致
- 旧生产代码中不再出现 edge hitbox DOM 创建路径

## 3.4 Stage 1.4 — IO / Culling / Progress / Pan-Zoom-Fit + Phase 1 收尾清理

### 产出

- `IOLayer.drawPill()`
- `CullManager.update(viewportBounds)`
- Viewport pan / wheel zoom / fit-to-view
- progress 文案固定为三阶段：
  - `正在计算 DAG 布局…`
  - `正在绘制节点与容器…`
  - `正在批绘连边…`
- `window.__renderSnapshot()` 补齐 io_pills / viewport / culling 状态

### 旧功能替换

| 旧 | 新 |
| --- | --- |
| SVG IO pill 绘制 | Canvas IOLayer |
| CSS transform fit | Viewport transform |
| SVG 全量 label 常驻 | Culling 后视口外 label 不创建 / 回收 |
| 旧 render 阶段文案 | 三段式 Canvas 阶段文案 |

### 旧 UT / 旧实现清理

- 删除 `test_frontend_html.py` 中所有仅验证 SVG 结构、且已被 P1-* 覆盖的测试
- 保留 C 类交互测试，但统一 skip 到 Phase 3
- 保留 `test_hover_index.js` 文件，但整个 suite skip 到 Phase 3；不删除其语义
- Phase 1 收尾后，仓库里不应再有任何**运行时会走到的** SVG 渲染代码

### 本阶段硬门槛

- P1-8 / P1-11 / P1-12 / P1-13 / P1-17 / P1-18 通过
- 全部 P1-* + P1-E* 通过
- smoke + 5698781 静态 HTML 可人工 review

---

## 4. 功能等价性定义（Phase 1）

Phase 1 不验证交互等价，只验证以下三层：

### 4.1 结构等价

- `snapshot.nodes` == 可见叶子节点数
- `snapshot.groups` == group 数
- `snapshot.edges` == `DATA.edges` 数
- `snapshot.io_pills` == `DATA.io_groups` 数

### 4.2 坐标等价

- `groupLayout` 深度相等
- `nodePortMap` key / value bit-for-bit 相等
- 直接边 / 跨层边几何与旧 SVG 路由一致（允许 0.5px 容差）

### 4.3 视觉参数等价

- 颜色仍沿用现有 `getNodeColor` / 边 type 配色
- `nodeW=150`、`nodeH=36`、圆角、字体、label 截断规则不变
- 折叠 group 仍只显示一个 group box，不显示内部 leaf 矩形

---

## 5. UT 架构调整

## 5.1 新增文件

- `testset/unit/frontend_test_infra.py`
- `testset/unit/test_frontend_canvas_phase1.py`

## 5.2 保留文件

- `testset/unit/test_frontend_html.py`（保留，但删 helper、改 probe、清 SVG 专属测试）
- `testset/unit/test_frontend_redirect.py`
- `testset/unit/frontend/test_edge_routing.js`

## 5.3 临时 skip 文件 / 测试

- `test_frontend_html.py` 中 C 类 14 个交互测试
- `testset/unit/frontend/test_hover_index.js` 整个 suite

## 5.4 删除文件 / 测试

满足等价 P1-* UT 落地后删除：

- `testset/unit/frontend/test_render_group_helpers.js`
- `scripts/render_group.js`
- `test_frontend_html.py` 中 B 类 16 个 SVG 结构/CSS 测试

---

## 6. 覆盖率与阶段门槛

### 6.1 每个 Stage 的统一门槛

1. 该 Stage 新增/改写的 UT 全 PASS
2. 该 Stage 新增代码 line coverage ≥ 90%
3. 该 Stage 新增代码 branch coverage ≥ 50%
4. 前序 Stage 无回退

### 6.2 Phase 1 全量门槛

- 仅跑前端相关 UT，不跑全量：
  ```bash
  python3 -m pytest \
    testset/unit/test_frontend_canvas_phase1.py \
    testset/unit/test_frontend_html.py \
    testset/unit/test_frontend_redirect.py -v
  ```
- JS 若仍保留 `test_edge_routing.js`，单独跑其 suite；不要求把旧 hover Jest 一并跑通
- `dag_html_adapter.py` diff 必须为 0
- 生成 smoke / 5698781 HTML 供人工 review

---

## 7. Phase 1 完成后的代码形态

Phase 1 结束后，仓库应满足：

- `frontend_html.py`：保留模板、SOURCE_MAP、布局函数、DOM 浮层逻辑；生产渲染路径已切到 Canvas
- `render_canvas.js`：承接静态渲染全部职责
- `render_group.js`：已删除
- `dag_html_adapter.py`：不变
- `test_frontend_html.py`：只保留与 Canvas / 数据 / 布局仍相关的测试
- `test_frontend_canvas_phase1.py`：成为 Phase 1 主回归入口
- `frontend_test_infra.py`：成为前端 probe helper 唯一来源

---

## 8. 待 NaN review 的点

1. **Phase 1 内是否接受“删除 `render_group.js` 与其 SVG helper UT”**，而不是留到更后面阶段。
2. **旧交互测试的处理**：本方案主张“保留语义、统一 skip 到 Phase 3”，而不是直接删除。
3. **`test_edge_routing.js` 保留**：因为它验证的是 routing 语义，不是 SVG DOM；是否同意继续保留。
4. **以当前源码为准修正测试数量**：B/C 不是 27，而是 30 个；是否按源码清单执行。

> 如果这些点确认，我下一步只进入实现，不再重复出 Phase 1 方案。