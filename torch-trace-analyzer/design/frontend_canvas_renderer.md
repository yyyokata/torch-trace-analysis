# DAG 前端渲染引擎 Canvas/WebGL 替换设计文档（方案 D）

> 状态：设计待 review，**不含任何实现代码**。
> 目标文件：`scripts/frontend_html.py`（HTML 模板 + 内联 JS 渲染逻辑，3091 行）、`scripts/render_group.js`（被 `__RENDER_GROUP_JS_PLACEHOLDER__` 注入的 group/node 绘制逻辑）。
> 数据来源：`scripts/dag_html_adapter.py`（后端序列化 → 前端 JSON）。

## 0. 背景与问题定义

### 0.1 现状

当前 DAG HTML 用 **内联 SVG** 渲染，整张图是一棵 SVG DOM 树（`group-box` / `leaf-node` / `edge-path` / `text` / `port` / `info-hit` 等元素）。布局坐标在**前端 JS** 实时计算（Sugiyama 风格分层布局），绘制阶段把坐标翻译成 SVG 元素。

### 0.2 性能瓶颈（实测）

| 现象 | 根因 |
| --- | --- |
| 大模型（5698781，37MB HTML）滚动时 GPUTask 峰值 388ms、GPU 上传 76~120MB，轻微掉帧 | SVG DOM 节点规模过大（千级 group + 7000+ edge，每条 edge 还带一个不可见 hitbox），滚动触发浏览器对整棵 SVG 子树重新合成/上传纹理 |
| hover 高亮卡顿 | 高亮通过切换 DOM `class`（`.hover-active` 下批量 `opacity`）实现，浏览器对命中 CSS 选择器的成片 SVG 元素重新 paint；高亮边还要 `appendChild` 复制 path 到 overlay 层，产生 reflow |
| 折叠/展开慢 | `toggleGroup()` 走 `invokeRender()` → `svg.innerHTML` 清空重建 → 全量重跑 `render()`，整图 DOM 重建 |

### 0.3 结论

SVG 的「一图元一 DOM 节点」模型在 1000+ group / 7000+ edge 规模下，DOM 数量与浏览器 paint/composite 成本是硬上限，靠局部优化无法根治。采用 **Canvas/WebGL** 把「图元」从 DOM 节点降级为 GPU draw call 中的顶点/纹理数据，是终极解。

---

## 1. 渲染引擎选型

### 1.1 候选对比

| 维度 | PixiJS（WebGL-first） | Konva.js（Canvas2D） | 纯 Canvas2D（无依赖） | 手写 WebGL/GLSL |
| --- | --- | --- | --- | --- |
| 渲染后端 | WebGL / WebGPU（v8），批处理 | Canvas2D | Canvas2D | WebGL |
| 1000+ group / 7000+ edge 滚动 | ★★★★★ GPU 批渲染 + 视口裁剪，平移缩放走变换矩阵不重绘 | ★★★ Canvas2D 单线程逐图元绘制，需自己做脏矩形/分层缓存 | ★★★ 同 Konva，但全靠手写 | ★★★★★ 极限性能 |
| hover 局部重绘 | ★★★★ 分层 Container，高亮只重绘 overlay 层 | ★★★★ Layer 独立 canvas，高亮层单独重绘 | ★★★ 需手写分层 + 脏区 | ★★★★★ 但需手写全部 |
| 折叠/展开动画 | ★★★★ 对象/Sprite 复用，可补间 transform | ★★★★ Tween 内置 | ★★ 全手写 | ★★ 全手写 |
| 文字质量（CJK / 长标签） | ★★★★ `Text` 走 Canvas 纹理，CJK 友好（`BitmapText` 不适合 CJK） | ★★★★ `Konva.Text` 原生 Canvas 文本，CJK 友好 | ★★★★ Canvas `fillText` 原生 | ★★ 需自建字形图集，CJK 几乎不可行 |
| 与现有 Python JSON 接口兼容性 | ★★★★★ 纯前端绘制层替换，JSON 不变 | ★★★★★ 同左 | ★★★★★ 同左 | ★★★★★ 同左 |
| 内置命中检测（交互） | ★★★★ 事件系统 + `hitArea`，但海量边仍建议自建空间索引 | ★★★★ 内置图形命中 | ☆ 全手写 | ☆ 全手写 |
| 生态 / 维护成本 | ★★★★★ 成熟、文档全、社区大 | ★★★★ 成熟、API 简单 | ★★ 全自研 | ☆ 工程量最大 |
| 单文件内联体积 | minified ≈ 400–500KB（可接受） | minified ≈ 150KB | 0 | 0 |

### 1.2 推荐：PixiJS（v8，WebGL，Canvas2D 自动降级）作为主方案；Konva.js 作为降级备选

**推荐理由**

1. **规模决定后端**：7000+ edge + 1000+ group 已超出 Canvas2D「每帧逐图元 `stroke()`」的舒适区。PixiJS 的 WebGL 批处理把同材质图元合并为少量 draw call，平移/缩放只更新变换矩阵、**不重绘几何**，直接消灭「滚动重新上传 GPU」的瓶颈。
2. **分层模型天然契合需求**：Pixi `Container` 层级可一一映射「背景层 / 边层 / group 层 / 节点层 / overlay 层」，hover 高亮只需重绘 overlay 层，base 层保持静态——这正是 SVG 方案做不到的「局部重绘」。
3. **CJK 文字可控**：用 `PIXI.Text`（Canvas 纹理化）渲染 CJK label，质量与 SVG 文本相当；配合视口裁剪只为可见 label 建纹理，规避海量文本纹理内存问题（详见第 7 章）。
4. **数据接口零改动**：选型只替换「坐标 → 图元」的绘制层，后端 JSON（`dag_html_adapter.adapt_serialized_dag` 的输出）和 JS 布局算法（`computeRanks`/`orderRanks`/`layoutGroup`）原样复用。
5. **降级保障**：Pixi v8 在 WebGL 不可用时可切 Canvas2D；若后续发现 Pixi 体积/交互复杂度不划算，可平滑退到 **Konva.js**（Layer/Group 模型 + 内置命中，API 更简单，体积更小），二者架构（分层 + 自建空间索引）一致，迁移成本低。

**不选其余三者的原因**

- **纯 Canvas2D**：性能与 Konva 同档，却要从零手写分层、脏矩形、命中检测、文本布局，工程量大且易踩坑，性价比低。
- **手写 WebGL/GLSL**：性能天花板最高，但 CJK 文字需自建字形图集（几乎不可行）、交互/拾取全自研，工程量与维护成本远超收益，不符合「如无必要勿增实体」。

### 1.3 选型对项目的硬约束

- **必须单文件自包含**：当前产物是 17–37MB 的单 HTML，靠链接分享、可离线打开。**禁止 CDN 引入**，引擎 bundle 必须像 `render_group.js` 一样内联进 HTML（详见第 5 章）。
- **布局算法不动**：选型只换绘制后端，分层布局（JS）保持现状。

---

## 2. 数据接口设计

### 2.1 现有后端 JSON 结构（直接复用，不改）

`dag_html_adapter.adapt_serialized_dag()` 输出的 dict 经 `frontend_html._generate_flowchart_html()` 注入到 `__FLOWCHART_DATA_PLACEHOLDER__`，前端以全局 `DATA` 消费。关键字段：

| 字段 | 含义 | 前端消费点 |
| --- | --- | --- |
| `groups[]` | 所有 group/container 节点；含 `id`、`label`、`class_name`、`attr_name`、`depth`、`node_type`(`group`/`container_group`)、`is_synthetic`/`synthetic_type`、`children_nodes`(id 引用)、`children_group_ids`(id 引用)、`call_order`、`in_ports`、`out_ports`、`src_file`/`src_start_line`、timing 默认值 | 布局递归、group box / label / port / timing 渲染 |
| `nodes[]` | 叶子节点（module / functional / IO leaf）；含 `id`、`label`、`depth`、`node_type`、`attr_name`、`class_name`、`call_loc` | 节点矩形 + label 渲染 |
| `edges[]` | 全局边；含 `from`、`to`、`type`、`from_node`、`to_node`、`flows`（数据流溯源，evidence 面板用） | 边路径渲染 + evidence 面板 |
| `io_groups[]` | 顶层 Input/Param/Const/Result 折叠组；含 `id`、`label`、`io_subtype`、`collapsed`、`member_ids`、`member_count` | 顶/底部 IO pill 渲染 |
| `root_groups[]` | 顶层 group id 列表 | 布局/渲染入口 |
| `input_node_ids`/`param_node_ids`/`const_node_ids`/`output_node_ids` | 四类 IO 叶子 id | IO 节点定位 |
| `meta` / `has_timing` | 元信息 / 是否有 timing | 头部信息、timing 显隐 |

> **children 已是 id 引用**（2026-06-20 优化，HTML 37MB→17MB），Canvas 方案继续按 id 索引取节点，不回退为内联对象。

### 2.2 是否需要改 `frontend_html.py` 序列化部分

- **`dag_html_adapter.py`：不改**。它只负责「序列化 DAG → 前端 JSON」，与渲染后端无关。
- **`frontend_html.py`：仅改 HTML 模板与注入的 JS**。`_generate_flowchart_html()` 的占位符替换机制（DATA / SOURCE_MAP / JS 注入）保留，只是：
  - 模板里的 `<svg>` 容器换成 Canvas 挂载点；
  - `__RENDER_GROUP_JS_PLACEHOLDER__` 注入的 `render_group.js` 换成新的 Canvas 渲染模块；
  - 新增引擎 bundle 注入占位符（见第 5 章）。

### 2.3 坐标系统：布局保留在前端 JS

**决策：layout 计算保持在前端 JS（维持现状），不迁移到后端 Python。**

理由：
1. 布局算法（`computeRanks` 909–986、`orderRanks` 988–1049、`layoutGroup` 1051–1166）是**渲染后端无关**的纯坐标计算，产出 `groupLayout`（每个 group 的 `w/h/childPositions`）和 `nodePortMap`（端口坐标）。Canvas 方案直接复用这套坐标，**只替换「坐标 → 图元」这一步**，把迁移范围收敛到最小，降低风险。
2. 折叠/展开会改变可见拓扑 → 需要重算布局，放在前端可即时响应，无需后端往返。
3. 后端 Python 不引入任何布局职责，符合「Trace 数据处理在后端、展示在前端」的边界（布局属于展示，归前端）。

> 备选（不在本次范围）：若未来超大图布局 CPU 成为瓶颈，可将 `computeRanks/orderRanks/layoutGroup` 下沉为 WASM 或后端预计算，但当前通过 `runChunked` 异步分片已可缓解，**本次不做**。

---

## 3. 渲染架构设计

### 3.1 整体分层（Pixi Container 层次 / Konva Layer）

```text
HTML <div id="dag-stage">              ← Pixi Application 挂载点（替代 <svg id="dag-svg">）
└─ PIXI.Application (canvas)
   └─ Viewport (pan/zoom 根容器，替代外层 scroll + CSS transform)
      ├─ L0 backgroundLayer   ← 画布底色、可选网格
      ├─ L1 edgeLayer         ← 所有边的静态绘制（base，常态低对比）
      ├─ L2 groupLayer        ← group box（展开框 / 折叠框）、group header/label/timing、info button、port
      ├─ L3 nodeLayer         ← 叶子节点矩形 + node label/sublabel
      ├─ L4 overlayLayer      ← hover/click 高亮（高亮边、节点描边、active edge）——唯一需要频繁重绘的层
      └─ L5 ioLayer           ← 顶/底部 IO group pill（与主图分离的固定带状区）
```

DOM（HTML，非 Canvas）仍保留为浮层，**继续用现有实现**：
- `tooltip`、`side-panel`（源码/evidence 面板）、`render-progress-overlay`（进度条）、`controls`（按钮）、`legend`、`summary`、`metric-help-popup`。

> 关键收益：base 层（L1/L2/L3）一次绘制后静态缓存，平移/缩放只动 Viewport 的变换矩阵；hover 只重绘 L4 overlay 层——彻底解决「滚动重新上传 GPU」「hover 触发成片 repaint」。

### 3.2 节点 / group 渲染单元设计

| 渲染单元 | 现状（SVG） | Canvas 设计 |
| --- | --- | --- |
| 叶子节点 | `renderNodeAt`（1502）创建 `rect.leaf-node` + `text.node-label`/`node-sublabel`，尺寸 `nodeW=150 / nodeH=36`（894） | 一个轻量「NodeView」：Graphics 圆角矩形 + `Text` 主/副标签。颜色沿用 `getNodeColor`（863）。Text 仅在视口内创建（裁剪） |
| 折叠 group | `render_group.js: renderCollapsedGroupBox`(22) + Label + Timing | 折叠态用单个圆角矩形 + label + timing 文本，作为一个可点击图元（带 hitArea） |
| 展开 group | `renderExpandedGroupBox`(91) + Header(109) + Timing(135) + Info(1471) + Ports | 展开态：外框 Graphics + header 文本 + info 圆点 + 端口圆点；子内容由布局递归填入 |
| 端口 port | `port` 圆/点，坐标记录在 `nodePortMap`（1534/1937） | 小圆点 Graphics，坐标复用 `nodePortMap`；可合并进 group 层批渲染 |
| info button | `appendGroupInfoButton`(1471) 的 `group-info-hit` 圆 | overlay 命中区 + 图标文本，点击触发 `showSourcePanel` |

**对象复用与裁剪**：
- 维护「id → View 对象池」，折叠/展开/重布局时复用对象、只更新位置与可见性，避免反复 new。
- **视口裁剪（culling）**：每帧只为视口内（含 margin）的节点创建/显示 Text 与 Graphics，视口外的回收。这是控制千级节点文字纹理内存的核心手段。

### 3.3 边路由渲染

| 项 | 现状（SVG） | Canvas 设计 |
| --- | --- | --- |
| 直接边 | `buildDirectEdgePath`(1563) 三次贝塞尔 | Graphics `bezierCurveTo` 绘制；几何来自同一套路由计算（复用现算法） |
| 跨层长边 | `buildIntraGroupEdgePath`(1590) + `laneGutter` 避障(1620) | 复用折线/避障路由结果，Graphics 画折线 |
| 箭头 | `<defs>` marker(1438–1448) | Canvas 无 marker，箭头作为小三角 Graphics 在边终点按切线方向绘制 |
| 边集合 | 每条边一个 `path` + 一个不可见 hitbox（1640） | **base 边全部画进 L1 edgeLayer 的少量 Graphics（按颜色/类型分桶批绘）**；不再为每条边建 DOM hitbox |
| 命中检测 | SVG `pointer-events: stroke` | **自建空间索引**（网格 / quadtree），存每条边的折线段，hover 时按鼠标坐标做「点到线段距离」查询（详见 4.x） |
| 高亮边 | `syncActiveEdgeOverlay`(418) `appendChild` 复制 path 到 overlay | 命中边重绘到 L4 overlayLayer 的 Graphics（高亮色/加粗），base 层整体降透明度（dim） |

> base 边「批绘」是性能关键：把 7000+ 边合并到极少量 Graphics 一次性提交，而非 7000+ DOM。

### 3.4 折叠/展开状态管理

- 沿用全局 `collapsedState`（折叠标志）、`nodeAncestorGroups`（祖先链，1057/`indexGroupAncestors` 820）、`resolveCollapsedAncestor`（356）。
- 状态变更流程（替换现 `toggleGroup`→`invokeRender`→`svg.innerHTML` 全量重建）：
  1. 改 `collapsedState`；
  2. 重算受影响子树布局（`layoutGroup`，可局部）；
  3. 在对象池上**增量更新**节点/边的位置与可见性，必要时补间动画（Phase 2）；
  4. 重画 base 层（首版可整层重画，后续优化为脏区/局部）。

---

## 4. 交互功能迁移清单

> 原则：所有交互在 Canvas 下通过「自建命中检测 + 仅重绘 overlay 层」实现，base 层尽量不动。

| # | 现有 SVG 交互 | 现实现（函数@行号） | Canvas 实现方式 |
| --- | --- | --- | --- |
| 1 | hover 高亮 group + 关联边 | `bindGroupHover`(772) → `applyGroupFocusState`(589) → `computeActiveNodeIds`(525)，切换 `classList` 触发 CSS 批量 dim | Viewport 上监听 `pointermove`；命中 group 后用 `computeActiveNodeIds` 算激活集合；**只重绘 L4 overlay**（高亮边 + 节点描边），base 层整体 dim 用层级 alpha，不逐元素改属性 |
| 2 | hover 高亮 IO node | `bindIONodeHover`(790) | IO 命中 → 同 #1，重绘 overlay |
| 3 | edge click 高亮 | `setEdgeFocus`(765) → `applyEdgeFocusState`(700) | 空间索引拾取边 → 设 focus → overlay 重绘高亮该边；再次点击/ESC 清除（`clearEdgeFocus` 755） |
| 4 | 高亮边置顶 | `syncActiveEdgeOverlay`(418)/`clearActiveEdgeOverlay`(405) | overlay 层天然在最上，命中边重绘进 overlay Graphics，无需复制 DOM |
| 5 | 长边激活样式 | `syncLongEdgeDisplay`(477)（dasharray 切换） | overlay 重绘时对长边用高亮样式（实线/加粗），base 长边可虚线 |
| 6 | 滚轮切换重叠边 | `handleWheel`(1813)，改 `hoveredEdgeIdx` 重绘 overlay | 命中点拾取到多条重叠边 → wheel 切 `hoveredEdgeIdx` → overlay 重绘当前选中边（逻辑直接移植） |
| 7 | 折叠/展开 group | `toggleGroup`(2260) → `invokeRender`(1265) 全量 | 见 3.4：增量更新对象池 + 重画 base，不重建 DOM 树 |
| 8 | Expand All / Collapse All | btn(2528/2532) 遍历 `DATA.groups` 改 `collapsedState` 后重绘 | 批量改 `collapsedState` → 重算布局 → 重画 base（一次） |
| 9 | Fit to View | btn-fit(2536)，CSS `transform: scale()` | Viewport `fitWorld()` / 设置 scale+offset，矩阵变换，无重绘几何 |
| 10 | IO group 面板（展开/折叠 pill） | `renderIOGroupPill`(1950)、`bottomIOItems`、`computeIOGroupExpandedLayout` | L5 ioLayer 绘制 pill；点击切换展开，复用现布局算法（`io_groups` 数据不变） |
| 11 | 渲染进度条 | `render-progress-overlay` + `setRenderProgress`(1182)/`runChunked`(1225)/`nextFrame`(1167) | **DOM 浮层保留不变**；进度回调改为按「布局 / base 绘制 / 边批绘」阶段更新百分比 |
| 12 | info button | `group-info-hit`(1476 click) → `showSourcePanel`(2356) | overlay 命中圆点 → 触发现有 `showSourcePanel`（DOM 面板不变） |
| 13 | 源码侧边栏 | `showSourcePanel`(2356)，用 `SOURCE_MAP` | **完全复用**（纯 DOM + SOURCE_MAP，与渲染后端无关） |
| 14 | 边 evidence 面板 | `handleClick`(1807) → `showEdgePanel`(2435)，用 `edge.flows` | 空间索引拾取边 → 调现有 `showEdgePanel`（DOM 面板 + flows 不变） |
| 15 | tooltip | `showTooltip`(2292)/`hideTooltip`(2311) | 命中节点/边 → 复用现有 DOM tooltip，坐标用鼠标位置 |
| 16 | edge active/dim 样式 | CSS `.hover-active .edge-active/.edge-dim` | base 层 alpha 降为 dim、overlay 层画 active；色值沿用现配色（`dep`/`flow`/`internal`） |

> 命中检测统一基建：维护 **节点矩形空间索引** 与 **边线段空间索引**（网格哈希即可，按视口世界坐标分桶）。`pointermove` 时先查节点、再查边，得到 hover 目标，驱动 overlay 重绘。这是替换 SVG `pointer-events` / `elementsFromPoint` 的关键。

---

## 5. 与现有后端的接口边界

### 5.1 `frontend_html.py` 保留 / 替换清单

| 部分 | 处理 |
| --- | --- |
| `_generate_flowchart_html(data)`（3064）占位符替换框架 | **保留**：仍负责把 DATA / SOURCE_MAP / 渲染 JS / 引擎 bundle 注入模板 |
| `_collect_source_files`(2591) / `_build_source_map`(2614) / `SOURCE_MAP` | **保留**：源码面板数据，与渲染后端无关 |
| `FLOWCHART_HTML_TEMPLATE` 的 `<style>` 中 **SVG 相关样式**（`.dag-svg .group-box`/`.leaf-node`/`.edge-path`/`.port` 等） | **删除**：改为 Canvas，无 SVG 元素样式（DOM 浮层样式如 `.tooltip`/`.side-panel`/进度条保留） |
| `<div class="dag-container"><svg id="dag-svg">`（234–236） | **替换**：改为 `<div id="dag-stage">` 作为 Pixi 挂载点 |
| 内联 JS 的布局函数（`computeRanks`/`orderRanks`/`layoutGroup`/`indexGroupAncestors`/`collapsedState`/`resolveCollapsedAncestor`/IO 布局） | **保留**（迁入新 JS 模块）：渲染后端无关 |
| 内联 JS 的 SVG 绘制函数（`renderNodeAt`/`createEdgePathElement`/`buildDirectEdgePath`/`buildIntraGroupEdgePath`/`renderGroupAt` 及 `render_group.js`） | **替换**：改为 Canvas 渲染模块（NodeView/EdgeView/GroupView + 空间索引 + overlay 重绘） |
| 交互中纯 DOM 部分（`showSourcePanel`/`showEdgePanel`/`showTooltip`/进度条/按钮事件） | **保留**：仅把「命中来源」从 SVG 事件换成 Canvas 命中检测回调 |
| `render_group.js` | **替换**：新增 `render_canvas.js`（或拆模块）承载 Canvas 绘制；旧文件下线 |

### 5.2 生成的 HTML 结构变化

- `<svg id="dag-svg">` → `<div id="dag-stage">`（Pixi 在此 append `<canvas>`）。
- 删除所有 `.dag-svg ...` SVG 样式；保留 DOM 浮层样式。
- 新增引擎 bundle 注入占位符（见 5.3）。
- DOM 浮层（tooltip / side-panel / progress overlay / controls / legend / summary）**结构不变**。

### 5.3 JS bundle 引入方式：**inline（强制），禁止 CDN**

- 约束：产物是单 HTML、靠链接分享、需离线打开（17–37MB），**不能依赖外网 CDN**。
- 方案：在 `_generate_flowchart_html()` 中，像现在 `render_group.js` 一样，把 **PixiJS minified bundle** 与 **新渲染模块 JS** 读入并替换到新占位符（如 `__ENGINE_BUNDLE_PLACEHOLDER__` / `__RENDER_CANVAS_JS_PLACEHOLDER__`），整体内联进 HTML。
- 体积评估：Pixi minified ≈ 400–500KB，相对 17–37MB 产物可忽略；若选 Konva 则 ≈ 150KB。
- 占位符缺失时必须 `raise`（沿用现有 `render_group.js` 占位符缺失即报错的写法，3069–3070），不做静默 fallback。

---

## 6. 分阶段实施路径

> 每个 Phase 独立可验收，验收基线统一为「与现 SVG 版在相同 DATA 下视觉/拓扑一致 + 性能不劣化」。验证模型：smoke（小图）+ 5698781（大图，17MB 级）。**本设计文档阶段不写任何实现代码**，以下为 review 通过后的实施顺序。

### Phase 1：基础静态渲染（无交互）
- **范围**：Pixi Application + Viewport 搭建；复用现有布局产出（`groupLayout`/`nodePortMap`）；绘制 group box（折叠/展开）、叶子节点 + label、端口、base 边（批绘）+ 箭头；顶/底 IO pill 静态展示。
- **Milestone**：5698781 能完整静态渲染出与 SVG 版一致的图形。
- **验收**：
  1. 节点/边总数与 SVG 版一致（**节点数只能增不能减**，对齐已验证基线）；
  2. 平移/缩放（Viewport）流畅，滚动无明显 GPU 上传峰值（对比当前 388ms 应显著下降）；
  3. 文字（含 CJK label）清晰无糊。

### Phase 2：折叠 / 展开
- **范围**：`toggleGroup` / Expand All / Collapse All 改为「改 `collapsedState` → 局部重算布局 → 对象池增量更新 → base 重画」；可选补间动画。
- **Milestone**：任意 group 折叠/展开、全展/全折，拓扑与 SVG 版一致。
- **验收**：
  1. 折叠/展开后节点/边数与 SVG 版一致；
  2. 大图折叠/展开响应时间显著优于现 `svg.innerHTML` 全量重建；
  3. 折叠祖先的边重路由（`resolveCollapsedAncestor`）正确。

### Phase 3：hover 高亮 + edge 高亮
- **范围**：节点矩形/边线段空间索引；`pointermove` 命中；overlay 层重绘高亮（group hover、IO hover、edge hover/click、滚轮切重叠边、长边样式、active/dim）。
- **Milestone**：交互 #1–#6、#16 全部可用。
- **验收**：
  1. hover 高亮集合与 SVG 版 `computeActiveNodeIds` 结果一致；
  2. hover 时仅 overlay 层重绘，base 层不重绘（用帧分析确认无成片 repaint）；
  3. 大图 hover 跟手、无掉帧。

### Phase 4：其余交互与浮层
- **范围**：info button → `showSourcePanel`、edge click → `showEdgePanel`（flows）、tooltip、IO group 展开面板、渲染进度条阶段化、Fit to View、legend/summary。
- **Milestone**：交互 #7–#15 全部可用，DOM 浮层全部接通 Canvas 命中回调。
- **验收**：
  1. 源码面板/evidence 面板内容与 SVG 版一致；
  2. 进度条覆盖布局/绘制/边批绘各阶段，无长时间假死；
  3. 全量交互回归通过，产物可离线打开。

> 各 Phase 完成后按既有规范跑相关 UT（前端产物结构校验），e2e 按当前「重构期暂停、整体收尾统一跑」的约定处理。

---

## 7. 风险与降级策略

### 7.1 文字渲染（CJK / 长标签）
- **风险**：`PIXI.Text` 为每个文本生成 Canvas 纹理；千级节点全量建纹理会吃显存。`BitmapText` 需预生成字形图集，CJK 字符集巨大、不可行。
- **对策**：
  1. **视口裁剪**：只为视口内（含 margin）节点创建/保留 `Text`，移出视口即回收纹理（对象池）。
  2. **缩放阈值**：缩放比过小（看不清）时隐藏 label，只画图元，避免无意义纹理。
  3. **长标签截断**：沿用现有截断逻辑（label 超宽加省略号），完整文本进 tooltip。
  4. **分辨率**：`Text` 按 `devicePixelRatio` 设置 resolution，保证高分屏清晰。

### 7.2 打印 / 截图（Canvas 不如 SVG 矢量）
- **风险**：Canvas 是位图，缩放/打印不如 SVG 矢量清晰；浏览器「打印整页」只会截到当前 canvas 像素。
- **对策**：
  1. **高倍导出 PNG**：提供「导出当前视图」按钮，用 `app.renderer.extract` / `canvas.toDataURL` 以 2–4× devicePixelRatio 渲染导出，满足截图清晰度。
  2.（可选，后续）**离屏 SVG 导出**：保留一份「坐标 → SVG」的轻量导出函数（复用布局结果），仅用于「导出矢量图」场景，不参与日常渲染。**本次不实现，列为后续可选项**。

### 7.3 浏览器兼容性
- **风险**：WebGL 上下文丢失（context lost）、老环境无 WebGL。
- **对策**：
  1. Pixi v8 在无 WebGL 时自动降级 Canvas2D；
  2. 监听 `webglcontextlost` / `webglcontextrestored`，丢失时重建场景（布局结果可复用，重画即可）；
  3. 目标浏览器为现代 Chromium（团队内部使用场景），不为 IE 等老浏览器兜底。

### 7.4 工程风险
- **对象/纹理泄漏**：折叠/展开、裁剪回收时必须 `destroy` 不再用的 Graphics/Text/纹理，避免显存累积——以对象池统一管理生命周期。
- **命中检测精度**：边线段空间索引的网格粒度需调参，过粗命中慢、过细内存大；以 5698781 实测调优。
- **范围蔓延**：严格遵守「只换绘制后端，不动布局算法、不动后端 JSON、不动 DOM 浮层逻辑」，把改动面收敛到 `frontend_html.py` 模板 + 新 Canvas JS 模块。

---

## 8. 接口边界一句话总结

- **后端 `dag_html_adapter.py`**：不动。
- **`frontend_html.py`**：保留占位符注入框架 + SOURCE_MAP + DOM 浮层逻辑 + 布局算法；替换 SVG 模板与 SVG 绘制 JS（含 `render_group.js`）为 Canvas 引擎 + Canvas 渲染模块（inline 引入）。
- **数据流**：DAG → `adapt_serialized_dag`（JSON 不变）→ JS 布局（不变）→ **Canvas 绘制（新）** → DOM 浮层（基本不变）。
