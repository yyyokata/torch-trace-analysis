# DAG 前端渲染进度条设计文档

## 1. 当前渲染流程梳理

### 1.1 代码入口与调用链

目标文件：`scripts/frontend_html.py`。

当前 HTML 模板中没有独立的 `initVisualization()` / `renderDAG()` 命名入口，主渲染入口是内嵌 JS 的 `render()`，在脚本末尾直接执行：

```text
页面脚本加载
  ├─ 构建全局索引：groupMap / nodeMap / collapsedState / nodeAncestorGroups / EDGE_BUNDLE_META
  ├─ 注册交互事件：side panel、Expand All、Collapse All、Fit to View
  └─ render()
```

`render()` 的主流程如下：

```text
render()
  ├─ 清空上一轮渲染状态
  │   ├─ groupLayout = {}
  │   ├─ nodePortMap = {}
  │   ├─ edgeDomRegistry = []
  │   └─ nodeDomRegistry = new Map()
  │
  ├─ layout root groups
  │   └─ DATA.root_groups.map(layoutGroup)
  │       └─ layoutGroup(gid)
  │           ├─ collapsed group: 直接记录单节点尺寸
  │           └─ expanded group:
  │               ├─ 递归 layout 子 group
  │               ├─ computeRanks(group)
  │               ├─ orderRanks(rankInfo, childSizes)
  │               ├─ rank 内 wrap 成多行
  │               ├─ skip-rank edge gutter 检测
  │               └─ 写入 groupLayout[gid]
  │
  ├─ 计算顶层 IO / root / svg 尺寸
  │   ├─ topIOItems / bottomIOItems
  │   ├─ computeIOGroupExpandedLayout
  │   ├─ calcIORowWidth / calcIORowHeight
  │   └─ svgW / svgH
  │
  ├─ 清空并初始化 SVG defs
  │   └─ svg.innerHTML = `<defs>...</defs>`
  │
  ├─ 渲染 root group
  │   └─ renderGroupAt(rp.id, rp.x, rp.y)
  │       ├─ renderCollapsedGroupBox / Label / Timing / Info / Ports
  │       └─ expanded group:
  │           ├─ renderExpandedGroupBox / Header / Info / Timing
  │           ├─ renderGroupChildren(ctx)
  │           │   ├─ renderNodeAt(...)
  │           │   └─ renderGroupAt(child group, ...)
  │           ├─ buildInternalEdgeRoutingContext(ctx)
  │           ├─ registerExpandedGroupPorts(ctx)
  │           └─ renderGroupInternalEdges(ctx, routeCtx)
  │               └─ renderEdge(...)
  │                   ├─ buildIntraGroupEdgePath / buildDirectEdgePath
  │                   ├─ applyEdgePresentation
  │                   ├─ bindEdgeInteractions
  │                   └─ finalizeEdgeRendering
  │
  ├─ 渲染顶层 IO pill
  │   ├─ renderIOPillRow
  │   ├─ renderIOGroupPill
  │   └─ renderIOPill
  │
  ├─ 渲染全局边 DATA.edges
  │   └─ for edge of DATA.edges:
  │       ├─ isEdgeVisible
  │       ├─ resolveCollapsedAncestor
  │       ├─ 查 nodePortMap
  │       └─ renderEdge(...)
  │
  ├─ 更新 Header / Legend
  └─ 更新 Summary
```

另有交互入口会重复触发完整 `render()`：

- `toggleGroup(gid)`：双击 group 展开/折叠后调用 `render()`。
- `btn-expand-all`：将所有 group 设为展开后调用 `render()`。
- `btn-collapse-all`：将深层 group 设为折叠后调用 `render()`。
- IO group 双击展开/收起后调用 `render()`。
- 多 tab HTML 的 tab 切换注入逻辑中也会重置 `DATA` 后调用 `render()`。

### 1.2 主要阶段与计算量判断

以下判断基于当前源码结构和大型模型特征（groups 数百、nodes 5000+）：

| 阶段 | 主要函数 | 计算量来源 | 预期耗时等级 |
|---|---|---|---|
| 全局索引构建 | 初始化脚本中的 `groupMap` / `nodeMap` / `nodeAncestorGroups` / `buildEdgeBundleState()` | 遍历 `DATA.groups`、`DATA.nodes`、`DATA.edges`、每个 group 的 `internal_edges` | 中等。一次性 O(groups + nodes + edges + internal_edges) |
| group layout | `render()` → `layoutGroup()` → `computeRanks()` / `orderRanks()` | 递归遍历 group 树；每个展开 group 遍历 `call_order` 和 `internal_edges`；rank 排序会对同层节点做多轮 sort | 高。大型模型中是渲染前最重的同步计算之一 |
| SVG 尺寸与 IO layout | `computeIOGroupExpandedLayout()` / `calcIORowWidth()` / `calcIORowHeight()` | 遍历 IO group 成员、计算宽高 | 中等。若 IO 展开成员很多会增加耗时 |
| group / node DOM 渲染 | `renderGroupAt()` / `renderGroupChildren()` / `renderNodeAt()` | 为每个 group/node 创建多个 SVG DOM 节点和事件监听；递归渲染所有展开节点 | 很高。nodes 5000+ 时 DOM 创建和 append 是明显瓶颈 |
| internal edge 渲染 | `renderGroupInternalEdges()` → `renderEdge()` | 每条 internal edge 做端点解析、路径计算、DOM 创建、事件绑定、`getTotalLength()` | 最高风险。边数量大且 `configureLongEdgeDisplay()` 调用 `path.getTotalLength()` 可能触发布局/几何计算 |
| global edge 渲染 | `for (const edge of DATA.edges)` → `renderEdge()` | 遍历全局边，执行 collapsed ancestor redirect、路径计算、DOM 创建和事件绑定 | 高。与边数量线性相关 |
| Header/Legend/Summary | `innerHTML` 更新、Top-N sort | Top-N 对 nodes/groups 排序 | 低到中。排序 O((nodes+groups) log n)，但数量相对 DOM/edge 开销通常较小 |

综合来看，卡顿主要不是“数据加载”本身，而是 `render()` 内部的大量同步 JS 计算和 SVG DOM 构建。最慢阶段预计集中在：

1. `layoutGroup()` 递归布局，尤其展开 group 很多时；
2. `renderGroupAt()` / `renderNodeAt()` 的大规模 SVG DOM 创建；
3. `renderGroupInternalEdges()` 和全局 `DATA.edges` 渲染，尤其 `configureLongEdgeDisplay(path)` 的 `getTotalLength()`。

### 1.3 现有 loading / spinner UI

当前模板中未发现已有的 loading、spinner 或进度条 UI：

- CSS 中已有 tooltip、legend、summary、side panel、overlap badge 等样式，但没有 loading/progress/spinner 相关类。
- DOM 结构中没有加载遮罩元素。
- `render()` 是同步函数，末尾直接调用 `render()`，没有异步阶段，也没有进度状态更新。

## 2. 进度条方案选型

### 2.1 方案对比

| 方案 | 做法 | 优点 | 缺点 | 对当前代码的适配性 |
|---|---|---|---|---|
| 方案 X：同步阶段式进度条 | 渲染前定义固定阶段，例如初始化 10%、构建 group 树 30%、渲染节点 60%、渲染边 80%、layout 100%；每个阶段开始/结束更新进度条 | 实现简单；改动小；不需要重构递归渲染 | 大阶段内部仍然阻塞主线程；用户看到的可能是进度突然跳变；如果最慢的是 edge DOM 构建，阶段内依旧长时间不动 | 只能解决“完全无反馈”的问题，不能解决 UI 卡死 |
| 方案 Y：chunked async 渲染 + 进度条 | 将最慢阶段拆成批次，每批执行一部分 DOM/edge 渲染，然后通过 `requestAnimationFrame` 让出主线程并更新进度 | 用户能持续看到进度；浏览器有机会绘制遮罩、响应事件；能真正缓解卡死感 | 改动更大；需要把部分 `render()` 流程改为 async；递归 group 渲染和 edge 渲染需要拆任务队列 | 更适合 5698781 这类大型模型，是推荐方案 |

### 2.2 推荐方案

推荐采用 **方案 Y：chunked async 渲染 + 进度条**。

理由：

1. 当前问题的核心是大型模型渲染时主线程长时间被同步 JS 和 SVG DOM 操作占满；只做阶段式同步进度条，进度条本身也可能无法及时 repaint。
2. `render()` 已经有明确阶段边界，适合先做阶段拆分，再逐步把最重的 DOM 阶段切成批次。
3. 大型模型的重负载主要来自批量 `renderGroupAt()` / `renderNodeAt()` / `renderEdge()`，这些都可以改造成任务队列按 chunk 执行。
4. 进度条的目标是“用户知道当前渲染到哪一步”，但更重要的是“页面不显得完全卡死”。方案 Y 能同时满足两点。

### 2.3 推荐落地策略

设计上分两层推进，但最终文档建议按方案 Y 实现：

#### 第一层：阶段化 async render

将 `render()` 改为 `async render()`，用显式阶段更新进度：

```text
0%   准备渲染状态
10%  构建布局
25%  计算画布与 IO 布局
35%  初始化 SVG
45%  渲染模块和节点
75%  渲染依赖边
90%  更新说明信息
100% 完成
```

每个阶段之间调用一次 `await nextFrame()`，确保浏览器能绘制进度条。

#### 第二层：对最慢阶段 chunk 化

将以下阶段切为批次：

1. **模块/节点渲染阶段**
   - 当前 `renderGroupAt()` 是递归同步渲染。
   - 设计上先新增“渲染任务收集”阶段，把 group/node 渲染拆成任务列表，再按批执行。
   - 每批例如处理 50～150 个 node/group DOM 操作，批次结束后更新进度并 `await requestAnimationFrame`。

2. **internal edge / global edge 渲染阶段**
   - edge 数量大，且每条边会创建 path、绑定交互、调用 `getTotalLength()`。
   - 将可见 edge 收集成任务列表，按批执行。
   - 每批例如处理 100～300 条边，批次结束后更新进度并让出主线程。

3. **layout 阶段**
   - `layoutGroup()` 递归计算也可能较慢，但其内部依赖父子尺寸，拆分成本高于 DOM 渲染。
   - 第一版可保留为单阶段 async 前后更新；若仍卡顿，再考虑把 layout 递归改为显式 post-order 任务栈。

### 2.4 进度计算方式

建议使用“阶段权重 + 阶段内比例”的方式，避免精确耗时预测：

| 阶段 | 建议进度范围 | 说明 |
|---|---:|---|
| 准备状态 | 0% → 5% | 清空 registry、重置 hover/focus 状态 |
| layout | 5% → 25% | `layoutGroup()`、IO 尺寸计算、SVG 尺寸计算 |
| SVG 初始化 | 25% → 30% | 设置 `width/height`、写入 `<defs>` |
| group/node 渲染 | 30% → 60% | chunked 渲染 group/node/IO pill |
| internal/global edge 渲染 | 60% → 90% | chunked 渲染所有可见边 |
| Header/Legend/Summary | 90% → 98% | 更新文字说明和 summary |
| 完成淡出 | 98% → 100% | 进度条完成后淡出 |

阶段内进度：

```text
progress = phaseStart + (done / total) * (phaseEnd - phaseStart)
```

要求：如果某阶段 total 为 0，直接推进到阶段结束；不能静默跳过错误。若任务构造出现未知 routing mode、缺失必要 DOM 容器等，应沿用当前 fail-fast 风格抛错或显示错误状态，不用 fallback 掩盖。

## 3. 进度条 UI 设计

### 3.1 DOM 位置

在现有 body 结构中增加一个渲染遮罩，建议放在 `dag-container` 前后均可，但视觉上覆盖整个 viewport：

```text
<body>
  <div class="header">...</div>
  <div class="controls">...</div>
  <div class="legend" id="legend"></div>
  <div class="dag-container" id="dag-container">...</div>
  <div class="tooltip" id="tooltip"></div>
  <div class="side-panel" id="side-panel">...</div>
  <div class="summary" id="summary"></div>

  <div id="render-progress-overlay" class="iter14-render-progress-overlay">
    <div class="iter14-render-progress-card">
      <div class="iter14-render-progress-title">正在渲染 DAG…</div>
      <div class="iter14-render-progress-stage" id="render-progress-stage">准备中…</div>
      <div class="iter14-render-progress-track">
        <div class="iter14-render-progress-bar" id="render-progress-bar"></div>
      </div>
      <div class="iter14-render-progress-percent" id="render-progress-percent">0%</div>
    </div>
  </div>
</body>
```

说明：

- 使用 `.iter14-*` 前缀，避免与现有 `.side-panel`、`.tooltip` 等通用类冲突。
- overlay 固定覆盖整个 viewport，不只覆盖 SVG 区域；渲染期间用户不能误触按钮。
- 如果后续只希望覆盖画布，也可以将 overlay 放入 `dag-container` 内，但当前需求是“遮罩层覆盖整个画布，中央显示进度条”，viewport 覆盖更稳妥。

### 3.2 样式

风格与当前深色主题一致：

- 页面背景主色：`#1a1a2e`。
- 卡片背景：`rgba(15, 22, 38, 0.94)` 或 `#0f1626`。
- 主色：`#7c83fd`。
- 辅助文字：`#8892b0`。
- 高亮文字：`#ffffff` / `#e0e0e0`。

建议样式语义：

```text
.iter14-render-progress-overlay
  ├─ fixed inset: 0
  ├─ z-index 高于 tooltip，但低于或高于 side-panel 需要明确：初始渲染建议高于 side-panel
  ├─ background: rgba(26, 26, 46, 0.86)
  ├─ display flex center
  └─ transition opacity 0.25s ease

.iter14-render-progress-card
  ├─ width: min(420px, calc(100vw - 48px))
  ├─ border: 1px solid rgba(124, 131, 253, 0.35)
  ├─ border-radius: 14px
  ├─ box-shadow: 0 18px 60px rgba(0,0,0,0.45)
  └─ padding: 22px 24px

.iter14-render-progress-track
  ├─ height: 8px
  ├─ background: rgba(255,255,255,0.08)
  └─ border-radius: 999px

.iter14-render-progress-bar
  ├─ width: 0%
  ├─ height: 100%
  ├─ background: linear-gradient(90deg, #7c83fd, #64b5f6)
  └─ transition: width 0.12s ease-out
```

### 3.3 文案

阶段文案建议：

| 阶段 | 文案 |
|---|---|
| 准备 | `正在准备渲染状态…` |
| 布局 | `正在计算 DAG 布局…` |
| SVG 初始化 | `正在初始化画布…` |
| 节点渲染 | `正在渲染模块节点… 45%` |
| 边渲染 | `正在渲染依赖边… 78%` |
| 信息面板 | `正在更新图例和摘要…` |
| 完成 | `渲染完成` |

### 3.4 完成与错误状态

完成后自动淡出：

```text
setProgress(100, '渲染完成')
等待一个 frame 或 150ms
给 overlay 加 closing class
transitionend 后 display none / hidden
```

错误状态：

- 如果 async render 过程中抛错，overlay 不应静默消失。
- 建议显示 `渲染失败，请查看 Console 错误`，并 `throw err` 继续暴露真实异常。
- 这符合当前研发约束：不引入 fallback、不静默跳过、不用默认值掩盖错误。

## 4. 改动范围

### 4.1 文件范围

只修改：

```text
scripts/frontend_html.py
```

具体只涉及其中 HTML 模板内嵌的 JS / CSS / DOM 片段。

不修改：

- 后端 DAG 数据结构；
- Python trace / DAG 构建逻辑；
- `analyze_trace.py`；
- 测试仓库；
- 真实模型数据；
- `render_group.js` 文件本身。

注意：虽然当前 `renderGroupAt()` 的实现来自 `render_group.js` 占位符注入，但本次设计要求“只改 `frontend_html.py`”。因此第一版 chunk 化应优先在 `frontend_html.py` 的主 `render()` 编排层完成，不直接改 `render_group.js`。若后续确认必须细粒度拆 `renderGroupAt()` 内部递归，再另起设计讨论是否允许改 `render_group.js`。

### 4.2 需要改的函数 / 区域

| 区域 / 函数 | 当前行为 | 设计改动说明 |
|---|---|---|
| `<style>` | 无进度条样式 | 新增 `.iter14-render-progress-*` CSS，保持深色主题 |
| `<body>` DOM | 无 loading/progress DOM | 新增 `#render-progress-overlay`、stage、bar、percent 等元素 |
| 全局状态区 | 仅有 `groupLayout`、`nodePortMap`、hover/focus 状态 | 新增渲染状态，例如 `let renderGeneration = 0`，用于避免重复 render 的异步竞态 |
| 新增 `nextFrame()` | 当前无 | 封装 `requestAnimationFrame`，用于 chunk 间让出主线程 |
| 新增 `setRenderProgress(percent, stageText)` | 当前无 | 统一更新 bar width、percent text、stage text；缺少 DOM 时应抛错，不静默 return |
| 新增 `showRenderProgress()` / `hideRenderProgress()` | 当前无 | 控制 overlay 显示、完成淡出 |
| `render()` | 同步函数，直接完成全量布局和 DOM 渲染 | 改为 async 编排：准备 → layout → SVG 初始化 → group/node 渲染 → edge 渲染 → header/summary → 完成 |
| `toggleGroup(gid)` | 直接调用 `render()` | 改为触发 async render；如果允许并发点击，需要用 generation 取消旧 render 或在渲染中禁用交互 |
| Expand All / Collapse All 事件 | 直接调用 `render()` | 同上，调用新的 async render 入口 |
| IO group 展开/收起事件 | 直接调用 `render()` | 同上，调用新的 async render 入口 |
| 脚本末尾 `render();` | 页面加载后同步渲染 | 改为启动 async render，并显示 overlay |
| 多 tab 注入的 `render()` 调用 | tab 切换后同步 `render()` | 需要同步改成调用 async render 入口，否则 tab 切换仍无进度反馈 |

### 4.3 建议新增函数设计

不写具体代码，仅定义职责：

1. `nextFrame()`
   - 返回 `Promise`。
   - 内部使用 `requestAnimationFrame`。
   - 用于让浏览器 repaint 进度条。

2. `setRenderProgress(percent, stageText)`
   - 更新进度条宽度、百分比和阶段文案。
   - `percent` clamp 到 0～100。
   - 若必需 DOM 不存在，直接 throw，避免静默失效。

3. `showRenderProgress(stageText)`
   - 显示遮罩。
   - 重置进度到 0。

4. `hideRenderProgress()`
   - 将进度推进到 100。
   - 加淡出 class。
   - transition 结束后隐藏。

5. `runChunked(items, handler, options)`
   - 按批处理任务数组。
   - `options` 包含 `batchSize`、`phaseStart`、`phaseEnd`、`stageText`、`generation`。
   - 每批后更新进度并 `await nextFrame()`。
   - 如果 generation 已过期，抛出明确错误或终止当前 render；不静默产生半成品 DOM。

6. `renderAsync()` 或直接将 `render()` 改为 `async function render()`
   - 建议保留外部入口名 `render`，减少 tab 注入逻辑和现有调用点的迁移成本。
   - 内部统一处理 generation、try/catch、progress show/hide。

### 4.4 静默路径检查

本设计主动检查以下可能的静默路径，并要求实现时避免：

- `setRenderProgress()` 找不到 overlay/bar/stage/percent DOM：不得 `return`，应 `throw new Error(...)`。
- `runChunked()` 遇到未知 task type：不得跳过，应 `throw new Error(...)`。
- edge routing mode 未知：保持当前 `renderEdge()` 的 `throw new Error` 行为。
- async render 失败：不得隐藏进度条装作完成，应显示失败状态并重新抛出错误。
- 旧 render 与新 render 竞态：不得让旧任务继续 append 到新 SVG；必须用 `renderGeneration` 检测并明确中止旧任务。

## 5. UT/验证说明

### 5.1 UT 范围

进度条属于纯前端 UI 展示，不改变后端 DAG 数据结构和 Python 分析逻辑，因此：

- 不需要新增 Python UT。
- 不需要修改 `testset/unit/`。
- 不需要跑 Python 覆盖率。
- 不需要跑 e2e。

### 5.2 手动验证方式

推荐在 smoke HTML 中目测验证：

1. 重新生成 smoke HTML。
2. 打开 HTML。
3. 观察页面初始渲染时是否出现遮罩。
4. 确认阶段文案按顺序变化。
5. 确认进度条从 0% 推进到 100%。
6. 确认完成后遮罩自动淡出。
7. 双击展开/折叠 group、点击 Expand All / Collapse All，确认重新渲染时也有进度反馈。
8. 若是多 tab HTML，切换 tab 后确认新 tab 渲染也有进度反馈。

### 5.3 可选自动化验证

当前不做自动化验证；如后续需要，可用 Puppeteer / Playwright 截帧：

- 打开 smoke HTML。
- 在首屏渲染开始后截取遮罩可见状态。
- 等待 `#render-progress-overlay` 进入 hidden/closing 状态。
- 截取完成后 DAG 可见状态。

由于这类验证依赖浏览器环境和渲染时序，当前阶段不建议作为必须 UT。