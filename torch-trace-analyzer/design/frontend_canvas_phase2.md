# Canvas/WebGL 渲染 — Phase 2 详细设计

> 状态：设计待 NaN review，**不含任何实现代码**。
>
> 分支：`feat/canvas-renderer`
>
> 上位规划：`design/frontend_canvas_renderer.md`
>
> 目标文件：`scripts/render_canvas.js`、`scripts/frontend_html.py`

---

## 1. 概述

### 1.1 Phase 2 目标

Phase 1 已完成 PixiJS v8 bundle 接入、基础图元绘制、`fitToView` 的 width-only 适配，以及一条可工作的静态渲染管线。当前 Canvas 版已经能“画出来”，但还不具备后续可用性所需的核心交互能力，且仍保留两类结构性问题：

1. 折叠/展开仍走全量 `resetScene` 重建路径，无法复用图元。
2. Expand All / Collapse All 仍复用 `invokeRender({ autoFit: true })`，会重新触发 auto-fit / resize 链路。
3. progress overlay 的 generation 竞态仍会以 throw 方式中断，导致 overlay 残留。
4. Canvas 图元尚未接回 group 级交互协议：单击 show panel、双击 toggle、右键双击进入 Semantic Zoom。

因此，Phase 2 的目标不是“再加一点交互”，而是把 Canvas 渲染从**静态首屏渲染器**升级为**可增量更新、可交互 drill-down 的前端运行时**。其核心交付包括：

- 基于对象池的增量 patch 渲染路径。
- progress overlay 竞态修复。
- group 单击 / 双击交互接入。
- Expand All / Collapse All 切换到增量路径。
- Semantic Zoom（drill-down focus）与 breadcrumb。

### 1.2 与 Phase 1 的关系

Phase 2 明确建立在 Phase 1 已有实现之上，不重写以下能力：

- **Pixi 舞台与分层结构**：继续使用当前 `L0..L5` scene graph。
- **布局算法**：继续复用前端 JS 侧 `computeFlowchartLayout()` 与递归布局逻辑，不迁到后端。
- **width-only fit 语义**：`ViewportController.prototype.fitToView()` 的“只按宽度 fit”语义保持不变。
- **后端 JSON 接口**：`dag_html_adapter.py` 零改动。
- **panel DOM 壳子**：继续复用 `frontend_html.py` 中现有 side-panel 容器和内容渲染逻辑，只增加 group panel 所需字段映射与调用入口。

Phase 2 的边界很清楚：**只改前端运行时的状态管理、对象生命周期、交互事件和局部 UI 流程；不改后端 trace 处理，不改布局语义，不改数据协议。**

### 1.3 本文依据的源码事实

以下结论均来自当前分支源码，Phase 2 设计必须以这些事实为基线：

1. `render_canvas.js` 现有全量重绘入口是 `canvasRenderPhase1()`，其中在渲染开始阶段直接执行 `resetScene()`，之后重新 `computeLayout`、`layoutAndDrawRoots`、`drawGlobalEdges`（`scripts/render_canvas.js:1409-1471`）。
2. `resetScene()` 会清空所有 layer children，并重置 `engine.nodes / groups / edges / io_pills`（`scripts/render_canvas.js:1210-1225`）。
3. 当前代码中**不存在** `diffAndPatch()`，说明对象池增量路径尚未落地。
4. `ViewportController.prototype.enablePan()` 当前为空实现，只 `return this`（`scripts/render_canvas.js:514-516`）。
5. `ViewportController.prototype.enableZoom()` 目前只设置 min/max scale，并未接 wheel 事件（`scripts/render_canvas.js:517-534`）。
6. `performAutoFit()` 会在 fit 后再次 `renderer.resize(cw, canvasHeight)`，其中高度可能被放大到内容全高（`scripts/render_canvas.js:1259-1347`）。
7. `canvasRenderPhase1()` 在“完成阶段”先调用 `assertActiveRenderGeneration(generation, '完成阶段')`，再 `await hideRenderProgress()`；一旦 generation 过期会直接 throw，overlay 无法收尾（`scripts/render_canvas.js:1469-1470`）。
8. `frontend_html.py` 中 `hideRenderProgress()` 与 `assertActiveRenderGeneration()` 当前都把 generation 过期视为异常并 throw（`scripts/frontend_html.py:1450-1471`）。
9. `toggleGroup(gid)` 目前仍是 `collapsedState[gid] = !collapsedState[gid]; invokeRender();`（`scripts/frontend_html.py:1528-1531`）。
10. Expand All / Collapse All 仍是批量改 `collapsedState` 后调用 `invokeRender({ autoFit: true })`（`scripts/frontend_html.py:1796-1802`）。

上述现状直接定义了 Phase 2 的设计目标：**不是优化旧路径，而是引入一条新的增量路径，并把 toggle / expand-all / collapse-all / semantic zoom 全部切过去。**

---

## 2. 对象池设计

### 2.1 设计目标

Phase 2 的对象池（Object Pool）负责解决两个问题：

1. **图元反复销毁 / 新建**：当前每次 toggle 都会 `resetScene()`，所有 Graphics / Text 全量重建。
2. **交互入口无法复用已有图元**：Semantic Zoom、Expand/Collapse All、本地 focus 切换都需要在“保留当前 scene”的前提下只改局部可见性与属性。

因此，Phase 2 采用“**稳定对象池 + 可见集合 diff**”的模型：

- **对象池是长期持有者**：View 对象一旦创建，不因一次 toggle / focus 退出而销毁。
- **可见集合是瞬时快照**：每次状态变化后重新计算当前“应该显示哪些 group/node/edge”。
- `diffAndPatch(prevVisible, nextVisible)` 只处理“新增 / 复用 / 归还池”三类变化。

### 2.2 数据结构

Phase 2 引入三张主池：

```text
nodePool  : Map<nodeId, NodeView>
groupPool : Map<groupId, GroupView>
edgePool  : Map<edgeKey, EdgeView>
```

其中：

- `nodeId` 直接复用前端序列化后的稳定叶子节点 id。
- `groupId` 直接复用 `DATA.groups[].id`。
- `edgeKey` 复用现有 `edgeKey(edge)` 规则，即 `from || to || type || parent_class` 的拼接键，保证与现运行时一致。

此外引入以下运行时状态：

```text
visibleNodeIds  : Set<nodeId>
visibleGroupIds : Set<groupId>
visibleEdgeKeys : Set<edgeKey>
focusStack      : string[]
selectedGroupId : string | null
```

说明：

- `visible*` 表示“当前这一帧应该显示”的集合，而不是池内对象是否存在。
- `focusStack` 是 Semantic Zoom 的唯一真值。
- `selectedGroupId` 是 panel 当前选中的 group，用于单击 show panel。

### 2.3 View 类型定义

#### 2.3.1 NodeView

`NodeView` 表示一个叶子节点对应的 Pixi 图元集合。结构如下：

```text
NodeView {
  id: string,
  root: PIXI.Container,
  box: PIXI.Graphics,
  titleText: PIXI.Text | null,
  subText: PIXI.Text | null,
  visible: boolean,
  snapshot: {
    x, y, w, h,
    label,
    sublabel,
    fill,
    stroke,
    alpha
  }
}
```

说明：

- `root` 作为该节点的总容器，便于整体移动/隐藏。
- `snapshot` 只存**上次成功绑定到图元上的渲染态**，用于 patch 时做字段级比较。
- `visible=false` 只表示当前从 scene 隐藏，不代表该对象从池中删除。

#### 2.3.2 GroupView

`GroupView` 表示一个 group/container 对应的 Pixi 图元集合：

```text
GroupView {
  id: string,
  root: PIXI.Container,
  box: PIXI.Graphics,
  headerText: PIXI.Text | null,
  timingText: PIXI.Text | null,
  infoBadge: PIXI.Graphics | PIXI.Container | null,
  visible: boolean,
  interactionEnabled: boolean,
  snapshot: {
    x, y, w, h,
    label,
    type,
    collapsed,
    alpha,
    childCount,
    metaVersion
  }
}
```

额外约束：

- `GroupView.box` 是 Phase 2 的主要交互命中对象。
- `GroupView.box.eventMode = 'static'`，并负责承接 click / dblclick / right-button dblclick。
- 父 group 在 Semantic Zoom focus 态下允许“淡出但保留”，因此 `alpha` 是 `snapshot` 的一部分。

#### 2.3.3 EdgeView

`EdgeView` 表示一条边对应的 Pixi 图元：

```text
EdgeView {
  key: string,
  root: PIXI.Container,
  path: PIXI.Graphics,
  arrow: PIXI.Graphics | null,
  visible: boolean,
  snapshot: {
    points,
    stroke,
    strokeWidth,
    alpha,
    dashed
  }
}
```

Phase 2 不改变 edge 的几何计算方式，只改变 edge 图元的生命周期管理方式。

### 2.4 池初始化与生命周期

#### 2.4.1 初始化阶段

初始渲染流程改为：

1. 初始化 Pixi 舞台。
2. 初始化三张池（空 Map）。
3. 计算首屏可见集合。
4. 执行一次“全量 patch”到对象池。
5. 记录 `visibleNodeIds / visibleGroupIds / visibleEdgeKeys`。

等价于：

```text
initEngine()
  -> initPools()
  -> computeVisibleScene()
  -> diffAndPatch(emptyVisible, firstVisible)
```

**`resetScene()` 只允许在初始化时调用一次。**

原因：

- Phase 1 的 `resetScene()` 会清空 layer children，这与对象池的“长期持有”模型冲突。
- 一旦对象池建立完成，后续 toggle / Expand All / Collapse All / Semantic Zoom 都只能走 patch 路径。

#### 2.4.2 归还池而非销毁

当对象从 `nextVisible` 中消失时，不销毁，而是：

- `root.visible = false`
- `interactionEnabled = false`（group）
- 维持其 `snapshot`，保留在对应 Map 中，供下一次复用

禁止的做法：

- `destroy()` 图元
- `removeChildren()` 整层清空
- 在池外额外维护一套 fallback 临时对象

如果后续需要彻底清空（例如 hard reset / 页面销毁），应由单独的 `destroyScenePools()` 负责统一销毁，而不是在日常 toggle 中做。

### 2.5 `diffAndPatch(prevVisible, nextVisible)` 算法

#### 2.5.1 输入/输出

输入：

- `prevVisible = { nodeIds, groupIds, edgeKeys }`
- `nextVisible = { nodeIds, groupIds, edgeKeys, nodeSnapshots, groupSnapshots, edgeSnapshots }`

输出：

- 对象池与 Pixi scene 更新完成
- 运行时 `visible*` 集合同步到 `nextVisible`

#### 2.5.2 核心步骤

对每种对象分别做三段式处理：

1. **新增**：`nextVisible` 有，`prevVisible` 没有
   - 从池里查对象。
   - 若池中没有，则 create。
   - 将其 `visible = true`。
   - 用当前 snapshot 全量绑定。

2. **复用**：`prevVisible` 与 `nextVisible` 都有
   - 从池中取现有 view。
   - 比较 snapshot 字段。
   - 只 patch 改动项：`position / size / label / style / alpha`。
   - **不得**重建整套 DisplayObject。

3. **归还**：`prevVisible` 有，`nextVisible` 没有
   - 设 `view.root.visible = false`。
   - group 额外设 `box.eventMode = 'none'`。
   - 保留对象于池中，不 destroy。

#### 2.5.3 伪代码

```text
function diffAndPatch(prevVisible, nextVisible) {
  patchGroups(prevVisible.groupIds, nextVisible.groupIds, nextVisible.groupSnapshots)
  patchNodes(prevVisible.nodeIds, nextVisible.nodeIds, nextVisible.nodeSnapshots)
  patchEdges(prevVisible.edgeKeys, nextVisible.edgeKeys, nextVisible.edgeSnapshots)

  visibleGroupIds = new Set(nextVisible.groupIds)
  visibleNodeIds = new Set(nextVisible.nodeIds)
  visibleEdgeKeys = new Set(nextVisible.edgeKeys)
}
```

其中 `patchGroups` 的逻辑最重要，因为 Phase 2 的所有交互都挂在 GroupView 上：

```text
function patchGroups(prevIds, nextIds, snapshotMap) {
  for gid in nextIds:
    view = groupPool.get(gid)
    if (!view):
      view = createGroupView(gid)
      groupPool.set(gid, view)
    view.root.visible = true
    patchGroupView(view, snapshotMap.get(gid))

  for gid in prevIds - nextIds:
    view = groupPool.get(gid)
    if (!view): raise
    view.root.visible = false
    view.box.eventMode = 'none'
}
```

#### 2.5.4 patch 规则

`patch*View()` 必须遵循“字段级 patch、显式报错、无静默路径”的原则：

- `snapshot` 缺字段：直接 throw，不得补默认值。
- 发现 view 存在但关键 DisplayObject 缺失：直接 throw。
- label 变化只更新 `Text.text`，不重新 new Text。
- 位置变化更新 `root.x / root.y` 或图元几何；样式变化才重画 Graphics。
- group 的交互事件只在 create 阶段绑定一次；patch 阶段只改 handler 所读状态，不重复绑监听。

### 2.6 与布局计算的关系

对象池不负责布局，它只消费布局结果。

Phase 2 保持如下边界：

```text
状态变更（collapsed / focus / selected）
  -> computeLayout()
  -> computeVisibleScene()
  -> diffAndPatch()
```

也就是说：

- **布局仍由现有 `computeFlowchartLayout()` 及其下游逻辑负责**。
- 对象池只看最终“哪些 node/group/edge 应显示、每个对象的几何和样式是什么”。

这样可以把 Phase 2 的复杂度收束在渲染层，而不扩散到布局层。

---

## 3. Progress Overlay 修复

### 3.1 当前 bug 根因

当前 overlay 残留的根因不是“hide 没写”，而是**generation 校验时机错误**。

现状链路：

1. `canvasRenderPhase1()` 在完成阶段先执行：
   - `assertActiveRenderGeneration(generation, '完成阶段')`
2. 若此时用户又触发了一次新 render，`renderGeneration` 已递增。
3. `assertActiveRenderGeneration()` 在 `frontend_html.py` 当前实现里直接 throw。
4. 由于 throw 发生在 `await hideRenderProgress()` 之前，旧 generation 的 overlay 永远没有执行收尾。
5. 结果是：新 render 已经开始，但旧 overlay 还挂在屏幕上。

对应源码证据：

- `scripts/render_canvas.js:1469-1470`
- `scripts/frontend_html.py:1450-1471`

### 3.2 修复原则

Phase 2 对 generation 过期做分类处理：

- **真实错误**：仍然 throw。
- **generation 过期**：视为预期并发竞态，属于“旧 render 被新 render 抢占”，不再 throw，而是正常退出。

这不是 fallback，而是明确区分两类语义：

1. “运行时缺少依赖 / 数据损坏 / 调用顺序错误”是错误。
2. “旧 render 已经过期，不应继续动 UI”不是错误，而是正常控制流。

### 3.3 接口改动

将：

```text
assertActiveRenderGeneration(generation, label)
```

的语义从“过期就 throw”调整为：

- 仍校验 generation 是否匹配。
- 匹配返回 `true`。
- 不匹配返回 `false`。
- 仅在入参非法、运行时状态缺失等真实异常情况下 throw。

即：

```text
function assertActiveRenderGeneration(generation, label) {
  if (typeof generation !== 'number' || !Number.isFinite(generation)) throw
  if (typeof renderGeneration === 'undefined') throw
  if (generation !== renderGeneration) return false
  return true
}
```

### 3.4 调用方改动

所有调用方必须显式检查返回值。

统一规范：

```text
if (!assertActiveRenderGeneration(generation, label)) {
  await hideRenderProgress()
  return
}
```

要求：

- 对于任何“提前退出”路径，都必须先 `await hideRenderProgress()`。
- 不允许“return 了但 overlay 没 hide”。
- 不允许把 hide 放到调用方之外赌外层 finally 自动处理；要在退出点就地收尾。

### 3.5 `hideRenderProgress()` 的配套修法

由于 `hideRenderProgress()` 当前内部也会因为 ownerGeneration 过期而 throw（`scripts/frontend_html.py:1455-1462`），Phase 2 还需要同步改掉这一点。

新的语义应为：

- 如果 overlay 已经被新 generation 接管，则旧 generation 的 hide 应视为“豁免退出”，直接完成隐藏收尾或短路返回，但**不得 throw**。
- 其余 DOM 缺失、元素不完整等仍是 hard error。

推荐收口方式：

1. `showRenderProgress()` 为 overlay 打上当前 generation。
2. `hideRenderProgress()` 读取 overlay 上的 ownerGeneration。
3. 若 ownerGeneration 已不是当前 generation：
   - 说明 overlay 已被新 render 复用。
   - 旧 render 不再抢 UI 所有权，直接返回。

### 3.6 修复后伪代码

```text
async function finishRenderGeneration(generation) {
  if (!assertActiveRenderGeneration(generation, '完成阶段')) {
    await hideRenderProgress()
    return
  }

  updateLegendAndSummary(DATA)

  if (!assertActiveRenderGeneration(generation, 'hide progress')) {
    await hideRenderProgress()
    return
  }

  await hideRenderProgress()
}
```

更严格一点，Phase 2 的规范是：**任何 return 前必须保证 overlay 已进入隐藏态。**

---

## 4. 双击 toggle + 单击 show panel

### 4.1 交互协议

用户已经明确确认的协议如下：

- **单击 group**：show panel
- **双击 group**：toggle collapse / expand
- **右键双击 group**：Semantic Zoom zoom in
- **ESC**：退出当前 focus 层

Phase 2 本节只覆盖前两条；右键双击与 ESC 见第 6 节。

### 4.2 事件绑定位置

Group 交互事件统一挂在 `GroupView.box` 上，而不是 DOM overlay 上。

绑定要求：

```text
groupView.box.eventMode = 'static'
```

并绑定以下事件：

- `click` → `engine.onGroupSelect(gid)`
- `dblclick` → `engine.onGroupToggle(gid)`

这里的关键设计不是“把逻辑直接写在 `render_canvas.js`”，而是**把运行时状态真值继续放在 inline runtime 里，由 render engine 只负责发出事件**。

### 4.3 为什么不在 `render_canvas.js` 直接改 `collapsedState`

当前 `collapsedState` 真值在 `frontend_html.py` 的 inline runtime 内（`scripts/frontend_html.py:254`, `1528-1531`, `1796-1802`）。

Phase 2 不复制这份状态，也不在 engine 内再维护一套折叠状态镜像。原因：

1. 真值只能有一份，否则 expand-all / collapse-all / breadcrumb 跳转会出现状态分裂。
2. `frontend_html.py` 已经持有 panel、按钮、keyboard 等全局交互逻辑，继续作为运行时编排层更自然。
3. `render_canvas.js` 应只负责渲染和 scene 生命周期，不吞业务状态。

因此，Phase 2 定义两条由 inline runtime 注入给 engine 的接口：

```text
engine.onGroupToggle(gid)
engine.onGroupSelect(gid)
```

### 4.4 `engine.onGroupToggle` 语义

`onGroupToggle(gid)` 在 inline runtime 侧完成以下逻辑：

1. 校验 `gid` 存在。
2. 更新 `collapsedState[gid] = !collapsedState[gid]`。
3. 若当前 `focusStack` 非空且目标 group 位于 focus path 内，focus 保持不变。
4. 重新计算 layout / visible scene。
5. 调 `diffAndPatch()`。
6. 不调用 `resetScene()`。
7. 不调用 `renderer.resize()`。
8. 不 auto-fit。

伪代码：

```text
engine.onGroupToggle = function (gid) {
  if (!groupMap[gid]) throw new Error(...)
  collapsedState[gid] = !collapsedState[gid]
  invokeIncrementalRender({ reason: 'toggle', gid })
}
```

### 4.5 `engine.onGroupSelect` 语义

`onGroupSelect(gid)` 负责 panel 展示，而不修改折叠状态。

逻辑：

1. `selectedGroupId = gid`
2. 读取 group 元数据
3. 调现有 panel 渲染入口
4. 打开 side-panel

伪代码：

```text
engine.onGroupSelect = function (gid) {
  const group = groupMap[gid]
  if (!group) throw new Error(...)
  selectedGroupId = gid
  showGroupPanel(group)
}
```

### 4.6 Group Panel 展示字段

panel 内容复用当前 SVG 版 side-panel 的 DOM 容器，不额外新建一套 panel 壳子。Phase 2 至少展示以下字段：

- `gid`
- `label`
- `node_type`
- `class_name`
- `attr_name`
- `depth`
- `collapsed` 当前状态
- `children_group_ids.length`
- `children_nodes.length`
- `has_timing` / `has_phase_timing`
- timing 摘要（若已有）
- `src_file` / `src_start_line` / `class_def_loc` / `def_loc`（若存在）

说明：

- 字段以“现有 SVG 版 panel 已展示的元数据能力”为参考。
- 不在 Phase 2 新造 trace 数据；只展示已有 JSON 中已存在的数据。
- 若某字段缺失，应按已有字段真实情况展示，不得前端猜值。

### 4.7 click / dblclick 冲突处理

浏览器原生事件里，双击通常会先产生两次 click，再产生 dblclick。若不处理，会出现：

- 用户双击 toggle 时，panel 先弹一次 / 两次，再触发 toggle。

Phase 2 必须显式处理该冲突。推荐方式：

- `click` 走一个短延迟确认（如 200~250ms）。
- 若在延迟窗口内收到 `dblclick`，取消单击处理。
- 最终保证：
  - 单击只 show panel
  - 双击只 toggle，不额外触发 panel

这是交互协议的一部分，不是实现细节可随意更改。

### 4.8 stopPropagation 要求

Group 图元事件必须阻止冒泡到舞台级 pan / zoom 手势层。

原因：

- 单击 group 不能被解释成拖拽起点。
- 双击 group 不能同时触发舞台层的其他双击逻辑。

因此，GroupView 的 click / dblclick / right-button dblclick 处理函数都必须执行 `stopPropagation`。

---

## 5. Expand/Collapse All 增量路径

### 5.1 当前旧路径

当前按钮逻辑仍在 `frontend_html.py`：

- Expand All：遍历 `DATA.groups` 把 `collapsedState[g.id] = false`，然后 `invokeRender({ autoFit: true })`
- Collapse All：遍历 `DATA.groups` 批量折叠后，同样 `invokeRender({ autoFit: true })`

源码证据：`scripts/frontend_html.py:1796-1802`

这条旧路径的问题不在“慢”，而在**它仍然把全局状态切换绑定到首屏渲染链路**。

### 5.2 renderer.resize 问题根因

Expand All / Collapse All 当前会重走 `canvasRenderPhase1()`，而 `canvasRenderPhase1()` 在 `wantAutoFit` 为真时会执行 `performAutoFit()`。

`performAutoFit()` 的关键行为：

1. 用 width-only 规则计算 scale。
2. 根据内容高度计算 `canvasHeight`。
3. 再次执行 `renderer.resize(cw, canvasHeight)`。

这意味着“全图状态切换”会重复触发 renderer 高度调整。该链路对首屏 render 是合理的，但对交互期的 expand-all / collapse-all 是错误职责分配：

- 交互期用户希望保留当前视口。
- 但 auto-fit / resize 会主动改 canvas 尺寸和视口。
- 这会造成进度条、页面滚动位置、图像显示比例都被牵连。

因此，Phase 2 的根本修法不是“继续在旧路径上修补”，而是**让 Expand/Collapse All 脱离 `invokeRender({ autoFit: true })`**。

### 5.3 新路径设计

#### 5.3.1 `expandAll()`

新语义：

```text
collapsedState = {}
computeLayout()
diffAndPatch()
```

硬约束：

- 不调用 `resetScene()`
- 不调用 `renderer.resize()`
- 不 auto-fit
- 保持当前 viewport scale / offset 不变

#### 5.3.2 `collapseAll()`

新语义：

```text
collapsedState = { 所有 group: true }
computeLayout()
diffAndPatch()
```

折叠范围以当前语义为准：如果生产逻辑仍要求 depth 0 根 group 不折叠，则在状态构建阶段按现有规则显式跳过；但无论规则如何，**都必须通过统一的增量路径落地**，而不是再走 `invokeRender({ autoFit: true })`。

### 5.4 新旧路径对比

| 项目 | 旧路径 | 新路径 |
| --- | --- | --- |
| 状态更新 | 批量改 `collapsedState` | 批量改 `collapsedState` |
| 渲染入口 | `invokeRender({ autoFit: true })` | `invokeIncrementalRender({ reason: 'expand-all/collapse-all' })` |
| scene 生命周期 | `resetScene()` 全清空 | 对象池 patch |
| renderer.resize | 会触发 | 不触发 |
| autoFit | 会触发 | 不触发 |
| viewport 保持 | 不保证 | 保持当前状态 |
| overlay | 走旧 overlay 链路 | 走修复后的轻量收尾链路 |

### 5.5 推荐伪代码

```text
function expandAll() {
  collapsedState = {}
  invokeIncrementalRender({ reason: 'expand-all' })
}

function collapseAll() {
  collapsedState = buildCollapseAllState()
  invokeIncrementalRender({ reason: 'collapse-all' })
}

async function invokeIncrementalRender(ctx) {
  const nextScene = computeVisibleScene(DATA, collapsedState, focusStack)
  diffAndPatch(currentVisibleScene, nextScene)
}
```

按钮绑定同步改为：

- `btn-expand-all` → `expandAll()`
- `btn-collapse-all` → `collapseAll()`

不再调用 `invokeRender({ autoFit: true })`。

---

## 6. Semantic Zoom 设计

### 6.1 目标

Semantic Zoom 不是普通 viewport zoom。它不是“放大当前图像”，而是**按语义聚焦到某个 group 的局部子图**。

Phase 2 约定的 drill-down focus 目标是：

- 进入某个 group 后，不再显示整图，只显示与该 group 直接相关的“局部语义闭包”。
- 用户可以逐级深入，也可以沿 breadcrumb 回退到任意上层。

### 6.2 触发与退出协议

- **触发**：右键双击 group box → zoom in
- **退出**：
  - focus 态下再次右键双击当前层 group → 弹栈一级
  - 或按 `ESC` → 弹栈一级
- **空栈**：表示全图模式

### 6.3 focus 栈

运行时新增：

```text
focusStack: string[]
```

规则：

- zoom in 某 group：`focusStack.push(gid)`
- 退出一级：`focusStack.pop()`
- 点击 breadcrumb 第 k 级：截断为 `focusStack.slice(0, k)`
- `focusStack.length === 0`：恢复全图模式

硬约束：

- `focusStack` 是唯一真值，不再额外维护 `focusedGroupId` 的并行状态。
- 任何显示逻辑都从 `focusStack[focusStack.length - 1]` 推导当前 focus root。

### 6.4 进入 focus 时的状态规则

进入某个 group `G` 的 focus 态时：

1. `G` 必须强制展开。
2. 若 `collapsedState[G] === true`，先显式改为 `false`。
3. 重新计算可见集合。
4. 只保留：
   - `G` 的所有子孙 group
   - `G` 的所有子孙 leaf node
   - 与 `G` 子树之间存在直接连接的一跳外部 src/dst 节点
   - 上述节点之间的相关边
5. 其余对象全部 `visible = false`，但仍留在对象池中。

### 6.5 可见集合计算算法

设：

- `Desc(G)`：group `G` 的所有子孙 group + node
- `Boundary(G)`：与 `Desc(G)` 内任一节点存在直接 edge 连接、但自身不属于 `Desc(G)` 的外部一跳节点

则 focus 模式下的目标集合定义为：

```text
VisibleNodes  = DescLeafNodes(G) ∪ Boundary(G)
VisibleGroups = DescGroups(G) ∪ AncestorPath(G)
VisibleEdges  = 所有满足 from/to 均属于 VisibleNodes 或 VisibleGroups 映射端点的边
```

这里 `AncestorPath(G)` 保留的原因是：

- 用户需要看到当前 focus 在整棵 group 树中的父链位置。
- 但父链不应继续强交互、也不应抢视觉中心，因此只做“淡出保留”。

### 6.6 父 group 淡出规则

当 focus 到子 group 时，父 group 仍显示，但必须弱化：

- `alpha = 0.3`（建议值）
- `eventMode = 'none'`
- 不响应 click / dblclick / right-dblclick

目的：

1. 让用户保留上下文感知。
2. 避免焦点外父层抢交互。
3. 让 breadcrumb 成为唯一“跨层跳转”的显式入口。

### 6.7 右键双击的事件约定

Semantic Zoom 使用 **右键双击**，因此事件侧要满足两点：

1. 阻止浏览器原生 context menu。
2. 正确区分左键双击 toggle 与右键双击 zoom。

推荐方案：

- 监听 `pointerdown` / `contextmenu` / 双击计时窗口。
- 仅当 `button === 2` 且两次点击命中同一 group 且间隔小于阈值时，判定为 right-button dblclick。

Phase 2 必须避免与普通 `dblclick` 混淆，不能让一次右键双击同时触发左键双击逻辑。

### 6.8 breadcrumb 结构

breadcrumb 为左上角 HTML overlay，不放进 Pixi scene。

原因：

- 它是全局 UI，而不是图内对象。
- 放在 DOM overlay 层更容易做点击、截断和样式管理。

结构示意：

```text
全图 > GroupA > GroupB > GroupC
```

规则：

- `全图` 永远存在。
- 每一级 group 名称来自该 group 的 `label` / `attr_name` 的展示名称。
- 点击任意一级，直接将 `focusStack` 截断到该层。
- 点击 `全图` 时，`focusStack = []`。

### 6.9 focus 渲染流程

推荐统一到一条入口：

```text
function enterFocus(gid) {
  if (!groupMap[gid]) throw
  collapsedState[gid] = false
  focusStack.push(gid)
  invokeIncrementalRender({ reason: 'focus-enter', gid })
  renderBreadcrumb()
}

function exitFocus() {
  if (focusStack.length === 0) return
  focusStack.pop()
  invokeIncrementalRender({ reason: 'focus-exit' })
  renderBreadcrumb()
}
```

`invokeIncrementalRender()` 内部根据 `focusStack` 是否为空，决定走全图可见集合还是 focus 可见集合。

---

## 7. 测试矩阵

Phase 2 的测试以“每个子任务至少有一条直接回归点”为原则，统一编号 T1-T14。

| 编号 | 子任务 | 测试点 | 预期 |
| --- | --- | --- | --- |
| T1 | 对象池初始化 | 首次渲染后 `nodePool/groupPool/edgePool` 已建立 | 三张 Map 非空且 key 数与首屏可见对象一致 |
| T2 | 对象池复用 | 单 group toggle 前后，不变对象的 View 引用保持一致 | 未变化对象不重新创建 |
| T3 | 对象池归还池 | group 折叠后子孙对象从可见集合消失 | 对应 view `visible=false`，但仍保留在池中 |
| T4 | overlay 修复 | generation 过期时调用 `assertActiveRenderGeneration()` | 返回 `false`，不 throw |
| T5 | overlay 收尾 | 任一提前退出路径 | `hideRenderProgress()` 已执行，overlay 不残留 |
| T6 | 单击 group | 点击 group box | 调用 `engine.onGroupSelect(gid)`，panel 打开且展示 group 元数据 |
| T7 | 双击 group | 双击 group box | 调用 `engine.onGroupToggle(gid)`，`collapsedState[gid]` 翻转 |
| T8 | click/dblclick 冲突 | 双击 group 时 | 不额外触发单击 panel 打开 |
| T9 | Expand All 增量路径 | 点击 Expand All | 不调用 `resetScene`、不调用 `renderer.resize`、不 auto-fit |
| T10 | Collapse All 增量路径 | 点击 Collapse All | 同 T9 |
| T11 | Semantic Zoom 进入 | 右键双击 group | `focusStack` 压栈，目标 group 强制展开，仅显示局部子图 + 一跳边界 |
| T12 | Semantic Zoom 退出 | focus 态按 ESC 或再次右键双击 | `focusStack` 弹栈一级，breadcrumb 同步更新 |
| T13 | 父 group 淡出 | 进入子 group focus | 父链 group `alpha=0.3` 且 `eventMode='none'` |
| T14 | breadcrumb 跳转 | 点击 breadcrumb 任意层 | `focusStack` 截断到目标层，画面回到对应 focus 级别 |

测试分层建议：

1. **运行时级单测**：验证 `diffAndPatch()`、可见集合计算、focusStack 状态机。
2. **浏览器交互测试**：验证 click / dblclick / right-dblclick / ESC / breadcrumb。
3. **视觉回归**：验证 focus 场景下对象显隐、父层淡出、panel 与 breadcrumb 是否同时正确。

---

## 8. 实现顺序与依赖关系

Phase 2 必须按以下顺序实施，原因是前面的结果是后面的依赖：

### 8.1 步骤 1：对象池

先落对象池与 `diffAndPatch()`，因为后续所有交互都依赖“可增量更新”。

若没有对象池，后面的 toggle / Expand All / Semantic Zoom 仍然只能走 `resetScene()` 全量路径，设计目标无法成立。

### 8.2 步骤 2：Progress Overlay 修复

在增量路径接入前先修 overlay generation 竞态，确保后续即使多次快速切换状态，也不会把旧 overlay 留在画面上。

### 8.3 步骤 3：双击 toggle + 单击 show panel

在对象池和 overlay 修复完成后，再接 group 基础交互：

- 事件绑定
- `engine.onGroupToggle`
- `engine.onGroupSelect`
- group panel

这一阶段完成后，Canvas 版就具备最基本的“点得动”的交互能力。

### 8.4 步骤 4：Expand/Collapse All 增量路径

将工具栏按钮从旧的 `invokeRender({ autoFit: true })` 迁移到增量路径。此时对象池、toggle 入口、overlay 修复都已到位，迁移成本最低。

### 8.5 步骤 5：Semantic Zoom

最后接 Semantic Zoom，因为它依赖前面四项都已经稳定：

- 需要对象池支持大规模显隐切换
- 需要稳定的增量 patch 入口
- 需要 group 交互事件系统
- 需要 overlay 在频繁切换 focus 时不残留

### 8.6 依赖图

```text
对象池
  -> Progress Overlay 修复
  -> Group click/dblclick 交互
  -> Expand/Collapse All 增量路径
  -> Semantic Zoom
```

其中：

- **对象池** 是基础设施依赖。
- **Semantic Zoom** 是最末端能力，不应在对象池前抢跑。

---

## 9. 一句话总结

Phase 2 的本质不是“补几个交互事件”，而是把 Canvas 渲染器从一次性静态输出升级为**长期存活、可增量 patch、可 drill-down 聚焦的前端运行时**。技术收口点只有三个：

1. 用对象池取代 `resetScene()` 日常路径。
2. 用返回 `false` 的 generation 校验取代 overlay 竞态 throw。
3. 用 `collapsedState + focusStack + selectedGroupId` 三个明确真值驱动所有 Phase 2 交互。

只要这三个收口点守住，后续 Phase 3/4 的 hover、edge focus、更多 panel 与语义缩放扩展都能在同一运行时骨架上继续演进，而不需要再返工渲染生命周期设计。
