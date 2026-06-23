# Stage 1.5 — 真实 PixiJS v8 bundle 接入 + 可视图元/Text 落地 + 首屏 FitToView 详细设计

> 状态：**设计待 NaN review，不含任何实现代码**。review 通过后才实现。
> 上位文档：`design/frontend_canvas_renderer.md`、`design/frontend_canvas_phase1.md`、`design/frontend_canvas_phase1_stage1_4.md`
> 前置：Stage 1.1 ~ 1.4 已完成，当前 HTML 运行时已彻底切到 Canvas 路径，但 smoke / 5698781 仍未真正可视。

---

## 1. 本阶段目标与边界

### 1.1 本阶段要解决什么

本阶段只解决一个问题：**让当前 Canvas 产物真正“画出来、放对位置、首屏可见”**。

具体拆成 4 件事：

1. **引擎**：把当前故意不绘制的 `pixi_engine_bundle.js` 桩替换为真实 **PixiJS v8** bundle，仍保持单 HTML 内联注入。
2. **图元**：把 `render_canvas.js` 里现在只“创建对象但不成形”的 node / group / io / edge 绘制路径，改成真实 v8 图元绘制。
3. **文字**：给 Text 补齐坐标、样式、锚点、层级关系，不能再只创建文本对象不定位。
4. **视口**：首屏渲染完成后自动 `fitToView` 一次，并把按钮 `Fit to View` 从“只按宽度缩放”改成“按宽高联合缩放 + 居中 + 留白”。

### 1.2 本阶段明确不做什么

以下内容仍然**不在本阶段范围**：

- hover / click / edge evidence / source panel 命中回接（留给 Phase 3 / 4）
- group toggle 的增量动画
- 后端 JSON、布局算法、SOURCE_MAP、DOM 浮层协议修改
- 任何前端 trace 数据处理

### 1.3 必须保持不变的边界

| 边界 | 约束 |
| --- | --- |
| HTML 注入框架 | 继续走 `frontend_html.py::_generate_flowchart_html()` 的内联占位符注入，缺占位符仍然直接 `raise`，不加 fallback。见 `scripts/frontend_html.py:2328-2350` |
| 数据接口 | `const DATA = ...` 的注入方式、`groups/nodes/edges/io_groups/root_groups/meta` 结构不变。见 `scripts/frontend_html.py:2352-2359`、`design/frontend_canvas_renderer.md:67-89` |
| 布局算法 | 继续复用前端现有布局结果 `computeFlowchartLayout()` / `groupLayout` / `nodePortMap`，不下沉到后端。见 `scripts/render_canvas.js:895-914, 967-993` |
| DOM 浮层 | `tooltip` / `side-panel` / `render-progress-overlay` / `controls` / `legend` / `summary` 结构不动。见 `scripts/frontend_html.py:203-239` |

---

## 2. 现状基线（源码证据，不靠猜）

### 2.1 当前运行时已经没有 SVG fallback

`render()` 在 Phase 1 模式下只允许走 `window.__canvasRenderPhase1(DATA)`；如果该入口不存在，会直接抛错；legacy SVG 路径已经删除。见：

- `scripts/frontend_html.py:1514-1521`

这意味着：**要让页面可视，只能修当前 Canvas 路径，不能靠恢复 SVG 兜底。**

### 2.2 当前引擎 bundle 是“故意不绘制”的桩

`scripts/pixi_engine_bundle.js` 文件头已经写明：当前 bundle 是 *minimal, self-contained PIXI-compatible scene-graph runtime*，并且 *It intentionally draws nothing*，后续才由 real WebGL Pixi bundle 替换。见：

- `scripts/pixi_engine_bundle.js:1-14`
- `scripts/pixi_engine_bundle.js:48-55`（`Graphics` 只有 `clear()`，没有真实绘制实现）
- `scripts/pixi_engine_bundle.js:71-86`（`Application` 只造 canvas / stage，不负责真实渲染）

### 2.3 当前 node / group / io 只是“记账”，没有真正落图元

当前 `render_canvas.js` 的现状不是“画错了”，而是**大部分核心图元根本还没画**：

1. `addLabel()` 只是 `layer.addChild(makeText(...))`，没有设置任何 `x/y/anchor/style`。见：
   - `scripts/render_canvas.js:240-248`
2. `drawNode()` 只 `addChild(makeGraphics('node-box:...'))`，随后把结构信息 push 到 `engine.nodes`，但没有任何矩形几何绘制。见：
   - `scripts/render_canvas.js:425-447`
3. `drawCollapsedGroup()` / `drawExpandedGroupShell()` 同样只创建空 `Graphics` 和未定位的 label。见：
   - `scripts/render_canvas.js:451-476`
   - `scripts/render_canvas.js:478-502`
4. `IOLayer.drawPill()` 也只是创建空 `Graphics` + label/sublabel，没有 pill 几何与文字定位。见：
   - `scripts/render_canvas.js:744-760`

所以当前页面空白不是单点故障，而是三个事实叠加：

- bundle 本身不绘制；
- 图元对象没落几何；
- Text 也没落坐标。

### 2.4 当前 edge 仍然是 Pixi v7 风格调用

虽然 `EdgeRoute` / `EdgeBatch` 已经把边几何和分桶整理好了，但真正下发给 Graphics 的 API 仍是旧风格：

- `g.lineStyle(...)` 画线：`scripts/render_canvas.js:671-680`
- `g.beginFill(...)` + `g.drawPolygon(...)` + `g.endFill()` 画箭头：`scripts/render_canvas.js:682-706`

这和本任务当前选型的 **PixiJS v8** builder 风格不一致。官方 v8 migration guide 明确要求“先构形，再 stroke/fill”；Application 也改成 `new Application()` 后 `await app.init(...)` 的异步初始化。参考：

- PixiJS v8 Migration Guide: <https://pixijs.com/8.x/guides/migrations/v8>
- PixiJS v8 Application Docs: <https://pixijs.download/dev/docs/app.Application.html>
- PixiJS v8 Graphics Guide: <https://pixijs.com/8.x/guides/components/scene-objects/graphics>

### 2.5 当前首屏不会自动 fit，按钮 fit 也只看宽度

1. 首次渲染结束后，`canvasRenderPhase1()` 直接 `return buildSnapshot()`，并没有自动执行 `fitToView()`。见：
   - `scripts/render_canvas.js:967-1017`
2. 页面里只有按钮点击才会触发 fit。见：
   - `scripts/frontend_html.py:1800-1810`
3. 当前 `fitToView(worldBounds, containerWidth)` 只用 `containerWidth` 和 `bounds.w` 算缩放，完全不看高度、也不做居中和 padding。见：
   - `scripts/render_canvas.js:340-350`
4. `buildEngine()` 初始化时 viewport 默认 `scale=1, x=0, y=0`。见：
   - `scripts/render_canvas.js:136-159`

因此当前首屏语义实际上是：**按原始世界坐标左上角、1 倍缩放直接显示**。对 smoke 可能只是看着挤在角落；对 5698781 这类大图，首屏基本不满足“可视可 review”。

---

## 3. 根因总结

基于上面的源码证据，本阶段根因可以收敛成 4 条：

1. **引擎根因**：当前 bundle 是占位桩，设计上就不会真正 rasterize。来源：`scripts/pixi_engine_bundle.js:1-14`
2. **图元根因**：node/group/io 只创建空 `Graphics`，没有矩形/圆/边框几何。来源：`scripts/render_canvas.js:425-447, 451-502, 744-760`
3. **文本根因**：Text 只创建对象、不设置位置和样式。来源：`scripts/render_canvas.js:240-248`
4. **视口根因**：首屏不自动 fit，现有 fit 算法也只按宽度缩放，缺少高度约束和居中。来源：`scripts/render_canvas.js:340-350, 967-1017`；`scripts/frontend_html.py:1800-1810`

---

## 4. 旧实现 → 新实现 映射

| 当前实现 | 新实现 | 设计要求 |
| --- | --- | --- |
| `pixi_engine_bundle.js` 最小桩 | vendored 真实 PixiJS v8 bundle（仍内联进 HTML） | 继续由 `frontend_html.py::_generate_flowchart_html()` 注入；缺占位符继续 `raise` |
| `new PIXI.Application({ ... })` 同步构造 | `const app = new PIXI.Application(); await app.init({...})` | 适配 v8 官方初始化模型；渲染 bootstrap 需变成 async |
| `makeGraphics()` + `addChild()` 空对象 | `Graphics` builder 真实落图元 | node/group/io 要能画出 rounded rect / circle / port / info hit / arrow |
| `addLabel()` 无坐标 | `Text` / `TextStyle` + 显式 `x/y/anchor` | label / sublabel / timing / info 字母都必须定位 |
| `lineStyle` / `beginFill` / `drawPolygon` | v8 风格 `stroke` / `fill` / `poly` / builder shape API | 不留 v7 风格生产路径 |
| 首屏不 fit | 首次 render 结束后自动 fit 一次 | 只在首屏或渲染代次变化后自动执行，避免覆盖用户后续 pan/zoom |
| `fitToView(worldBounds, containerWidth)` | `fitToView(worldBounds, containerWidth, containerHeight, options)` | 同时考虑宽高、padding、居中和 min/max scale |

---

## 5. 详细设计

### 5.1 真实 PixiJS v8 bundle 接入

#### 5.1.1 接入方式

保持现有单 HTML 注入链路不变：

- bundle 文件位置仍是 `scripts/pixi_engine_bundle.js`
- Python 仍从 `_ENGINE_BUNDLE_PATH` 读取文本并替换 `__ENGINE_BUNDLE_PLACEHOLDER__`
- 缺占位符继续硬报错

对应现有入口：

- `scripts/frontend_html.py:41-44`
- `scripts/frontend_html.py:2333-2344`

#### 5.1.2 本阶段约束

- **不走 CDN**，不引外链 `<script src=...>`
- **不保留 stub / real 双路径 fallback**，生产文件直接替换成真实 bundle
- **headless probe 仍由 `render_canvas.js` 内部 `createHeadlessPixi()` 承担**；浏览器运行时必须优先使用真实 `window.PIXI`

#### 5.1.3 设计理由

当前 `resolvePixi()` 的优先级本来就是“先取 `window.PIXI`，拿不到再走 headless mock”。见：

- `scripts/render_canvas.js:38-44`

因此本阶段只要把 `window.PIXI` 提供成真实 v8 runtime，现有 headless 测试辅助分支仍可保留，不需要再额外发明 shim。

### 5.2 Canvas bootstrap 改成 v8 async init

#### 5.2.1 现状

当前 `buildEngine(container)` 内直接：

- `const app = new PIXI.Application({ width: 0, height: 0, antialias: true, backgroundAlpha: 0 })`
- 若 `app.canvas` 存在则 append 到容器

见：

- `scripts/render_canvas.js:118-123`

#### 5.2.2 新设计

- `initCanvasEngine()` / `ensureEngine()` / `canvasRenderPhase1()` 统一改成 async bootstrap 语义
- `buildEngine()` 负责：
  1. `new PIXI.Application()`
  2. `await app.init({...})`
  3. append `app.canvas`
  4. 创建 world + L0~L5 layers
- 初始化参数显式包含：`backgroundAlpha`、`antialias`、`resolution=devicePixelRatio`、`preference='webgl'`

#### 5.2.3 关键约束

- 初始化失败必须直接抛错，不吞异常
- 不能在 `app.init()` 完成前开始 draw
- 不新增“初始化失败就回退 stub 绘制”的软化路径

### 5.3 图元绘制落地（Node / Group / IO / Edge）

#### 5.3.1 Node / Group / IO 统一 painter 化

本阶段不再接受“先记 snapshot，图稍后再画”的状态。每个可见对象在 push snapshot 之前，必须已经完成真实几何绘制。

建议结构：

- `NodePainter.draw(node, rect)`
- `GroupPainter.drawCollapsed(group, rect)`
- `GroupPainter.drawExpanded(group, rect)`
- `PortPainter.draw(portPoint, style)`
- `IOPainter.drawPill(spec)`
- `EdgePainter.strokePolyline(points, style)`
- `EdgePainter.drawArrow(points, style)`

#### 5.3.2 v8 API 形态

本阶段统一采用 v8 builder 风格：

- 矩形 / 圆角矩形：`roundRect(...).fill(...).stroke(...)`
- 圆点 / port：`circle(...).fill(...)`
- 箭头三角形：`poly([...]).fill(...)`
- 折线：`setStrokeStyle(...); moveTo(...); lineTo(...); stroke()`

这里的重点不是 API 名字本身，而是**生产路径统一收敛到 v8 风格，不再继续混用 `lineStyle/beginFill/drawPolygon/endFill`**。

#### 5.3.3 样式来源

颜色与视觉参数继续复用现有语义，不改视觉分层：

- node 颜色继续来自 `getNodeColor()`：`scripts/render_canvas.js:187, 429`
- edge 颜色继续复用 `EDGE_STYLE`：`scripts/render_canvas.js:537-541`
- node / group 尺寸继续复用布局结果与 `nodePortMap`，不在本阶段重写尺寸规则

### 5.4 Text 定位与样式补齐

#### 5.4.1 现状问题

当前 `makeText()` / `addLabel()` 只负责“创建”和“挂载”，完全不管：

- `x / y`
- `anchor`
- `TextStyle`
- `resolution`
- 可见层级与跟随关系

见：`scripts/render_canvas.js:240-248`

#### 5.4.2 新设计

每一种文字都要有确定的布局规则：

| 文本类型 | 定位规则 |
| --- | --- |
| node 主标题 | 左上内边距对齐到 node box 内部 |
| node sublabel | 贴近主标题下方或右侧，不能与主标题重叠 |
| collapsed group 标题 | 左上 header 位置，沿用 `▶/▼` 视觉语义 |
| group timing | 右上或 header 区固定槽位，不和标题打架 |
| info `i` | 固定在 info 圆点中心 |
| io label / sublabel | 以 pill 中心或左对齐规则布置，保持折叠 / 展开两态一致 |

#### 5.4.3 文本实现约束

- 统一定义 `TextStyle` 常量，不允许散落 hardcode
- 高 DPI 下使用 `resolution = devicePixelRatio`
- label culling 逻辑继续保留，但“不可见时不创建文本”与“可见时必须准确定位”是两回事，不能混在一起

### 5.5 首屏自动 FitToView

#### 5.5.1 现状不足

当前按钮 fit 有两个问题：

1. 只能手动点，首屏不会自动 fit。见 `scripts/frontend_html.py:1800-1810, 1813`
2. 算法只按宽度缩放，不考虑高度。见 `scripts/render_canvas.js:340-350`

#### 5.5.2 新设计

`fitToView` 升级为：

```text
fitToView(worldBounds, containerWidth, containerHeight, {
  padding,
  minScale,
  maxScale,
  center = true,
})
```

核心语义：

- `scaleX = (containerWidth - 2*padding) / worldWidth`
- `scaleY = (containerHeight - 2*padding) / worldHeight`
- `scale = min(scaleX, scaleY)`
- 再按 `minScale/maxScale` 夹紧
- 计算 `viewport.x/y`，让世界包围盒在容器中居中，而不是贴左上角

#### 5.5.3 自动触发时机

自动 fit 只在下面两种时机触发：

1. **首屏第一次成功渲染后**
2. **点击 Expand All / Collapse All 导致整图重新布局后**

不在普通 hover / 面板交互里重复触发，避免覆盖用户手动拖拽后的视角。

### 5.6 渲染阶段编排

本阶段仍保留现有 3 段 render 进度骨架：

- 布局阶段
- 节点/容器阶段
- 连边阶段

见：`scripts/render_canvas.js:972-1016`

但补一条顺序约束：

```text
layout -> draw world geometry -> resize/flush -> auto fit -> update legend/summary -> hide progress
```

也就是：**auto fit 必须发生在 worldBounds 已确定、几何已落盘之后，但在 overlay 关闭之前完成。**

---

## 6. 验证设计（review 通过后执行，不在本文实现）

### 6.1 smoke / 5698781 的验收标准

本阶段只验“真正可视”，验收标准收敛到 5 条：

1. **画面非空**：打开 HTML 后，首屏无需点击任何按钮即可看到 DAG 主体
2. **图元成形**：node/group/io/edge 都有真实几何，不是只有 snapshot 有数据
3. **文字就位**：label / timing / info 不全部堆在 `(0,0)`
4. **首屏适配**：smoke 与 5698781 都能首屏 fit 到可 review 范围
5. **无 fallback**：真实 Pixi v8 路径失败时直接抛错，不回退旧 SVG / stub

### 6.2 后续实现时需要补的 UT 方向

本文不写测试代码，但 review 通过后，UT 方向至少包括：

- `snapshot.layers` 非零且与图元数量一致
- `labelsCreated > 0` 且文字坐标不全相同
- `fitToView` 同时受宽高约束
- 首屏自动 fit 后 `viewport.scale/x/y` 与期望一致
- 生成的 HTML 中确实内联真实 Pixi bundle，而不是当前占位桩注释文本

---

## 7. 待 NaN review 的点

1. **Stage 1.5 是否接受直接替换 `scripts/pixi_engine_bundle.js` 为真实 PixiJS v8 vendored bundle**，而不是额外引入第二个 bundle 文件。
2. **v8 绘制 API 风格是否按本方案统一收敛**：矩形/圆点/箭头/折线全部切到 builder 风格，不再保留 `lineStyle/beginFill/drawPolygon` 生产路径。
3. **首屏自动 fit 的触发时机** 是否接受“首次 render 成功后 + Expand/Collapse 后”，而不是只保留按钮手动触发。
4. **fit 算法是否改为宽高联合缩放 + 居中 + padding**；若你更希望保持“只按宽度 fit”的老语义，需要在这里明确。

---

## 8. 一句话总结

Stage 1.5 不改 DAG 数据、不改布局算法、不改 DOM 浮层；只做三件事：**把真 PixiJS v8 接进来、把图元/Text 真正画出来、把首屏自动 fit 到可 review 状态。**
