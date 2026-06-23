# Stage 1.4 — IO 层 / 视口裁剪 / 进度条 / Pan-Zoom-Fit + 旧 SVG 死代码清理 详细设计 + UT 详细设计

> 状态：**设计待 NaN review，不含任何实现代码**。review 通过后才实现。
> 上位文档：`design/frontend_canvas_phase1.md`、`design/frontend_canvas_phase1_stages.md`、`design/frontend_canvas_renderer.md`
> 前置：Stage 1.1/1.2/1.3 已合入（节点/容器/端口/连边均在 Canvas 上绘制）。
> 本阶段是 Phase 1 收口：补齐 IO/进度/视口交互（仅 pan/zoom/fit，不含节点交互），并**彻底删除运行期不再可达的 SVG 渲染死代码**，使 Phase 1 结束后运行时零 SVG 渲染路径。

---

## 1. 设计目标（本阶段范围）

1. **IO 层（L5）**：把 `renderIOGroupPill` / `renderIOPill` 迁到 `IOLayer`，snapshot.io_pills 真实化。
2. **进度条**：`__canvasRenderPhase1` 接入 3 阶段进度（复用 `setRenderProgress/runChunked/nextFrame/showRenderProgress/hideRenderProgress` + generation 守卫，不重写）。
3. **视口**：Viewport 支持拖拽 pan + 滚轮 zoom + Fit to View；修复 `btn-fit`（当前引用已删除的 `#dag-svg`，运行期点击即抛错）。
4. **视口裁剪（culling）**：`CullManager` 只对**屏幕外节点的文字**跳过创建，节点/边集合与布局**不受影响**。
5. **死代码清理**：删除 `frontend_html.py` 中 `render()` 之后整段 legacy SVG 渲染体、`RENDER_GROUP_EXPORTS` 残留、`svg` 变量、`#dag-svg` 相关 CSS、`btn-fit` 的 SVG 实现。

布局算法、后端 JSON、DOM 浮层（tooltip/side-panel/legend/summary/progress overlay）一行不动。节点/边的 hover/click/evidence 交互**不做**（Phase 3）。

---

## 2. 现状基线（源码证据）

| 事实 | 位置 |
| --- | --- |
| 进度 API：`setRenderProgress`（含 NaN 校验抛错）/ `showRenderProgress` / `hideRenderProgress`（generation 守卫）/ `runChunked`（generation + allowedTypes 守卫，未知 type 抛错）/ `nextFrame` | `frontend_html.py:1182-1285` |
| legacy 渲染阶段文案：准备/`正在渲染模块节点…`(30-60)/`正在渲染依赖边…`(60-90)/`正在更新图例和摘要…`(90-98)/`渲染完成`(100) | `frontend_html.py:2179,2216,2225,2268`、`hideRenderProgress:1220` |
| `__canvasRenderPhase1` 当前**不接进度条**（直接 layout+draw 返回） | `render_canvas.js:333-350` 区域 |
| IO pill：`renderIOGroupPill(ioGroup,cx,cy,w,h,availableW)`（折叠 pill / 展开网格） | `frontend_html.py:1975+` |
| IO leaf pill：`renderIOPill(nid,cx,cy,w,h,label,sublabel,fillColor)` | `frontend_html.py:1926+` |
| IO 任务收集：`collectIORenderTasks(row,startY)`（top + bottom 两行），展开列宽 `EXPAND_COL_GAP` | `frontend_html.py:2084+, 2123-2132` |
| `btn-fit` 引用 `#dag-svg`（已不存在 → 运行期点击抛错） | `frontend_html.py:2561-2572` |
| Viewport 当前 scale=1, x=y=0，无 pan/zoom；`cullingEnabled=false` | `render_canvas.js` snapshot 区 |
| `render()` 在 phase1 直接短路；其后整段 SVG body（节点/IO/边/legend/summary，含 `RENDER_GROUP_EXPORTS`、`svg.appendChild`）运行期不可达 | `frontend_html.py:1293-1298` 短路，body `~1299-2330` |
| `isEdgeVisible` 当前恒 `true`（边可见性当前为 no-op） | `frontend_html.py:367-369` |

---

## 3. 旧功能替换映射

| 旧实现（SVG/死路径） | 新实现（Canvas） | 等价定义 |
| --- | --- | --- |
| `renderIOGroupPill` 折叠 pill | `IOLayer.drawGroupPill()` collapsed | 颜色 `IO_GROUP_FILL`、成员标签 `IO_GROUP_MEMBER_LABEL`、(cx,cy,w,h) 保持 |
| `renderIOGroupPill` 展开网格 | `IOLayer.drawGroupPill()` expanded | 复用 `computeIOGroupExpandedLayout` / `calcIOGroupExpandedHeight` / `EXPAND_COL_GAP`，列数与列宽不变 |
| `renderIOPill` leaf | `IOLayer.drawPill()` | label/sublabel/fillColor 保持 |
| `collectIORenderTasks`（top/bottom 两行） | 复用同函数产 task，IOLayer 消费 | top/bottom 排布与 y 计算不变 |
| `btn-fit` 基于 `#dag-svg` `width` 算 scale | `fitToView()` 基于 Pixi world 包围盒 + 容器宽算 scale，写 `viewport.{scale,x,y}` | 「整图缩放到容器可见」语义保持；不再依赖 SVG |
| 无 pan/zoom | Viewport 拖拽平移 + 滚轮缩放（限幅） | Phase 1 新增能力（旧版本就没有，属于增强非回退） |
| 全量绘制文字 | `CullManager`：屏幕外节点 label 不创建 | 仅影响**文字创建**，节点矩形/边/坐标/snapshot 集合不变 |
| `render()` 后 legacy SVG body | **删除** | 运行期早已不可达，删除不改变任何可见行为 |
| `btn-fit`/`btn-expand-all`/`btn-collapse-all` 对 `invokeRender()` 的调用 | 保留按钮与 `invokeRender`，`btn-fit` 改走 `fitToView` | 展开/折叠按钮行为不变（仍触发重渲染） |

---

## 4. 功能等价性保证（怎么证明）

| 等价维度 | 保证方式 | 对应 UT |
| --- | --- | --- |
| IO 集合等价 | `snapshot.io_pills` 的 id/类型集合 == `DATA.io_groups`/IO 节点经 `collectIORenderTasks` 后的可见集 | P1-17 |
| IO 几何等价 | 展开列宽/列数来自未改的 `computeIOGroupExpandedLayout`；snapshot 记录每 pill `(cx,cy,w,h)` 与 legacy 计算值相等 | P1-17b |
| 进度阶段等价 | 进度只增不减、终值 100、`渲染完成` 收尾；overlay DOM 结构不变；generation 守卫语义保留（旧 render 被新 render 覆盖时抛 abort） | P1-18 / P1-18b |
| 进度错误语义等价（强化） | `setRenderProgress(NaN)` 抛错；`runChunked` 未知 type 抛错 | P1-E1 |
| Fit 语义等价 | `fitToView` 在 world 宽 > 容器宽时 `scale = 容器宽/world 宽`（同 legacy 公式），否则 scale=1；写入 viewport 并反映到 snapshot | P1-20 |
| Pan/Zoom 正确性 | pan 改 `viewport.x/y`；zoom 改 `viewport.scale` 且限幅 [min,max]；均反映到 snapshot | P1-21 / P1-22 |
| 裁剪不改变模型 | 开/关 culling，`snapshot.nodes`/`edges`/`groups`/`io_pills` 集合**完全相同**；仅 `labelsCreated` 数不同 | P1-19 |
| 零 SVG 运行路径 | 生成的 HTML 运行期无 `createElementNS`/`dag-svg`/`<marker>`/`RENDER_GROUP_EXPORTS` 调用 | P1-23（静态扫描） |

> Pan/Zoom 是新增能力，等价性表现为「不破坏已有渲染」：开启 pan/zoom 后 snapshot 的节点/边/IO 集合不变，只有 viewport 变换变化。

---

## 5. `render_canvas.js` 新增结构（设计，无实现代码）

- `IOLayer`：`drawGroupPill(ioGroup, geom, availableW)` / `drawPill(spec)` → L5；push `engine.io_pills`（snapshot 记 `{id,subtype,cx,cy,w,h,expanded}`）。
- `Viewport` 增强：`enablePan()` / `enableZoom({min,max})` / `fitToView(worldBounds, containerWidth)`；headless 下为纯对象，方法直接改 `viewport.{scale,x,y}`。
- `CullManager`：`isVisible(rect, viewportBounds)`；NodeView/GroupView 绘制时若不可见则跳过 label/text 创建（矩形仍画或仅记录）；`cullingEnabled=true`。
- `__canvasRenderPhase1` 改造：`showRenderProgress` → runChunked(布局阶段) → runChunked(节点/容器/IO 阶段) → runChunked(连边阶段) → `hideRenderProgress`；全程复用既有 generation 守卫。
- snapshot：`io_pills` 真实化；`viewport` 反映 pan/zoom/fit；新增 `labelsCreated`（供裁剪断言）；`cullingEnabled=true`。

---

## 6. UT 详细设计

测试文件：`testset/unit/test_frontend_canvas_phase1.py`（续）；helper 一律进 `frontend_test_infra.py`。

| ID | 测例 | 输入 fixture | 断言（等价点） |
| --- | --- | --- | --- |
| P1-17 | `test_snapshot_io_pills_match_data` | flowchart 含 input/output io_group + IO leaf | `snapshot.io_pills` id/subtype 集合 == 期望可见集；`layers.l5 > 0` |
| P1-17b | `test_io_group_expanded_layout_equivalent_to_legacy` | 1 展开 io_group（多成员） | 列数/列宽来自 `computeIOGroupExpandedLayout`，每 pill `(cx,cy,w,h)` == legacy 值 |
| P1-18 | `test_render_progress_three_stages_monotonic` | 最小 flowchart，hook `setRenderProgress` 调用序列 | 阶段文案含三段（布局/节点容器/连边）；percent 单调不减；终值 100 + `渲染完成` |
| P1-18b | `test_render_progress_overlay_dom_unchanged` | 生成 HTML | overlay/bar/stage/percent DOM 仍存在且 id 不变 |
| P1-19 | `test_culling_preserves_model_changes_only_labels` | 大 fixture，viewport 只覆盖部分区域 | culling on vs off：`nodes/edges/groups/io_pills` 集合相等；`labelsCreated(on) < labelsCreated(off)` |
| P1-20 | `test_fit_to_view_scale_matches_legacy_formula` | world 宽 > 容器宽 / world 宽 ≤ 容器宽 两组 | 前者 `viewport.scale == round(容器宽/world宽,3)`；后者 scale==1；不引用 `#dag-svg` |
| P1-21 | `test_pan_updates_viewport_offset` | 模拟 drag dx,dy | `viewport.x/y` 相应变化，节点集合不变 |
| P1-22 | `test_zoom_clamped_within_bounds` | 连续放大/缩小越界 | `viewport.scale` 限幅在 [min,max]；集合不变 |
| P1-23 | `test_generated_html_has_no_runtime_svg_render` | 生成 HTML 文本 | 无 `createElementNS` 调用、无 `getElementById('dag-svg')`、无 `<marker`、无 `RENDER_GROUP_EXPORTS`（死代码已删） |
| P1-E1 | `test_set_render_progress_invalid_raises` | `setRenderProgress(NaN)` / `runChunked` 未知 type | 抛错，不静默 |

覆盖率门槛（仅本阶段新增代码）：line ≥ 90%，branch ≥ 50%（IO 折叠/展开、fit 两分支、zoom 限幅上下界、culling 可见/不可见 全覆盖）。

---

## 7. 旧 UT / 旧实现清理动作

### 实现侧（`frontend_html.py`）
1. 删除 `render()` 后整段 legacy SVG 渲染体（节点/IO/边/legend/summary 中走 `svg.appendChild` / `RENDER_GROUP_EXPORTS` / `createElementNS` 的部分）；legend/summary/mode-badge/meta-info 等 **DOM 浮层更新逻辑迁入 `__canvasRenderPhase1` 收尾阶段**（这些是 DOM 不是 SVG，保留行为）。
2. 删除 `renderIOGroupPill` / `renderIOPill` / `renderIOPill` 的 SVG 版（职责迁 `IOLayer`）；保留 `collectIORenderTasks` / `computeIOGroupExpandedLayout` / `calcIOGroupExpandedHeight`（被 Canvas 复用）。
3. `btn-fit` 改为调用 `fitToView`，删除 `#dag-svg` 引用。
4. 删除 `<style>` 中 `.dag-svg ...` 死样式；保留 `.dag-stage`。
5. 删除残留 `svg` 局部变量、`edgeOverlayLayer` 的 SVG appendChild、`RENDER_GROUP_EXPORTS` 常量。

### 测试侧（`testset/unit/`）
6. `test_frontend_html.py` 中 B/C 类（27 个）**保持 `@pytest.mark.skip`，不删**（按既定约定）；其断言的 DOM 已随实现删除，skip reason 仍准确。
7. 新增 IO/进度/视口/裁剪用例统一进 `test_frontend_canvas_phase1.py`，helper 进 `frontend_test_infra.py`，禁止重复定义。

---

## 8. 验收门槛（Phase 1 收口）

- P1-17 ~ P1-23 + P1-E1 PASS；Stage 1.1~1.3 全部 P1-* 仍 PASS（回归不破）。
- `frontend_html.py` grep 无 `dag-svg` / `createElementNS` / `<marker` / `RENDER_GROUP_EXPORTS` / `buildIntraGroupEdgePath` 残留。
- `btn-fit` 运行期可用，不抛错。
- culling on/off 模型集合相等。
- 只跑前端 UT：`python3 -m pytest testset/unit/test_frontend_canvas_phase1.py testset/unit/test_frontend_html.py -v`，新增/改写全 PASS，skip 项 reason 明确。
- Phase 1 结束态：运行时零 SVG 渲染路径，DOM 浮层（tooltip/side-panel/legend/summary/progress）行为不变，「可看不可动」达成。
