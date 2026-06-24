# Stage 1.1 — 模板切换 + Canvas 引擎基建（AS-BUILT 回顾 + 等价性说明）

> 状态：**已实现并合入 `feat/canvas-renderer`**
> 研发分支 commit：`d683eb4 feat: canvas phase1 stage1.1 — switch flowchart template to Canvas engine skeleton`
> 测试分支 commit：`8308bd0 test: canvas phase1 stage1.1 — add Stage 1.1 UTs and group-skip legacy B/C suites`
>
> 上位文档：`design/frontend_canvas_phase1.md`、`design/frontend_canvas_phase1_stages.md`
>
> 本文是 Stage 1.1 的 **as-built** 复盘：记录实际改了什么、旧功能怎么替换、用哪些已通过 UT 证明等价。后续 Stage 1.3/1.4 在此基线上继续。

---

## 1. 设计目标（本阶段范围）

只做「渲染容器 + 脚本注入链路」的切换，不画任何真实图元：

- `<svg id="dag-svg">` → `<div id="dag-stage">`（Pixi 挂载点）
- 注入链路从 `render_group.js` 改为 `pixi_engine_bundle.js` + `render_canvas.js`
- `render_canvas.js` 建立 `PIXI.Application + world(viewport) + L0~L5` 六层容器
- 暴露 `window.__phase1NoInteractionMode = true` 与只读骨架版 `window.__renderSnapshot()`
- 占位符 / 容器缺失一律 `raise`，不做任何 fallback

布局算法、后端 JSON、DOM 浮层一行不动。

---

## 2. 现状基线（源码证据）

| 事实 | 位置 |
| --- | --- |
| 容器已是 `<div id="dag-stage" class="dag-stage">` | `scripts/frontend_html.py:243` |
| 两个新占位符 `__ENGINE_BUNDLE_PLACEHOLDER__` / `__RENDER_CANVAS_JS_PLACEHOLDER__` | `scripts/frontend_html.py:269,272`（常量 `:42,:44`） |
| `render()` 在 phase1 模式下直接委派给 `window.__canvasRenderPhase1(DATA)`，flag 开但入口缺失即抛错 | `scripts/frontend_html.py:1293-1298` |
| 引擎工厂 + L0~L5 层 + headless mock + 只读 snapshot | `scripts/render_canvas.js:42-152, 384-422` |
| `__phase1NoInteractionMode = true` 全局开关 | `scripts/render_canvas.js:415` |
| `#dag-stage` 容器缺失抛 `Error`（无静默 fallback） | `scripts/render_canvas.js:146-149` |

> 注：`render()` 之后（1299 行起）仍保留整段 legacy SVG 渲染体，但在 phase1 模式下**永不执行**（1293 行短路返回）。这些死代码由 Stage 1.3 / 1.4 分批删除。

---

## 3. 旧功能替换映射

| 旧实现 | 新实现 | 等价/约束 |
| --- | --- | --- |
| `<svg class="dag-svg" id="dag-svg">` 容器 | `<div id="dag-stage">` | 必须唯一存在；SVG 容器字面量从 HTML 消失 |
| `__RENDER_GROUP_JS_PLACEHOLDER__` 注入 `render_group.js` | `__ENGINE_BUNDLE_PLACEHOLDER__`（Pixi）+ `__RENDER_CANVAS_JS_PLACEHOLDER__`（render_canvas.js） | 旧占位符在 HTML 中不再以字面量出现；不存在「Canvas 失败回退 SVG」兼容层 |
| `.dag-svg .*` SVG 样式 | 暂保留在 `<style>`（死样式），Stage 1.4 统一删 | DOM 浮层样式（tooltip/side-panel/progress overlay/legend/summary）保留不变 |
| 无 Canvas 根对象 | `PIXI.Application` + `world` + L0~L5 | 本阶段各 layer 只建空容器，不放图元 |
| 渲染入口 `render()` 直接构建 SVG | `render()` 委派 `__canvasRenderPhase1(DATA)` | DATA / SOURCE_MAP 注入逻辑完全复用 |

---

## 4. 功能等价性保证（本阶段验证什么）

Stage 1.1 不画图元，等价性聚焦在「壳层等价 + 错误硬失败」：

1. **数据等价**：`const DATA = ...` 注入逻辑不变（占位符替换框架未动）。
2. **布局等价**：布局函数（`computeRanks/orderRanks/layoutGroup/indexGroupAncestors`）源码 diff = 0。
3. **DOM 浮层等价**：`tooltip / side-panel / summary / legend / render-progress-overlay` 节点仍在模板里。
4. **行为约束等价**：运行期 `window.__phase1NoInteractionMode === true`，L4 交互层为空（无 hover/click 接线）。
5. **错误语义等价（强化）**：占位符缺失、`#dag-stage` 缺失都必须 `raise`；这是对旧版「占位符缺失即报错」语义的保留与强化，而非弱化。

---

## 5. UT 详细设计（已落地，全部 PASS）

测试文件：`testset/unit/test_frontend_canvas_phase1.py`
基础设施：`testset/unit/frontend_test_infra.py`（`render_minimal_flowchart_to_string` / `run_canvas_snapshot_probe` / `_canvas_dom_stub_js` / `RENDER_CANVAS_JS_PATH` / `ENGINE_BUNDLE_JS_PATH`）

| ID | 测例 | 输入 | 断言（等价点） |
| --- | --- | --- | --- |
| P1-1 | `test_canvas_stage_div_replaces_svg` | 最小 flowchart | HTML 无 `id="dag-svg"` / `<svg class="dag-svg"`；含 `<div id="dag-stage"` |
| P1-2 | `test_engine_and_canvas_js_placeholders_replaced` | 最小 flowchart | 两个新占位符字面量已被消费；无 `__RENDER_GROUP_JS_PLACEHOLDER__`；含 `PIXI` / `render_canvas.js` / `<script id="engine-bundle">` / `<script id="render-canvas-js">` |
| P1-3 | `test_engine_placeholder_missing_raises` | 人工删掉 engine 占位符的模板 | `_generate_flowchart_html()` 抛 `RuntimeError("engine bundle placeholder is missing")` |
| P1-3b | `test_render_canvas_placeholder_missing_raises` | 人工删掉 canvas 占位符 | 抛 `RuntimeError("render_canvas.js placeholder is missing")` |
| P1-10 | `test_phase1_no_hover_listeners_attached` | snapshot probe | `flags.noInteractionMode === true`；`layers.l4 === 0` |
| P1-10b | `test_phase1_snapshot_skeleton_shape` | snapshot probe | snapshot 是完整空骨架：nodes/edges/io_pills=[]，ports={}，6 个 layer 全 0，`cullingEnabled=false`，viewport 含 `scale/x/y/worldWidth/worldHeight` |
| P1-10c | `test_phase1_snapshot_matches_with_engine_bundle_loaded` | headless vs 加载 Pixi bundle | 两条路径 `layers` / `flags` 结构一致（证明 headless mock 与浏览器路径同构） |
| P1-10d | `test_phase1_missing_container_raises` | `getElementById` 返回 null | `__renderSnapshot()` 抛错，不静默 no-op |

---

## 6. 旧 UT / 旧实现清理动作（已执行）

- `test_frontend_html.py` 中 B 类（16，SVG/CSS/DOM 强绑定）+ C 类（14，hover/click/active 行为）统一加 `@pytest.mark.skip`，reason 二选一：
  - `canvas phase1: svg-specific assertions removed`
  - `canvas phase1: interaction deferred to phase3`
- A 类（10）+ redirect 测试不动语义，仅探针迁移（Stage 1.2 收尾）。
- `test_edge_routing.js` 不动。
- 运行时不再注入/执行 `render_group.js`。

---

## 7. 验收门槛（已满足）

- 仅模板 / 注入 / Canvas 骨架相关改动；布局函数 diff = 0。
- P1-1 / P1-2 / P1-3 / P1-10 PASS。
- 占位符 / 容器缺失明确报错。
