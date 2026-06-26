# Frontend Canvas Keyboard / Scale Bar / Help Popup 设计

## 功能概述

本次 Canvas 前端增强包含三部分：

1. 键盘导航：支持 `W/A/S/D` 与方向键平移视图，`Q/E` 以视口中心为锚点缩放。
2. Scale Bar：在 Canvas 左下角显示当前缩放比例，并随 viewport scale 实时更新。
3. 操作说明浮层：在 topbar legend 后新增 `⌨ Help` 按钮，点击展示快捷键说明。

## 改动文件和行号

- `scripts/render_canvas.js`
  - `applyViewport()`：约 611 行，写回 Pixi world transform 后触发 `global.__onViewportUpdate(engine.viewport.scale)`。
  - `global.__canvas*` 接口区：约 3164 行，新增 `global.__canvasPan(dx, dy)` 与 `global.__canvasZoom(factor, minScale, maxScale)`。
- `scripts/frontend_html.py`
  - CSS：约 224 行，在 `</style>` 前新增 `.help-popup*` 与 `.scale-bar*` 样式。
  - topbar DOM：约 308 行，在 legend 结束后的 `</span>` 后新增 help button + help-popup DOM，未改动已有 legend 内容。
  - Scale Bar DOM：约 322 行，放置在 `<div class="dag-container"` 之前。
  - keydown / UI wiring：约 2288 行，扩展键盘 handler，并追加 help-popup toggle 与 `updateScaleBar/window.__onViewportUpdate` 逻辑。
- `storage/testset/unit/test_frontend_canvas_phase2.py`
  - 文件末尾新增 `__canvasPan` 与 `__canvasZoom` 单测，复用 `run_canvas_render_probe` 与既有 fixture。

## WASD/QE 键盘逻辑说明

- `Escape` 保留既有优先级：先退出 Semantic Zoom；若没有 focus stack，再关闭 side-panel。
- 当事件目标是 `INPUT` / `TEXTAREA` / `SELECT` 时直接忽略，避免影响输入控件。
- `W` / `ArrowUp`：调用 `window.__canvasPan(0, panStep)`。
- `S` / `ArrowDown`：调用 `window.__canvasPan(0, -panStep)`。
- `A` / `ArrowLeft`：调用 `window.__canvasPan(panStep, 0)`。
- `D` / `ArrowRight`：调用 `window.__canvasPan(-panStep, 0)`。
- `Q`：调用 `window.__canvasZoom(1 / 1.12, 0.05, 8)` 缩小。
- `E`：调用 `window.__canvasZoom(1.12, 0.05, 8)` 放大。
- 快捷键命中后统一 `preventDefault()`，避免页面滚动等浏览器默认行为。

## Scale Bar 更新机制

更新链路为：

```text
applyViewport → __onViewportUpdate → updateScaleBar
```

- `render_canvas.js` 中所有视口变更最终进入 `applyViewport()`。
- `applyViewport()` 在同步 `engine.world.scale/x/y` 后，若 `global.__onViewportUpdate` 是函数，则传入当前 `engine.viewport.scale`。
- inline runtime 中将 `window.__onViewportUpdate = updateScaleBar`。
- `updateScaleBar(scale)` 更新 `#scale-bar-value` 的百分比文本，并根据 scale 调整 `.scale-bar-track` 宽度。

## 操作说明浮层 DOM 结构和 CSS

DOM 位于 topbar 右侧 legend 后：

```html
<span class="help-popup-wrap">
  <button id="btn-help" class="tb-btn" type="button" aria-expanded="false" aria-controls="help-popup">⌨ Help</button>
  <div id="help-popup" class="help-popup" role="dialog" aria-label="Canvas 操作说明">
    <div class="help-popup-title">Canvas 操作说明</div>
    ...快捷键说明...
  </div>
</span>
```

CSS 要点：

- `.help-popup` 使用 `background: #16213e` 与 `border: 1px solid rgba(255,255,255,0.12)`。
- `.help-popup.open` 控制显示。
- `.scale-bar` 同样使用 `#16213e` 背景与 `rgba(255,255,255,0.12)` 边框，定位在 Canvas 左下角。

## UT 覆盖点

- `__canvasPan(dx, dy)`：初始化并渲染后执行 pan，断言 viewport `x/y` 按 dx/dy 改变，scale 不变。
- `__canvasZoom(factor, minScale, maxScale)`：断言缩放比例按 factor 更新并受 min/max clamp；同时验证以视口中心为锚点时 `x/y` 按公式变化。
- `__onViewportUpdate`：在 UT 中挂载回调，验证 pan/zoom 触发 applyViewport 后上报最新 scale。
