# Canvas 边样式设计（Phase 2 Step 6：长边截断 + IO 边圆点 + hover/select 屏蔽）

实现位置：`scripts/render_canvas.js`。本文档记录已与 NaN 确认并落地的边样式方案。

## 1. 长边截断（所有长边统一适用）

- **触发条件**：`polylineLength(points) >= LONG_EDGE_MIN_SPAN`（260px）。该判定结果以
  `dashed` 字段持久化在 edge snapshot 上（`dashed === true` 即长边）。
- **渲染规则**：只画 src 侧 `EDGE_TRUNCATE_HEAD`（40px）头段 + dst 侧
  `EDGE_TRUNCATE_TAIL`（40px）尾段，中间完全隐藏，无虚线、无半透明。
- **箭头**：仅在 dst 端（尾段末端）绘制，头段末端不画箭头。
- **弧长截取**：基于 `EdgeRoute.compute` 产出的真实 `sampleCubic` / `sampleQuad`
  采样点做弧长截取（`takeHeadByLength` / `takeTailByLength`），边界点在跨越截断
  阈值的那一小段上线性插值得到，不用直线近似。`260 > 40 + 40` 保证头尾段不重叠。

## 2. IO 边额外规则（在截断基础上叠加）

- **IO 边定义**：edge 的原始端点 `e.from` / `e.to` 命中以下任一即为 IO 边
  （`isIOEdge` + `ensureIOEdgeIndex`，按字符串 key 匹配）：
  `input_node_ids` / `param_node_ids` / `const_node_ids` / `output_node_ids`、
  io_group 的 `id`、io_group 的 `member_ids`。用原始端点而非 collapse/focus 重定向
  后的端点，保证 IO 归属稳定。该结果以 `isIO` 字段持久化在 edge snapshot 上。
- **端点圆点（Version B）**：在头段末端 + 尾段起点各画一个实心圆点，半径
  `IO_EDGE_DOT_RADIUS=4`，带 1px 白色描边（ring）；颜色与边同色。仅在被截断的长 IO 边
  上绘制（`drawIOEdgeDot`）。
- **交互限制（屏蔽 hover/select）**：IO 边 `interactive=false`，`path.eventMode='none'`，
  hover 不补全完整边、不触发高亮。可选中时机由点击对应 IO group（group panel）或进入
  Semantic Zoom focus 实现，不在边自身上开放。

## 3. 普通长边（非 IO）

- 截断逻辑同 §1。
- `interactive=true`，`path.eventMode='static'`；hover 时（`engine.hoveredEdgeKey === key`）
  补全显示完整 polyline（`bindEdgeHover` 监听 `pointerover` / `pointerout`，
  `setEdgeHover` 仅重绘受影响的两条边）。
- 无端点圆点。

## 落地范围与边界

- 已实现并有 UT：弧长截取几何、IO 分类、截断渲染、IO 圆点、hover 补全 / IO 屏蔽、
  `eventMode` / `interactive` 标志、`pointerover`/`pointerout` 接线。
- **边 panel（设计第四条）本次不做**，留到下一步。
- 细线 hit-area 精度（thin-stroke 是否真正命中 pointer）属浏览器侧验证项。

## 测试

UT：`storage/testset/unit/test_frontend_canvas_phase3.py`（18 例）。
- 几何精度：`run_edge_route_probe` 直接验证 `takeHeadByLength` / `takeTailByLength`
  在 down / back / quad 真实曲线上头尾段弧长 == 40px。
- 分类：`isIOEdge` 覆盖 input/param/const/output/io_group id/member/字符串 key/负例。
- 渲染：`run_canvas_snapshot_probe` 比对 `path.__drawOps`（moveTo/stroke/poly/fill/circle
  计数）验证截断、IO 圆点、full vs truncated、hover 补全、IO hover 屏蔽。
