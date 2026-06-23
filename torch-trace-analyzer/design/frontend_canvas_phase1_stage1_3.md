# Stage 1.3 — 连边批绘（EdgeRoute + EdgeBatch → L1）详细设计 + UT 详细设计

> 状态：**设计待 NaN review，不含任何实现代码**。review 通过后才实现。
> 上位文档：`design/frontend_canvas_phase1.md`、`design/frontend_canvas_phase1_stages.md`、`design/frontend_canvas_renderer.md`
> 前置：Stage 1.1 / 1.2 已合入（节点/容器/端口已在 Canvas L2/L3 绘制，`nodePortMap` 与 snapshot 端口逐值相等）。

---

## 1. 设计目标（本阶段范围）

把「连边」从 SVG `<path>` 迁到 Pixi L1 的 `EdgeBatch`，同时删除 SVG 边相关死代码。

- 新增 `EdgeRoute`（纯几何，把端点 → 折线点序列）+ `EdgeBatch`（按 type 分桶批绘 + 三角箭头）到 `render_canvas.js`
- `__canvasRenderPhase1` 增加连边阶段：复用现有 `collectGlobalEdgeTasks` 的同一套**可见性 / redirect / 端口解析 / self-skip / 缺端点抛错**逻辑
- `__renderSnapshot().edges` 从空数组变为真实边记录
- 删除 `frontend_html.py` 中 SVG 边死代码：`buildDirectEdgePath` / `buildIntraGroupEdgePath` / `createEdgePathElement` / `createEdgeHitboxElement` / `applyEdgePresentation` 的 marker-end 分支 / `<marker>` defs / `.edge-path` 死样式
- 边交互（hover 高亮 / evidence 侧栏）**不做**，留到 Phase 3（保持 Phase 1「可看不可动」）

布局算法、`DATA.edges`、后端 JSON、`EDGE_BUNDLE_META`、`isEdgeVisible`、`resolveCollapsedAncestor` 一行不动。

---

## 2. 现状基线（源码证据）

| 事实 | 位置 |
| --- | --- |
| 生产连边模型：**所有边都走 `collectGlobalEdgeTasks` → `routingMode:'direct'`** | `frontend_html.py:1915-1924, 2186-2219` |
| 端点解析：`resolveCollapsedAncestor` 折叠重定向 + `nodePortMap[id+'__out']||[id]` / `[id+'__in']||[id]` | `frontend_html.py:2191-2197` |
| self-skip（重定向后同 id）/ 缺端点抛错 | `frontend_html.py:2193-2200` |
| 可见性过滤 `isEdgeVisible(edge)` | `frontend_html.py:1918` |
| bundle 偏移 `EDGE_BUNDLE_META.get(edgeKey(edge))` | `frontend_html.py:2207`、`:360-361` |
| 直连几何 3 分支（dy>8 下行 cubic / dy<-8 回环 cubic / 否则 quadratic） | `frontend_html.py:1598-1611` |
| 长边虚线 `configureLongEdgeDisplay` | `frontend_html.py:1565-1586`（`.long-edge`） |
| 边呈现：class `edge-path {type}` + marker-end（dep/internal/else 三种箭头） | `frontend_html.py:1718-1727` |
| **`buildIntraGroupEdgePath` 定义但当前编排从未调用（死代码）** | 定义 `:1615-1657`；编排无 `routingMode:'intra_group'` 推送 |
| snapshot.edges 当前恒为空 | `render_canvas.js:402-407` 区域 |

> 重要结论：当前生产路径**没有** intra-group 边渲染，`buildIntraGroupEdgePath` 是历史遗留死代码。Stage 1.3 只需复刻 `buildDirectEdgePath` 几何 + 全局边编排逻辑，并把 intra-group 死代码一并删除。

---

## 3. 旧功能替换映射

| 旧实现（SVG） | 新实现（Canvas） | 等价定义 |
| --- | --- | --- |
| `buildDirectEdgePath(x1,y1,x2,y2,routeMeta)` → SVG path string | `EdgeRoute.direct(x1,y1,x2,y2,routeMeta)` → 采样点数组 `[{x,y}...]` | 控制点公式逐字搬运；点序列对贝塞尔做固定步数采样，端点严格相等 |
| `<path marker-end="url(#arrowhead-*)">` | `EdgeBatch` 在终点按切线方向画三角箭头 | 箭头朝向 = 终点切线；按 type 取色（dep/internal/else 三色保持） |
| `edge-path {type}` class 控制描边色/宽 | `EdgeBatch` 按 type 分桶设 `lineStyle(color,width,alpha)` | 三类颜色/线宽/透明度参数从 CSS 平移到 JS 常量表 |
| `createEdgeHitboxElement`（透明粗描边 hitbox） | **删除**（Phase 1 无边交互） | Phase 3 用空间索引命中，不再造 DOM hitbox |
| `applyEdgePresentation` marker-end 分支 | **删除** | — |
| `<defs><marker>` 箭头定义 | **删除** | 箭头改由 EdgeBatch 几何绘制 |
| `buildIntraGroupEdgePath` + `routingMode:'intra_group'` 分支 | **删除（死代码）** | 当前生产从未调用，删除不改变任何可见行为 |
| `configureLongEdgeDisplay`（长边虚线） | `EdgeBatch` 按 `routeMeta`/长度阈值用 dashed 描边 | 长边阈值与短划参数保持；视觉等价（见 §4 等价点 4） |

---

## 4. 功能等价性保证（怎么证明不是「看着差不多」）

等价拆成可断言的 5 点，全部用 headless snapshot probe 验证：

1. **边集合等价**：`snapshot.edges` 的 `(from,to)` 集合 == 旧路径下 `collectGlobalEdgeTasks` 过滤后逐边重定向、去 self、去不可见后的结果。
   - 同一 `DATA`，对比「Stage 1.3 snapshot.edges」与「直接调用复用后的 `collectGlobalEdgeTasks`+redirect 纯函数」结果，集合相等。
2. **端点几何等价**：每条边的起点 == `nodePortMap[fromId+'__out']||[fromId]` 的 `(cx,cy)`，终点 == `nodePortMap[toId+'__in']||[toId]`。snapshot 记录 `start/end`，与 `nodePortMap` 逐值相等。
3. **路由几何等价**：把 `EdgeRoute.direct` 与 legacy `buildDirectEdgePath` 同时导出为可测纯函数，对同一组 `(x1,y1,x2,y2,routeMeta)` 随机/边界样本，断言：
   - 分支选择一致（dy>8 / dy<-8 / 否则）；
   - 端点严格相等；
   - 采样曲线与 legacy 贝塞尔在相同参数 t 上的点距 < 0.5px（数值等价，不是肉眼）。
4. **长边样式等价**：超过长边阈值的边，snapshot 标 `dashed:true`；阈值与 legacy `configureLongEdgeDisplay` 同源常量。
5. **类型/颜色/箭头等价**：snapshot 每条边记 `type` 与 `colorKey`；dep/internal/else 三类映射到与 CSS 同值的颜色常量；箭头存在且方向 = 终点切线。
6. **错误语义等价（强化）**：重定向后端点缺失仍 `throw 'global edge endpoint missing: a -> b'`；未知 routingMode/taskKind 仍抛错；不得静默 return。

> 为支撑等价点 3，实现时把几何公式抽成 `EdgeRoute.direct` 单一函数，legacy 的 `buildDirectEdgePath` 在删除前先用它产出 path string，做一轮「新几何函数 → 旧 path」一致性回归，再删 SVG 包装。

---

## 5. `render_canvas.js` 新增结构（设计，无实现代码）

- `const EDGE_STYLE = { dep:{color,width,alpha}, internal:{...}, default:{...} }`：与 CSS `.edge-path.*` 同值。
- `EdgeRoute.direct(x1,y1,x2,y2,routeMeta) -> {points:[{x,y}...], branch:'down'|'back'|'quad', dashed:bool}`：复刻 §2 三分支公式，按固定步数采样贝塞尔。
- `EdgeBatch`：
  - `collect(edge, start, end, routeMeta)`：算路由、按 type 入桶、push `engine.edges`（snapshot 记录 `{from,to,type,start,end,branch,dashed}`）。
  - `flush()`：每个 type 桶用一个/少量 `PIXI.Graphics` 批量描边到 L1，再画箭头。
- `drawGlobalEdges(DATA)`：编排，逐边复用 `resolveCollapsedAncestor` + `nodePortMap` + `isEdgeVisible` + `EDGE_BUNDLE_META`（这些保留在内联 JS，不迁移），self-skip，缺端点 throw，调 `EdgeBatch.collect`，最后 `flush`。
- snapshot：`edges` 返回上述记录数组；`layers.l1` > 0。

---

## 6. UT 详细设计

测试文件：`testset/unit/test_frontend_canvas_phase1.py`（续）
基础设施：`frontend_test_infra.py` 复用 `run_canvas_snapshot_probe` / `run_canvas_render_probe`；新增（若需要）`edge_fixture()` 构造含跨容器边的 flowchart。

| ID | 测例 | 输入 fixture | 断言（等价点） |
| --- | --- | --- | --- |
| P1-6 | `test_snapshot_edges_match_visible_global_edges` | 2 root + 跨容器边若干，含 1 条不可见、1 条 self（重定向后同 id） | `snapshot.edges` 的 `(from,to)` 集合 == 期望可见集；不可见/ self 被剔除（等价点 1） |
| P1-8 | `test_edge_endpoints_match_node_port_map` | 含折叠容器目标的边 | 每条 edge `start`/`end` == `nodePortMap[...__out/__in]||[...]` 的 `(cx,cy)`（等价点 2） |
| P1-11 | `test_edge_route_direct_geometry_equivalent_to_legacy` | 参数样本：dy>8 / dy<-8 / |dy|≤8 三组 + 各种 offset/dx 符号 | `EdgeRoute.direct` 分支选择与端点匹配 legacy；采样点 vs legacy 贝塞尔点距 < 0.5px（等价点 3） |
| P1-11b | `test_edge_route_degenerate_returns_null` | `|dy|<3 && |dx|<3` | 返回 null / 不产边（与 legacy 一致） |
| P1-12 | `test_long_edge_marked_dashed` | 1 条超阈值长边 + 1 条短边 | 长边 `dashed:true`、短边 `dashed:false`，阈值同源常量（等价点 4） |
| P1-13 | `test_edge_type_color_and_arrow` | dep / internal / 其它 三类边 | 每条 edge `type` 与 `colorKey` 正确，箭头存在；颜色常量 == CSS 同值（等价点 5） |
| P1-16 | `test_snapshot_layer_l1_populated_after_edges` | 含边 fixture | `layers.l1 > 0`；L2/L3 仍 > 0；L4 == 0（边不触发交互层） |
| P1-E2 | `test_global_edge_missing_endpoint_raises` | 人为构造重定向后端口缺失的边 | 抛 `RuntimeError`/`Error("global edge endpoint missing: ...")`（等价点 6） |
| P1-E3 | `test_unknown_edge_routing_mode_raises` | 直接调 `EdgeRoute`/`renderEdge` 传非法 mode | 抛错，不静默（等价点 6） |

覆盖率门槛（仅本阶段新增代码）：line ≥ 90%，branch ≥ 50%（三条几何分支 + degenerate + dashed + 三类 type 全覆盖）。

---

## 7. 旧 UT / 旧实现清理动作

### 实现侧（`frontend_html.py`）
1. 删除 `buildIntraGroupEdgePath`、`createEdgePathElement`、`createEdgeHitboxElement`、`finalizeEdgeRendering` 中 SVG 边逻辑、`applyEdgePresentation`、`configureLongEdgeDisplay` 的 SVG 版本。
2. 删除 `<defs>` 内 `<marker id="arrowhead*">` 定义与 `.edge-path` / `.long-edge` 死样式。
3. `renderEdge` 的 SVG 版整体删除（其编排职责由 `render_canvas.js:drawGlobalEdges` 承接）；保留 `collectGlobalEdgeTasks`/`isEdgeVisible`/`resolveCollapsedAncestor`/`EDGE_BUNDLE_META`/`edgeKey`（被 Canvas 复用）。
4. `buildDirectEdgePath` 在等价回归（§4 注）通过后删除。

### 测试侧（`testset/unit/`）
5. D 类 `test_html_emits_edge_index_globals` 等「断言 SVG path/marker DOM」用例改写为 snapshot 断言或并入 P1-6/P1-13；纯 SVG path 断言用例打 `@pytest.mark.skip(reason="canvas phase1: svg-specific assertions removed")`（不删）。
6. `frontend/test_edge_routing.js`：若其断言的是 `buildDirectEdgePath` path string，则改为断言 `EdgeRoute.direct` 点序列；若断言的是已删的 intra-group 路由，整文件 skip。
7. 不新增重复 helper，边 fixture 统一进 `frontend_test_infra.py`。

---

## 8. 验收门槛

- P1-6 / P1-8 / P1-11 / P1-12 / P1-13 / P1-16 / P1-E2 / P1-E3 PASS。
- `frontend_html.py` 内无 `<marker>` / `.edge-path` / `buildIntraGroupEdgePath` / `createEdgeHitboxElement` 残留（grep 验证）。
- snapshot.edges 与复用编排逻辑的边集合相等；几何点距 < 0.5px。
- 只跑前端 UT：`python3 -m pytest testset/unit/test_frontend_canvas_phase1.py testset/unit/test_frontend_html.py -v`，新增/改写用例全 PASS，skip 项 reason 明确。
