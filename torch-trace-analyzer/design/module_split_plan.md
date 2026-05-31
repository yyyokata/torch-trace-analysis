# analyze_trace.py 模块拆分方案
> 建立时间：2026-05-31  
> 当前 analyze_trace.py 约 9311 行，目标：<900 行（只保留 CLI + orchestration）

---

## 一、模块职责划分

### 图生成侧（DAG 构图）

| 模块 | 职责 | 状态 |
|---|---|---|
| `ast_types.py` | `CallLoc` 等基础数据结构 | 已独立 |
| `ast_constants.py` | 常量求值基础设施（`ConstantTable`） | 已独立 |
| `ast_resolver.py` | AST 节点求值（`ConstantResolver`） | 已独立 |
| `attr_types.py` | Attr 四类一等公民定义（`ModuleAttr` / `InputAttr` / `ResultAttr` / `ContainerAttr` + `ForwardArgAttr` / `ReturnValAttr`） | 已独立，清理中 |
| `attr_scanner.py` | 扫描源码构建 Attr 实例，分配 `attr_id` | 已独立 |
| `ast_frontend.py` | `ASTFrontend`，AST 解析/缓存，`build_dag_context()` 入口 | 已独立，待加 `build_dag_context` |
| `dag_types.py` | DAG IR 数据结构（`DagContext` / `DAG` / `DagNode` / `DataFlowEdge` / `VarEvidence`） | **待新建**，设计见 `DESIGN_DAG_TYPES.md` |
| `source_index.py` | 源文件扫描和 class map 构建（`_build_ast_frontends` / `build_module_like_set` / `_build_class_map` 等） | 拆出 Phase A |
| `legacy_static_tree.py` | 旧 dict 型 `build_static_module_tree`（存续过渡期） | 拆出 Phase C，新 DAG IR 落地后删 |
| `legacy_dataflow_edges.py` | 旧 `_build_data_dependency_edges`（存续过渡期） | 拆出 Phase C，新 DAG IR 落地后删 |
| `legacy_ast_module_tree.py` | 旧 `_AST_ModuleTreeExtractor`（存续过渡期） | 拆出 Phase C，新 DAG IR 落地后删 |

### Timeline 分析侧（运行时）

| 模块 | 职责 | 状态 |
|---|---|---|
| `trace_io.py` | `load_trace` / `load_model_code`，纯 IO | 拆出 Phase A |
| `common_utils.py` | `format_duration` / `pct_str` / `_strip_inline_comment` / `_join_logical_lines`，无依赖工具函数 | 拆出 Phase A |
| `trace_analysis.py` | trace 事件解析：`extract_metadata` / `detect_trace_type` / `build_parent_index` / `classify_threads` / `analyze_*` 系列，只依赖 trace JSON | 拆出 Phase A |
| `source_hotspot.py` | 源码热点分析：`analyze_source_hotspots` / `enrich_kernel_modules_with_source` / `analyze_kernel_call_stacks`，依赖 trace + source_index | 拆出 Phase A |
| `timing_attribution.py` | Timing 归因核心：`build_loc_index` / `match_kernel_to_instance` / `build_fwdbwd_flow_index` / `rollup_instance_timing` / `build_timing_panel_data` + `StackFrame` / `InstanceKey` / `KernelChain` 等数据结构，~1700 行 | 拆出 Phase B |

### 展示/输出侧

| 模块 | 职责 | 状态 |
|---|---|---|
| `frontend_html.py` | HTML 渲染，DAG 前端展示，evidence 面板，TIMING 面板 | 已独立 |
| `report_summary.py` | 所有 `print_*` / `save_*_md` / `save_markdown_report`，纯文本报告，~650 行 | 拆出 Phase A |

### 入口/编排

| 模块 | 职责 | 状态 |
|---|---|---|
| `analyze_trace.py` | CLI parse / `main()` / pipeline 编排，不含任何业务实现 | 保留壳，目标 <900 行 |
| `fake_runner.py` | Mock runtime 入口 | 已独立 |
| `wrapper_swimlane.py` | 泳道图生成 | 已独立 |

---

## 二、迁移后依赖拓扑

```
                    ┌─────────────┐
                    │analyze_trace│  ← CLI + orchestration only
                    └──────┬──────┘
           ┌───────────────┼────────────────────┐
           ▼               ▼                    ▼
  [图生成侧]          [Timeline 侧]         [展示侧]
  source_index        trace_io             frontend_html
       ↓               trace_analysis       report_summary
  ast_frontend         source_hotspot
       ↓               timing_attribution
  attr_scanner              ↑
       ↓               [共同依赖]
  attr_types          common_utils
  ast_resolver        trace_io
  ast_constants
  ast_types
       ↓
  dag_types（新）
  legacy_*.py（过渡）
```

**关键约束**：

1. **图生成侧和 Timeline 侧不互相依赖**，只在 `analyze_trace.py` orchestration 层汇合
2. `timing_attribution.py` 当前依赖 `source_index`（loc → attr 对应关系）；新 DAG IR 落地后改为依赖 `DagContext.loc_attr_index`，切断与 `source_index` 的依赖
3. `frontend_html.py` 只依赖图生成侧产物（`DagContext`）+ timing 侧产物（timing panel data），不反向依赖 `analyze_trace`
4. `legacy_*.py` 三个过渡模块只被 `analyze_trace.py` 直接调用，其他新模块不得新引入对 legacy 的依赖

---

## 三、总体行数收益预估

| 类别 | 行范围 | 估算行数 | 处理方式 |
|---|---|---:|---|
| 静态 DAG 构图大块 | 2716-4903 + 6598-8836 | ~4400 行 | 拆出 → 新 DAG IR 落地后删 |
| Timing attribution | 696-2426 | ~1700 行 | Phase B 拆到 `timing_attribution.py` |
| Trace 统计分析 | 4916-5890 | ~950 行 | Phase A 拆到 `trace_analysis.py` |
| Markdown/console 报告 | 5941-6590 | ~650 行 | Phase A 拆到 `report_summary.py` |
| CLI / IO / glue | 1-364 + 8951-9311 | ~700 行 | 保留在 `analyze_trace.py` |

**压缩路径**：
- Phase A + B 纯搬迁完成后：`analyze_trace.py` 约 **2500 行**
- Phase C legacy 搬出后：约 **700 行**（壳）
- Phase D 新 DAG IR 替换 + 删 legacy 模块：最终 **<900 行**（含 orchestration 实现）

---

## 四、立即可删（不依赖任何任务进度）

| 内容 | 原因 |
|---|---|
| `USE_NEW_EVAL` / `_AB_EVAL_DIFFS` / `--use-new-eval` / `--ab-eval-report` / `_emit_ab_summary_if_enabled()` | `ConstantResolver` 是唯一生产路径，legacy AB 实验已结束 |
| `_ast_frontend_smoke_test()` | 硬编码 testset 路径的临时函数，应迁入 `testset/unit/` |

---

## 五、新旧 timing 归因关键路径对比

**现在（旧 dict 索引）**：
```python
build_loc_index(static_module_tree)   # key=(basename, line)
    → match_kernel_to_instance(chain, loc_index)
```

**新 DAG IR 落地后**：
```python
DagContext.loc_attr_index[(file, line, attr_id)]   # 自动派生，O(1)
    → node_id → DagNode（直接回写 timing 字段）
```

`timing_attribution.py` 内部实现切换，`analyze_trace.py` orchestration 调用签名不变。

---

## 六、执行顺序

```
Phase A（低风险，纯搬迁）
    trace_io / common_utils / source_index / trace_analysis / source_hotspot / report_summary
        ↓
Phase B（中风险，timing 整块搬迁）
    timing_attribution.py（~1700 行整块）
        ↓
Phase C（标 legacy，搬出但不删）
    legacy_static_tree / legacy_dataflow_edges / legacy_ast_module_tree
        ↓
Phase D（新 DAG IR 替换 → 删 legacy 三个模块）
    dag_types.py 新建 + build_dag_context() 落地 + timing_attribution 切换新索引
```

**当前状态**：等 `attr_types_cleanup_icAFBmfj` 落地后，从 Phase A 开始。
