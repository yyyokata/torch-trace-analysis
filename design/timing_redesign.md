# Timing 模块重构设计 — 4 步 Pipeline

> 目标：用 instance 级 kernel 归属彻底取代 class 级均摊，解决 GroupTower 多实例时间相同（DenseTower、tower_a/tower_b）等回退场景。
>
> 范围：仅替换 `analyze_trace.py` 的 timing 子系统；DAG 构建（`build_static_module_tree` 等）不动。

---

## 0. 现状诊断（要解决的核心问题）

### 0.1 旧链路简述

```
events → aggregate_runtime_instance_phase_times (wrapped event 主路径)
       → _aggregate_kernel_callsite_buckets       (synthetic 兜底，仅当 wrapped 缺失)
       → 合并为 runtime_instance_timings_by_class
```

### 0.2 已知 bug

- **GroupTower / DenseTower 多实例时间相同**：`_pick_instance_timing` 用 `[idx]` hint 匹配 `runtime_index`，但 stack_traces 路径丢失父层 callsite，多个实例共用同一 callsite_line，结果时间被均摊（详见 baseline_history → Bug B）。
- **wrapped 优先，synthetic 跳过**：`_aggregate_kernel_callsite_buckets` 在 wrapped event 已有该 class 时整体跳过，导致同 class 的「未 wrapped 实例」被忽略而非补齐。
- **backward 归属粗糙**：`_attribute_backward_kernel` 只看 stack 第二层，缺少 fwdbwd flow，正向 scope 信息没用上。
- **路径分裂**：wrapped event 路径和 synthetic 路径数据结构、归一化、phase 划分各写一份，难以维护。

---

## 1. 整体架构：4 步 Pipeline

```
                    ┌─────────────────────────────┐
                    │ Step 1: fwdbwd flow 解析    │
                    │  build_fwdbwd_flow_index    │
                    └──────────────┬──────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │ Step 2: kernel 归属到 instance                   │
        │  build_kernel_attribution_table                  │
        │   ├─ forward kernel: stack_traces 解析路径       │
        │   ├─ backward kernel: stack_traces + fwdbwd flow │
        │   │                   缩小到对应正向 scope 范围  │
        │   └─ wrapped event 兜底                          │
        └──────────────────────────┬───────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Step 3: bottom-up sum       │
                    │  rollup_instance_timing     │
                    │  （DAG 静态结构为骨架）     │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Step 4: 输出汇总            │
                    │  build_timing_panel_data    │
                    │   ├─ instance 级           │
                    │   └─ class 级（聚合）      │
                    └─────────────────────────────┘
```

---

## 2. 数据结构定义

```python
# 2.1 fwdbwd flow index（Step 1）
# Backward 时间区间 → 对应 Forward 时间区间
FwdbwdFlowEntry = {
    "fwd_start_us": float,
    "fwd_end_us": float,
    "fwd_tid": int,
    "bwd_start_us": float,
    "bwd_end_us": float,
    "bwd_tid": int,
    "flow_id": int,
}
# 索引：按 bwd 时间区间排序的列表 + 按 bwd_tid 分组的 dict
FwdbwdFlowIndex = {
    "by_bwd": [FwdbwdFlowEntry, ...],       # 按 bwd_start_us 排序
    "by_bwd_tid": {tid: [Entry, ...]},      # tid 分组后按 bwd_start_us 排序
}

# 2.2 instance key（Step 2 内部 + Step 3 主键）
# 唯一标识一个 nn.Module 实例的 callsite 链
InstanceKey = (
    class_name: str,
    callsite_file: str,        # 父 forward 调用此实例的源文件
    callsite_line: int,        # 父 forward 调用此实例的源码行
    ancestors: tuple[(file, line), ...],  # 从直接父到根的 (file,line) 链
)

# 2.3 kernel_attribution_table（Step 2 输出）
KernelAttribution = {
    kernel_idx: {
        InstanceKey: weight (float, 0..1)   # weight 之和 = 1.0
    }
}

# 2.4 instance_timing（Step 3 中间产物）
InstanceTiming = {
    InstanceKey: {
        "class_name": str,
        "callsite_file": str,
        "callsite_line": int,
        "ancestors": tuple,
        "self_us": {"forward": float, "backward": float, "optimize": float, "other": float},
        "inclusive_us": {"forward": float, "backward": float, "optimize": float, "other": float},
        "child_keys": set[InstanceKey],
    }
}

# 2.5 timing_data（Step 4 输出 — 对外接口，兼容 build_timing_data_from_trace 现有 caller）
timing_data["runtime_instance_timings_by_class"] = {
    class_name: [
        {
            "runtime_name": "ClassName@file:line",
            "runtime_index": int|None,
            "callsite": int,                  # 兼容字段
            "callsite_file": str,
            "callsite_line": int,
            "forward_us": float,              # exclusive (self)
            "backward_us": float,
            "optimize_us": float,
            "other_us": float,
            "total_us": float,                # = sum of 4 phases above (self)
            # 新增字段：inclusive 数据，给前端 Timing 面板用
            "self_us": float,
            "inclusive_us": float,
            "ancestors": [[file, line], ...], # 父链
        },
        ...
    ]
}
timing_data["class_durations"]      = {class_name: total_us}      # = sum of all instances inclusive
timing_data["class_durations_fwd"]  = {class_name: float}
timing_data["class_durations_bwd"]  = {class_name: float}
```

---

## 3. 各 Step 详细设计

### 3.1 Step 1: fwdbwd flow 解析

```python
def build_fwdbwd_flow_index(events: list[dict]) -> FwdbwdFlowIndex:
    """
    扫描 trace 中 cat=='fwdbwd' 的 flow 事件（ph='s'/'f'），按 id 配对：
      - ph='s'（source）位于 forward tid，ts 落在 forward op 内
      - ph='f'（finish/dest）位于 backward tid，ts 落在 backward op 内
    
    再从 events 里取 cat=='cpu_op' 且名字含 'CompiledFunction' / 含 'Backward' 的事件，
    用 ts 包含关系把 flow source/dest 锚定到具体 op，得到正向/反向 scope。

    返回：FwdbwdFlowIndex（见 §2.1）

    复杂度 O(N_events + N_flow log N_flow)。
    """
```

**用途**：Step 2 处理 backward kernel 时，先按 bwd_tid + kernel.ts 匹配到一个 fwdbwd entry → 得到对应正向 scope（fwd_start, fwd_end, fwd_tid）→ 在该正向 scope 时间窗口内做 instance 匹配，从而把候选实例缩小到「这次正向调用涉及的实例」而不是「class 全体实例」。

### 3.2 Step 2: kernel 归属到 instance

```python
def build_kernel_attribution_table(
    events: list[dict],
    source_files: dict,
    class_map: dict,
    step_infos: list,
    fwdbwd_index: FwdbwdFlowIndex,
) -> tuple[KernelAttribution, dict]:
    """
    对每个 kernel：
      1. 解析 stack_traces，得到 user 模块 callsite 链（内→外）。
      2. 从内到外遍历每条 frame，识别 class 边界，组装出一系列 candidate InstanceKey。
      3. 若是 backward kernel：
         a. 用 fwdbwd_index 找对应正向 scope。
         b. 在正向 scope 内（按 events 的 ts 范围）收集本次正向涉及的 InstanceKey 集合。
         c. 用 b 集合过滤候选 InstanceKey。
      4. 若候选 N==1 → weight=1.0；若 N>1 → 各 1/N 均摊。
      5. 若 stack_traces 缺失：退化到 wrapped event（_build_external_id_to_module_map）→
         构造一个 ancestors 留空的「弱 InstanceKey」，weight=1.0。

    返回：(KernelAttribution, debug_stats)
    
    debug_stats = {
        "total_kernels": int,
        "fwd_attributed": int, "fwd_unattributed": int,
        "bwd_attributed": int, "bwd_unattributed": int,
        "bwd_via_flow_narrowed": int,    # 通过 fwdbwd flow 把候选 N 缩小过的 backward kernel 数
        "wrapped_fallback": int,         # stack_traces 缺失走 wrapped 兜底的数
    }
    """
```

**实例 key 提取（关键算法）**：

```python
def _extract_instance_keys_from_stack(frames, class_map) -> list[InstanceKey]:
    """
    frames: [(fname, lineno, func_name), ...]  outer-first
    
    沿 outer→inner 行走，每当 class 切换：
      - parent_frame 是 class 切换点的「父侧最后一帧」（caller side）
      - child_class 是切换后的 class
      - 父链 ancestors 是再往外、所有先前 class-切换点的 (file, line) 列表
    
    每次 class 切换都产出一个 InstanceKey：(child_class, parent_file, parent_line, ancestors)
    
    若整条 stack 完整 → 最深的 InstanceKey 即是叶 instance；
    若 stack 被截断（frames < 2 或 class 不切换）→ 候选 InstanceKey 列表退化，
        最深一项的 ancestors 可能是空 tuple，多个真实实例共用此 key（N>1，均摊触发）。
    """
```

### 3.3 Step 3: bottom-up sum（DAG 静态结构为骨架）

```python
def rollup_instance_timing(
    kernel_attribution: KernelAttribution,
    events: list[dict],
    step_infos: list,
    static_tree: dict,        # build_static_module_tree() 输出
) -> InstanceTiming:
    """
    Phase A — self_us 累计：
      遍历 kernel_attribution，对每个 kernel：
        - dur = event.dur
        - phase = classify_kernel_phase(...)
        - 对每个 (instance_key, weight) → instance_timing[key].self_us[phase] += dur*weight
      最后所有 self_us[phase] 除以 num_steps。
    
    Phase B — parent-child 关系建立：
      利用 InstanceKey.ancestors 链 + static_tree 校验：
        - parent_key 由 ancestors[0] 直接得出 (parent_class 通过 file/line 在 class_map 找)
        - 若 ancestors[0] 在 static_tree 中确认存在 parent→child 类层级，建立 child→parent
        - 若 ancestors 空（stack 截断 → 弱 key）：在 static_tree 中查找该 class 的所有可能 parent，
          均摊（1/N）权重加到每个 parent。
    
    Phase C — bottom-up 累加：
      用拓扑排序（叶→根），对每个 instance：
        inclusive_us[phase] = self_us[phase] + sum(child.inclusive_us[phase] for child in children)
      返回 instance_timing dict。
    """
```

**为什么 DAG 静态结构是骨架**：
- 提供 class 层级关系（GroupTower → DenseTower），帮助校验 ancestors 链是否合理。
- 当 stack 截断、ancestors 缺失时，作为兜底参考。
- 不直接驱动 timing 累加（kernel 归属仍由 stack_traces 主导），只在父-子关系建立时介入。

### 3.4 Step 4: 输出格式

```python
def build_timing_panel_data(
    instance_timing: InstanceTiming,
    class_map: dict,
    step_dur_us: float,
) -> dict:
    """
    转换为现有 timing_data 接口（兼容 generate_html_flowchart 现有消费链路）：
      - runtime_instance_timings_by_class: instance 级，每个 class 下若干 instance items
      - class_durations / class_durations_fwd / class_durations_bwd: class 级聚合（=该 class 所有 instance 的 inclusive 之和）
    
    runtime_name 命名规则：
      - 完整 InstanceKey → "ClassName@file:line"
      - 弱 key（ancestors 空、callsite_line None） → "ClassName"

    runtime_index 兼容字段：
      - 若 InstanceKey 中存在 [idx] 提示（来自 ModuleList for-loop frame），保留 idx；否则 None。
    """
```

---

## 4. 与现有代码的衔接边界

| 现有代码 | 处理方式 | 替换位置 |
|---|---|---|
| `_build_callsite_runtime_map` | **删除** | Step 2 内部完成，无需独立函数 |
| `_attribute_backward_kernel` | **删除** | 与 Step 2 forward 路径合并 |
| `aggregate_source_module_phase_times` | **重写** | 改为基于 instance_timing 聚合（class 级 = sum of instances inclusive） |
| `aggregate_runtime_instance_phase_times` | **重写** | 改为调 Step 2+3 |
| `_aggregate_kernel_callsite_buckets` | **删除** | Step 2 已统一处理（synthetic 不再独立分支） |
| `build_static_module_tree` | **保留不动** | Step 3 的骨架来源 |
| `_build_external_id_to_module_map` | **保留** | Step 2 wrapped 兜底用 |
| `classify_kernel_phase` | **保留** | Step 2、Step 3 phase 分类 |
| `_normalize_runtime_module_name` | **保留** | Step 4 命名兼容 |
| `_find_class_for_line` | **保留** | Step 2 frame→class 映射 |
| `build_timing_data_from_trace` | **重写主体** | 调 Step 1→4，输出兼容字段 |
| `_pick_instance_timing` | **保留不动** | 现有 callsite hint 匹配逻辑可继续用，新数据带 callsite_line 字段 |

---

## 5. 函数签名清单

```python
# Step 1
def build_fwdbwd_flow_index(events: list[dict]) -> dict: ...

# Step 2
def build_kernel_attribution_table(
    events, source_files, class_map, step_infos, fwdbwd_index,
) -> tuple[dict, dict]: ...

def _extract_instance_keys_from_stack(frames, class_map) -> list: ...

def _narrow_backward_candidates_via_flow(
    bwd_kernel, fwdbwd_index, events_by_tid,
) -> tuple[float, float] | None: ...   # 返回正向 scope 时间窗 (fwd_start, fwd_end)

# Step 3
def rollup_instance_timing(
    kernel_attribution, events, step_infos, static_tree,
) -> dict: ...

# Step 4
def build_timing_panel_data(instance_timing, class_map, step_dur_us) -> dict: ...

# 顶层入口（替换原 build_timing_data_from_trace 中部）
def build_instance_timing_pipeline(
    events, source_files, class_map, step_infos, static_tree,
) -> dict:
    """
    一次跑完 Step 1→4，返回 timing_data 接口字段。
    内部依次调用 build_fwdbwd_flow_index → build_kernel_attribution_table
        → rollup_instance_timing → build_timing_panel_data。
    """
```

---

## 6. 回归与验证

### 6.1 回归测试基线

- `repo_migrate_workdir/internal_repo/testset/test_dag_rules.py` 12 条规则全 PASS（DAG 不应受 timing 重构影响）。
- `testset/test_timing.py`（已存在）现有 A1/A2 用例必须 PASS。

### 6.2 新增测例方案（落地到 `testset/test_timing.py`）

1. **Test C1: fwdbwd flow 解析**
   - synthetic events：构造 1 个 fwd op（tid=1, ts=0, dur=100）+ 1 个 bwd op（tid=2, ts=200, dur=50）+ 配对 fwdbwd flow（id=1）。
   - 断言：`build_fwdbwd_flow_index` 输出包含 1 个 entry，bwd→fwd 映射正确。

2. **Test C2: GroupTower 多实例时间区分**
   - 用 5698781 真实 trace + 源码。
   - 跑完整 timing pipeline。
   - 断言：`runtime_instance_timings_by_class["DenseTower"]` 中至少 2 个 instance 的 `inclusive_us["forward"]` 差异 > 5%（不再被均摊）。

3. **Test C3: backward kernel via fwdbwd flow narrowed**
   - 同 5698781 trace。
   - 断言：debug_stats["bwd_via_flow_narrowed"] > 0 且 bwd_unattributed == 0。

4. **Test C4: class 级 = sum of instances inclusive**
   - 任一 trace。
   - 断言：`class_durations[cls] ≈ sum(item.total_us for item in runtime_instance_timings_by_class[cls])`。

5. **Test C5: per-step 平均（防 Bug A 复发）**
   - 5698781 trace（5 ProfilerSteps）。
   - 断言：root.total_us / step_dur_us ≤ 100% （即 inclusive 没被乘 5x）。

### 6.3 防回退基准

完成后对比 baseline_test_log.txt 中：
- 5698781 root inclusive ≤ 100% step_dur
- DenseTower 多实例时间不再相同（CV > 0.05）
- backward recovered kernels 不少于 stacktrace baseline 数（约 8430 kernels）

---

## 7. 实施顺序

1. **保存 BASELINE.bak**（已完成）
2. 写本设计文档 → 提交 feat/timing-fix
3. 实现 Step 1 + 单元 print 验证 fwdbwd flow 数量
4. 实现 Step 2 + 验证 attribution_table 大小、weight 之和
5. 实现 Step 3 + 验证 inclusive ≥ self、根节点 inclusive 合理
6. 实现 Step 4 + 跑 build_timing_data_from_trace 接通现有调用方
7. 跑 test_dag_rules.py（12/7 PASS）+ test_timing.py（含新增 C1–C5）
8. 在 5698781 真实 trace 上目检 HTML：DenseTower / GroupTower 多实例时间不同
9. commit + push

---

## 8. 风险与回退

- 若 Step 2 candidates 过多导致大量均摊 → 通过 fwdbwd flow narrowing 应可压低；若仍发散，临时退回旧 callsite_map（保留作为内部 fallback）。
- 若 Step 3 拓扑排序遇环（不应发生，static_tree 已 DAG 化）→ 报警并退化为「无父子，仅 self」。
- 任一阶段破坏 test_dag_rules.py 7/7 PASS → 回退 `analyze_trace_timing_redesign_BASELINE.bak`。
