# Step 1b 设计文档：`filter_kernel_stack_chains`

## 1. 背景与目标

Step 1a 的 `build_kernel_stack_cost_table` 已经把 profiler trace 中的 CUDA kernel 统一整理为一条 kernel 对应一条 `KernelStackEntry`，并输出：

1. kernel 基本信息：`name`、`dur_us`。
2. 粗粒度阶段：`phase = "bwd" / "non-bwd"`。
3. 调用链：`chains`，包含 kernel 原生 `stack_traces` 解析出的 `base_chains`，以及 Step 1a 补齐的上层 `nn.Module python_function` frames。
4. 质量标识：`has_stack_traces`、`unmatched`。

Step 1b 在 Step 1a 之后执行，负责把 Step 1a 的输出进一步清洗为 Step 2 attribution 可以直接消费的输入。

核心目标：

1. **过滤非模型源码 frame**：从 `chains` 中剔除所有不属于用户模型源码的 frame（包括 torch、dynamo、autograd 等框架栈，以及框架注入的 `nn.Module` scope 标记）。
2. **精简归属链**：只保留能映射回 `source_files` 的 frame，确保 Step 2 attribution 逻辑只处理用户可理解的代码行。
3. **正式细化 phase**：把 Step 1a 的 `non-bwd` 进一步拆为 `fwd` / `other`；`bwd` 保持为 `bwd`。
4. **不改变 kernel 顺序**：输出顺序必须与 Step 1a 输入顺序一致。
5. **不在 Step 1b 做 instance attribution**：Step 1b 只做调用链清洗和 phase 细化，不识别 `InstanceKey`，不做权重分配。

<callout icon="bulb" bgc="5">
  Step 1b 的定位是 **normalization layer**：输入是 Step 1a 的 kernel stack cost table，输出仍然是一张 kernel stack cost table，只是 `chains` 更干净、`phase` 更精确。真正的 `kernel -> InstanceKey` 归属仍然属于 Step 2。
</callout>

---

## 2. 函数签名与输入输出

### 2.1 函数签名

```python
def filter_kernel_stack_chains(
    rows: list[KernelStackEntry],
    source_files: dict[str, str],
    step_infos: list[dict],
) -> list[KernelStackEntry]:
    ...
```

参数说明：

- `rows`：Step 1a `build_kernel_stack_cost_table(...)` 的输出。
- `source_files`：用户模型源码映射，key 通常是 basename，例如 `main_model.py`、`model.py`。
- `step_infos`：`extract_step_phase_intervals(...)` 的输出，用于将 `non-bwd` kernel 按时间窗口细化为 `fwd` / `other`。

### 2.2 输出约定

输出仍然是 `list[KernelStackEntry]`，字段沿用 Step 1a，但语义有如下变化：

```python
@dataclass
class KernelStackEntry:
    name: str
    dur_us: float
    phase: str
    has_stack_traces: bool
    chains: list[FrameInfo]
    unmatched: bool
```

字段变化：

- `phase`：取值从 Step 1a 的 `"bwd" / "non-bwd"` 变为 `"fwd" / "bwd" / "other"`。
- `chains`：从 Step 1a 的原始合并链，变为过滤后的用户模型链。
- `unmatched`：根据过滤后的 `chains` 重新计算。若清洗后没有任何可用 frame，则为 `True`。
- `has_stack_traces`：保持 Step 1a 原语义，不因过滤后链为空而改写。它只表达 kernel 原生 `stack_traces` 是否存在并被 Step 1a 解析到。

### 2.3 兼容字段

当前实现为了兼容现有 Step 2，`KernelStackEntry` 可能仍是 dict，并携带这些 legacy 字段：

- `event`
- `ts`
- `tid`
- `traces`
- `is_bwd`
- `mod_name`
- `legacy_chains`
- `event_idx`

Step 1b 需要保留这些字段，并只更新与本步骤相关的字段：

```python
row["phase"] = refined_phase
row["chains"] = filtered_chains
row["unmatched"] = len(filtered_chains) == 0
row["is_bwd"] = refined_phase == "bwd"
row["legacy_chains"] = [filtered_chains] if filtered_chains else []
```

---

## 3. Frame 分类规则

Step 1a 输出的 `chains` 按 **innermost first** 排列。Step 1b 对每个 frame 做分类，再决定是否保留。

### 3.1 用户模型源码 frame

满足以下任一条件时，认为是用户模型源码 frame，应保留：

1. `os.path.basename(frame.file) in source_files`。
2. `frame.file` 经路径归一化后能与 `source_files` 中任一文件路径后缀匹配。
3. `frame.file` 不是绝对路径，但 basename 命中 `source_files`。

建议实现为：

```python
def _is_user_source_frame(frame, source_files):
    fname = os.path.basename(str(frame.file or ""))
    if fname in source_files:
        return True
    normalized = str(frame.file or "").replace("\\", "/")
    for src_name in source_files:
        if normalized.endswith("/" + src_name) or normalized == src_name:
            return True
    return False
```

### 3.2 框架与冗余 frame (全部删除)

所有不命中用户源码 frame 的记录（包括 site-packages、torch 源码、以及框架自动注入的 `nn.Module` scope frame）均被视为 framework/redundant frame，应予以删除。

判断逻辑极其简化：
1. 调用 `_is_user_source_frame(frame, source_files)`。
2. 若返回 `True` 则保留，否则直接丢弃。

不再保留 `nn.Module` 特判。对于用户自定义 Module，其 `CallFrom` 记录的 callsite 必定位于模型源码内，已能被 3.1 规则覆盖。对于框架内置 Module（如 `torch.nn.Linear`），其 callsite 位于框架内部，属于应过滤的范畴。

### 3.3 保留顺序与去重

过滤后仍保持 **innermost first**，不反转顺序。

去重规则：

1. 连续重复 frame 可以压缩为一个。
2. 非连续重复 frame 不删除，避免误删递归或多次调用路径。
3. 移除“用户源码优先于 nn.Module”等复杂冲突处理，因为所有保留下来的 frame 均已确定属于用户源码。

### 3.4 Parallel Wrapper 处理规则 (DDP/FSDP)

对于并行训练中引入的 Wrapper 层，Step 1b 需要进行清洗，以保证归属类名的纯洁度。

#### 3.4.1 Wrapper 类跳过 (Wrapper Class Skip)

为了减少调用链中的冗余层，对于以下已知的并行/显存优化 Wrapper 类，其 frame 直接跳过：
- `PARALLEL_WRAPPER_CLASSES = ("FullyShardedDataParallel", "DistributedDataParallel", "CheckpointWrapper", "OffloadWrapper")`

逻辑：
1. 如果 frame 的 `func`（或解析出的 `class_name`）命中了上述列表。
2. 则该 frame 不进入 `filtered_chains`。

#### 3.4.2 类名前缀剥离 (Prefix Strip)

为了将 DDP/FSDP 包裹后的实例还原为原始模型类，对以下前缀进行剥离：
- `PARALLEL_WRAPPER_PREFIXES = ("FSDP", "DDP")`

逻辑：
1. 如果 frame 的 `class_name` 以上述前缀开头（如 `DDPDenseTower`、`FSDPTransformerLayer`）。
2. 则剥离该前缀，还原为真实类名（如 `DenseTower`、`TransformerLayer`）。
3. 剥离后的名称用于 Step 2 的 `class_map` 匹配。

---

## 4. phase 细化规则

### 4.1 Step 1a phase 输入

Step 1a 只区分：

- `bwd`：launch tid != main_thread_tid，或兼容逻辑判断为 backward。
- `non-bwd`：非 backward 的主线程 kernel。

Step 1b 输出：

- `bwd`
- `fwd`
- `other`

### 4.2 bwd 保持不变

如果输入 `row.phase == "bwd"`，Step 1b 直接输出：

```python
refined_phase = "bwd"
```

原因：Step 1a 已经通过 launch thread / cpu_op / fwdbwd flow 识别 backward；Step 1b 不重新推翻 backward 判断。

### 4.3 non-bwd 细化为 fwd / other

如果输入 `row.phase == "non-bwd"`，Step 1b 使用 kernel 时间与 profiler step phase interval 的重叠关系进一步判断。

推荐复用现有 `classify_kernel_phase(kernel_start, kernel_end, step_infos)`：

```python
phase = classify_kernel_phase(row["ts"], row["ts"] + row["dur_us"], step_infos)
if phase == "forward":
    refined_phase = "fwd"
else:
    refined_phase = "other"
```

映射规则：

<table header-row="true" header-col="false" col-widths="180,180,360">
    <tr>
        <td>Step 1a phase</td>
        <td>时间窗口分类</td>
        <td>Step 1b phase</td>
    </tr>
    <tr>
        <td>`bwd`</td>
        <td>任意</td>
        <td>`bwd`</td>
    </tr>
    <tr>
        <td>`non-bwd`</td>
        <td>`forward`</td>
        <td>`fwd`</td>
    </tr>
    <tr>
        <td>`non-bwd`</td>
        <td>`backward`</td>
        <td>`other`，并记录 debug anomaly</td>
    </tr>
    <tr>
        <td>`non-bwd`</td>
        <td>`optimize`</td>
        <td>`other`</td>
    </tr>
    <tr>
        <td>`non-bwd`</td>
        <td>`other` 或无匹配 step</td>
        <td>`other`</td>
    </tr>
</table>

说明：

- Step 1b 的正式 phase 名称使用 `fwd`，不是 `forward`，与用户需求“正式区分 fwd 和 bwd”保持一致。
- `optimize` 暂不独立暴露给 Step 2 attribution；它归入 `other`。后续如果 Timing 面板需要单独 optimize，可以在 Step 3/Step 4 再扩展。
- 若 `non-bwd` kernel 被 `classify_kernel_phase` 判为 `backward`，不强行改成 `bwd`，而是归为 `other` 并记录 debug 计数。这类情况代表 Step 1a thread-based 判断与 profiler phase window 不一致，应作为诊断信号。

---

## 5. 主流程伪代码

```python
def filter_kernel_stack_chains(rows, source_files, step_infos):
    out = []
    stats = {
        "total": 0,
        "phase_bwd": 0,
        "phase_fwd": 0,
        "phase_other": 0,
        "frames_in": 0,
        "frames_kept_user": 0,
        "frames_kept_module_scope": 0,
        "frames_dropped_framework": 0,
        "unmatched_after_filter": 0,
        "non_bwd_overlaps_backward": 0,
    }

    for row in rows:
        stats["total"] += 1

        raw_chains = row.chains
        filtered = []
        for frame in raw_chains:
            stats["frames_in"] += 1
            if _is_user_source_frame(frame, source_files):
                filtered.append(frame)
                stats["frames_kept_user"] += 1
            else:
                stats["frames_dropped_framework"] += 1

        filtered = _dedup_consecutive_frames(filtered)
        refined_phase = _refine_phase(row, step_infos, stats)

        new_row = copy_row_preserving_legacy_fields(row)
        new_row.phase = refined_phase
        new_row.chains = filtered
        new_row.unmatched = len(filtered) == 0
        new_row.is_bwd = refined_phase == "bwd"  # dict legacy only
        new_row.legacy_chains = [filtered] if filtered else []

        stats[f"phase_{refined_phase}"] += 1
        if new_row.unmatched:
            stats["unmatched_after_filter"] += 1
        out.append(new_row)

    attach_or_return_debug_stats(out, stats)
    return out
```

---

## 6. 与现有 Step 2 的衔接

当前 Step 2 `build_kernel_attribution_table(...)` 仍需要：

1. `row["chains"]` 或 `row["legacy_chains"]` 用于提取用户 frame 链。
2. `row["phase"]` / `row["is_bwd"]` 用于区分 backward 路径和非 backward 路径。
3. `row["mod_name"]` 用于 wrapped fallback。
4. `row["event_idx"]` 用于回写 kernel attribution。

Step 1b 接入后，建议 pipeline 变为：

```python
fwdbwd_index = build_fwdbwd_flow_index(events)
raw_rows = build_kernel_stack_cost_table(events, source_files, fwdbwd_index)       # Step 1a
clean_rows = filter_kernel_stack_chains(raw_rows, source_files, step_infos)       # Step 1b
kernel_attribution, debug_stats = build_kernel_attribution_table_from_rows(
    clean_rows, events, source_files, class_map, step_infos, fwdbwd_index,
)
```

如果暂时不拆 `build_kernel_attribution_table(...)` 签名，也可以在其内部插入 Step 1b：

```python
kernel_rows = build_kernel_stack_cost_table(events, source_files, fwdbwd_index)
kernel_rows = filter_kernel_stack_chains(kernel_rows, source_files, step_infos)
```

<callout icon="bulb" bgc="3">
  建议优先采用“内部插入 Step 1b”的方式，减少对现有调用方的侵入；待 Step 2 稳定后，再把 `build_kernel_attribution_table_from_rows(...)` 拆出来，让 Step 1a/1b/2 边界更清晰。
</callout>

---

## 7. Debug stats 与日志

Step 1b 建议输出 debug stats，便于后续排查过滤过严或 phase 异常。

```python
step1b_stats = {
    "total": int,
    "phase_fwd": int,
    "phase_bwd": int,
    "phase_other": int,
    "frames_in": int,
    "frames_kept_user": int,
    "frames_dropped_framework": int,
    "unmatched_after_filter": int,
    "non_bwd_overlaps_backward": int,
}
```

推荐打印或并入 Step 2 debug stats：

```text
[Timing Step1b] kernels=12345 phase_fwd=6789 phase_bwd=4321 phase_other=1235
[Timing Step1b] frames_in=88888 kept_user=22222 dropped_framework=66666 unmatched=12
```

质量预期：

- `frames_dropped_framework > 0`：说明清洗确实发生。
- `frames_kept_user > 0`：说明用户源码 frame 没被误删。
- `phase_fwd > 0` 且 `phase_bwd > 0`：真实训练 trace 应同时存在正反向 kernel。
- `non_bwd_overlaps_backward` 应接近 0；若很大，需要回查 Step 1a 的 backward 判定。

---

## 8. 测例设计方案

### 8.1 T-1b-1：过滤 framework frame，只保留用户 frame

构造 synthetic `KernelStackEntry`：

```python
chains = [
    FrameInfo("/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", 1500, "_call_impl"),
    FrameInfo("/repo/model.py", 42, "forward"),
    FrameInfo("/usr/local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", 300, "torch_dynamo_resume_in_forward"),
]
```

`source_files = {"model.py": "..."}`。

断言：

```python
assert [basename(f.file) for f in out[0].chains] == ["model.py"]
assert out[0].unmatched is False
assert out[0].has_stack_traces is True
```

### 8.2 T-1b-2：`nn.Module` frame 若在模型内则保留，否则删除

构造两条记录：
1. `chains = [FrameInfo("main_model.py", 169, "nn.Module: DDPAwemeLiveCVR_0, callsite: 169")]`
2. `chains = [FrameInfo("torch/nn/modules/linear.py", 114, "nn.Module: Linear_0, callsite: 114")]`

`source_files = {"main_model.py": "..."}`。

断言：
- 记录 1 保留，`chains` 长度为 1。
- 记录 2 过滤，`chains` 为空，`unmatched` 为 `True`。

### 8.3 T-1b-3：`non-bwd` + forward window → `fwd`

构造：

- row.phase = `"non-bwd"`
- row.ts / row.dur_us 落在 `step_infos[0]["forward"]` 区间内

断言：

```python
assert out[0].phase == "fwd"
assert out[0].is_bwd is False
```

### 8.4 T-1b-4：`non-bwd` + optimize/无窗口 → `other`

构造两条 row：

1. 落在 `optimize` 区间。
2. 不落在任何 profiler step 区间。

断言：

```python
assert out[0].phase == "other"
assert out[1].phase == "other"
```

### 8.5 T-1b-5：`bwd` 保持 `bwd`

构造 row.phase = `"bwd"`，即使 ts 落在 forward window，也不改写。

断言：

```python
assert out[0].phase == "bwd"
assert out[0].is_bwd is True
```

### 8.6 T-1b-6：过滤后空链应 unmatched，但不改 `has_stack_traces`

构造 row：

- `has_stack_traces = True`
- `chains` 全部是 framework frame

断言：

```python
assert out[0].chains == []
assert out[0].unmatched is True
assert out[0].has_stack_traces is True
```

### 8.7 T-1b-7：FSDP/DDP prefix strip

构造 `FrameInfo`：
- `chains = [FrameInfo("model.py", 100, "nn.Module: DDPDenseTower_0, callsite: 100")]`
- `source_files = {"model.py": "..."}`

断言：
- `chains[0]` 的 `func`（或 class_name）应被处理为 `nn.Module: DenseTower_0, callsite: 100`。
- 归属逻辑应能正确识别出 `DenseTower` 类。

### 8.8 T-1b-8：Wrapper class skip

构造 `FrameInfo`：
- `chains = [FrameInfo("torch/nn/parallel/distributed.py", 50, "DistributedDataParallel.forward")]`
- 虽然可能因为某些原因未被 framework filter 滤掉（或属于被允许的 wrapper 路径），但命中 `PARALLEL_WRAPPER_CLASSES`。

断言：
- 该 frame 被跳过，`chains` 长度减少。

### 8.9 T-1b-9：Phase 空链归 other 的修正

修正：当 kernel 既不属于 `forward` 也不属于 `backward` 窗口时，其 `phase` 必须被归为 `other`。

断言：
- 构造 `non-bwd` kernel 落在空档期或 `optimize` 期，`phase` 应为 `other`。
- 若 `chains` 为空且无法通过窗口判定，默认归为 `other`。

---

## 9. 实施顺序建议

1. 在 `build_kernel_stack_cost_table(...)` 后新增独立函数 `filter_kernel_stack_chains(...)`，不直接改 Step 1a 主体。
2. 给 frame 访问做兼容 helper，支持 dataclass `FrameInfo` 和现有 tuple `(file, line, func)` 两种形态。
3. 先接入 Step 2 内部调用，保证现有 `build_kernel_attribution_table(...)` caller 不变。
4. 增加 `testset/test_timing.py` Step 1b synthetic tests。
5. 跑 timing 单测与 `python3 testset/test_dag_rules.py`，确认 DAG 规则不回退。
6. 在 5698781 trace 上检查 Step 1b stats：`fwd/bwd/other` 分布合理，framework frames 有明显下降，DenseTower 多实例归属不回退。

---

## 10. 风险与回退

### 10.1 过滤过严

风险：用户模型 frame 被误判为 framework frame，导致 `unmatched_after_filter` 上升，Step 2 attribution 覆盖率下降。

缓解：

- 用户源码命中优先级最高，只要 basename 命中 `source_files` 就保留。
- 首版采用 conservative deny-list，不做复杂正则兜底。
- stats 中跟踪 `unmatched_after_filter` 与 Step 2 attributed/unattributed 数。

### 10.2 过滤过松

风险：framework frame 混入 Step 2，干扰 class boundary 识别。

缓解：

- 对明显框架路径和函数名建立 deny-list。
- Step 2 `_extract_instance_keys_from_stack(...)` 仍通过 `class_map` 做最终 class 识别，非用户 frame 即使残留也不应直接生成 InstanceKey。

### 10.3 phase 与 Step 3 命名不一致

风险：Step 1b 输出 `fwd`，但现有 Step 3 `self_us` 可能使用 `forward` key。

缓解：

- Step 1b 是 Step 2 attribution 输入；Step 3 累加仍可在 `rollup_instance_timing(...)` 内把 `fwd` 映射回 `forward`。
- 或者 Step 1b 同时保留兼容字段：`phase="fwd"`，`phase_legacy="forward"`。
- 首选方案：Step 1b 对外正式输出 `fwd/bwd/other`；Step 3 做一次显式映射。

### 10.4 `nn.Module` scope frame 被误删

风险：无 stack trace 或 stack trace 截断时，Step 2 wrapped fallback 和 module scope 信息不足。

缓解：

- 明确 `_is_nn_module_scope_frame(...)` 优先于 framework deny-list。
- 对 module scope frame 增加专门测例 T-1b-2。

---

## 11. 本 Step 的非目标

Step 1b 不处理以下事项：

1. 不生成 `InstanceKey`。
2. 不做 kernel 权重分配。
3. 不做 backward fwdbwd narrowing。
4. 不计算 self/inclusive timing。
5. 不修改 DAG 静态构建逻辑。
6. 不基于正则解析用户源码；源码结构识别仍归 Step 2 / DAG AST 逻辑负责。
