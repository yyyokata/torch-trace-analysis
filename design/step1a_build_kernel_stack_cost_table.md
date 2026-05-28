# Step 1a 设计文档：`build_kernel_stack_cost_table`

## 1. 概述与函数签名

`build_kernel_stack_cost_table` 是 Timing attribution pipeline 的 Step 1a，负责把 profiler trace 中的 CUDA kernel 事件转换为统一的 kernel 调用栈成本表。

核心目标：

1. **一条 kernel 输出一条记录**，保留 kernel 名称、耗时、阶段、是否有原生 stack trace、调用链与 unmatched 状态。
2. **统一两段式流程**：
   - Step 1：所有 kernel 先解析 `args["stack"]["stack_traces"]`，得到 `base_chains`。
   - Step 2：所有 kernel 再按 phase 补上层 `nn.Module` 调用帧。
3. **不再使用 Path A / A' / B / C 标识**。旧 `path` 字段删除，替换为 `has_stack_traces: bool`。
4. **无论有无 stack_traces、无论正反向，都执行 Step 2 补上层**。
5. **保持输入顺序稳定**。输出列表顺序必须与 `events` 中 kernel 出现顺序一致。

函数签名保持：

```python
def build_kernel_stack_cost_table(
    events: list[dict],
    source_files: dict[str, str],
    fwdbwd_index: dict,
) -> list[KernelStackEntry]:
    ...
```

## 2. 数据结构定义

### 2.1 `FrameInfo`

```python
@dataclass
class FrameInfo:
    file: str
    line: int
    func: str
```

字段说明：

- `file`：frame 所在源码文件路径。
- `line`：frame 所在行号。无法解析时使用 `0`。
- `func`：函数名、方法名或 `nn.Module` 名称。

### 2.2 `KernelStackEntry`

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

字段说明：

- `name`：kernel name，来自 kernel event 的 `name` 字段。
- `dur_us`：kernel duration，单位为 microsecond，来自 kernel event 的 `dur` 字段。
- `phase`：kernel 阶段，取值为 `"bwd"` / `"non-bwd"`。`bwd` 表示 launch tid != main_thread_tid；`non-bwd` 表示 launch tid == main_thread_tid。Step 1a 不区分 fwd/other，留给 Step 1b。
- `has_stack_traces`：Step 1 是否解析到非空 `stack_traces`。有原生 stack trace 时为 `True`，否则为 `False`。
- `chains`：统一调用链格式，`innermost first`。内容为 `base_chains + upper_frames`。
- `unmatched`：是否未找到任何有效调用链。`True` 表示 `chains=[]`。

### 2.3 顺序约定

`chains` 必须统一为 **innermost first**：

```text
chains[0] = 最内层 frame / 最具体的 nn.Module
chains[-1] = 最外层 frame / 最外层 nn.Module
```

`base_chains` 来自 kernel 自带 stack trace，`upper_frames` 来自 phase 对应的 `nn.Module python_function` 覆盖链。最终：

```python
chains = base_chains + upper_frames
unmatched = len(chains) == 0
```

## 3. 两段式流程

### 3.1 Step 0：公共索引与 phase 判定

Step 1a 复用 `build_fwdbwd_flow_index(events)` 的索引结构：

```python
{
    "corr_to_launch":       {correlation: launch_event},
    "fwdbwd_f_by_tid_ts":   {(tid, ts): event},
    "fwdbwd_s_by_id":       {id: event},
    "tid_cpuop_sorted":     {tid: [events sorted by ts]},
    "tid_nn_module_sorted": {tid: [events sorted by ts]},
    "main_thread_tid":      int,
}
```

每个 kernel 先通过 correlation 找 launch：

```python
launch = corr_to_launch.get(kernel.args.correlation)
phase = "bwd" if (launch and launch.tid != main_thread_tid) else "non-bwd"
```

现有实现中保留了少量兼容逻辑，用于 synthetic trace 或缺少主线程 `nn.Module` 的场景；但主设计原则仍是 launch 线程与 `main_thread_tid` 的关系。

### 3.2 Step 1：所有 kernel 解析 `stack_traces`

所有 kernel 都执行：

```text
解析 args["stack"]["stack_traces"] → base_chains（innermost first）
has_stack_traces = len(base_chains) > 0
若无 stack_traces → base_chains = []
```

说明：

- `has_stack_traces` 只表达 Step 1 是否解析到非空 stack trace。
- Step 1 不再区分 dynamo resume 顶帧，不再使用 `is_dynamo_resume_top_frame`。
- Step 1 不做模型源码匹配和 framework frame 过滤；这些逻辑留给 Step 1b/Step 2 attribution。

### 3.3 Step 2：按 phase 补上层调用帧

Step 2 对所有 kernel 执行。

#### `phase == "non-bwd"`

```text
在 main_thread_tid 上查找覆盖 launch_ts 的 nn.Module python_function 帧
取所有覆盖帧按 dur 升序排序（innermost first）→ upper_frames
```

使用 main thread 的原因：正向或其他主线程 kernel 的上层 `nn.Module` 调用帧在主线程 `python_function` 事件中最稳定。

#### `phase == "bwd"`

```text
走 fwdbwd flow 跳回正向线程：
  1. 在 backward launch tid 上找覆盖 launch_ts 的 cpu_op 链
  2. 找 cpu_op.ts 对齐的 fwdbwd ph=f
  3. 用同 id 找 fwdbwd ph=s
  4. 在正向 tid 上查找覆盖 s_ev.ts 的 nn.Module python_function 帧
  5. 覆盖帧按 dur 升序排序（innermost first）→ upper_frames
```

### 3.4 合并输出

```python
chains = base_chains + upper_frames
unmatched = len(chains) == 0
```

输出字段：

```python
KernelStackEntry(
    name=kernel.get("name", ""),
    dur_us=float(kernel.get("dur", 0.0) or 0.0),
    phase=phase,
    has_stack_traces=has_stack_traces,
    chains=chains,
    unmatched=unmatched,
)
```

## 4. 主函数伪代码

```python
def build_kernel_stack_cost_table(events, source_files, fwdbwd_index):
    if fwdbwd_index is None:
        fwdbwd_index = build_fwdbwd_flow_index(events)

    rows = []
    for kernel in events:
        if not is_cuda_kernel_event(kernel):
            continue

        launch = find_launch_event(kernel)
        phase = infer_phase(launch, fwdbwd_index)
        launch_ts = launch.get("ts") if launch else None

        # Step 1: parse stack_traces for every kernel
        traces = kernel.get("args", {}).get("stack", {}).get("stack_traces", [])
        base_chains = parse_stack_traces(traces)
        has_stack_traces = len(base_chains) > 0

        # Step 2: append upper nn.Module frames for every kernel
        if phase == "non-bwd":
            upper_events = get_covering_nn_modules(main_thread_tid, launch_ts)
        else:
            upper_events = get_forward_nn_modules_via_fwdbwd_flow(launch)
        upper_frames = [python_module_event_to_frame(ev) for ev in upper_events]

        chains = base_chains + upper_frames
        rows.append(KernelStackEntry(
            name=kernel.get("name", ""),
            dur_us=float(kernel.get("dur", 0.0) or 0.0),
            phase=phase,
            has_stack_traces=has_stack_traces,
            chains=chains,
            unmatched=(len(chains) == 0),
        ))

    return rows
```

## 5. 流程图

```text
CUDA kernel
  |
  |-- Step 0: correlation -> cuda_runtime launch -> phase
  |
  |-- Step 1: parse args["stack"]["stack_traces"]
  |       |
  |       `-- base_chains, has_stack_traces
  |
  `-- Step 2: append upper frames by phase
          |
          |-- phase == non-bwd
          |       `-- main_thread_tid covering nn.Module python_function frames
          |
          `-- phase == bwd
                  `-- backward cpu_op -> fwdbwd ph=f -> ph=s
                      -> forward covering nn.Module python_function frames

chains = base_chains + upper_frames
unmatched = len(chains) == 0
```

## 6. 测例更新方案

### 6.1 T-1a-5：有 stack_traces 的 non-bwd kernel

基于真实 trace 中一个带 `stack_traces` 且 launch 在 main thread 的 kernel。

断言：

```python
assert entry["has_stack_traces"] is True
assert entry["phase"] == "non-bwd"
assert entry["unmatched"] is False
assert entry["chains"]
assert "torch_dynamo_resume_in_" in str(entry["chains"][0][2])
assert entry["chains"][-1] == expected_outer_frame
```

不再断言 `path == "A'"`。

### 6.2 T-1a-6：有 stack_traces 但无 Step 2 覆盖帧

使用真实 kernel + 真实 launch + 不覆盖 launch_ts 的 `nn.Module` 事件构造 trace 截断场景。

断言：

```python
assert entry["has_stack_traces"] is True
assert entry["phase"] == "non-bwd"
assert entry["unmatched"] is False
assert entry["chains"]
assert entry["chains"][-1] != non_covering_module_frame
```

注意：两段式后，即使 Step 2 没找到覆盖帧，只要 Step 1 有 `base_chains`，也不应 `unmatched=True`，更不应清空 `chains`。

### 6.3 T-1a-7：有 stack_traces 的 backward kernel

新增真实或 synthetic backward kernel，要求：

1. kernel 自带非空 `stack_traces`。
2. launch tid 不是 `main_thread_tid`，因此 `phase == "bwd"`。
3. backward launch tid 上存在 cpu_op 覆盖链。
4. cpu_op.ts 能通过 `fwdbwd ph=f/ph=s` 跳回正向线程。
5. 正向线程存在覆盖 `s_ev.ts` 的 `nn.Module python_function` 帧。

断言：

```python
assert entry["has_stack_traces"] is True
assert entry["phase"] == "bwd"
assert entry["unmatched"] is False
assert entry["chains"]
assert entry["chains"][-1] == expected_forward_nn_module_frame
```

该测例验证：即使 backward kernel 已有 stack trace，Step 2 仍会走 fwdbwd flow 补正向上层 `nn.Module` 帧。

## 7. 兼容性说明

当前实现为了兼容后续 attribution pipeline，仍可保留内部 legacy 字段，例如：

- `event`
- `ts`
- `tid`
- `traces`
- `is_bwd`
- `mod_name`
- `legacy_chains`
- `event_idx`

但对 Step 1a 新接口来说，`path` 已删除，`has_stack_traces` 是唯一公开标识。