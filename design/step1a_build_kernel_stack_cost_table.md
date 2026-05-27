# Step 1a 设计文档：`build_kernel_stack_cost_table`

## 1. 概述与函数签名

### 1.1 目标

`build_kernel_stack_cost_table` 是 Timing attribution pipeline 的 Step 1a，负责把 profiler trace 中的 CUDA kernel 事件转换为统一的 kernel 调用栈成本表。

该步骤的核心目标是：

1. **一条 kernel 输出一条记录**，保留 kernel 名称、耗时、阶段、路径和调用链。
2. **统一三类数据来源**：
   - Path A：kernel 自带 `stack_traces`。
   - Path B：无 `stack_traces` 的 backward kernel，通过 `fwdbwd` flow 跳回正向线程，恢复正向 `nn.Module` 层级。
   - Path C：无 `stack_traces` 的 forward kernel，直接在 launch CPU 线程上查找覆盖的正向 `nn.Module`。
3. **只做调用链收集，不做 frame 过滤**。Step 1a 保留所有可用 frame，模型源码匹配和 framework frame 过滤推迟到 Step 1b。
4. **保持输入顺序稳定**。输出列表顺序必须与 `events` 中 kernel 出现顺序一致。

### 1.2 函数签名

```python
def build_kernel_stack_cost_table(
    events: list[dict],
    source_files: dict[str, str],
    fwdbwd_index: dict,
) -> list[KernelStackEntry]:
    ...
```

### 1.3 参数说明

| 参数 | 类型 | Step 1a 中的用途 |
|---|---|---|
| `events` | `list[dict]` | Chrome Trace 的事件列表。Step 1a 从中遍历 kernel、查找 `cuda_runtime` launch 事件、查找 `cpu_op` 覆盖链、查找 `python_function` 中的 `nn.Module` 覆盖链。 |
| `source_files` | `dict[str, str]` | Step 1a 暂不使用。保留该参数是为了与 Step 1b 的模型代码结构匹配接口保持一致。 |
| `fwdbwd_index` | `dict` | 预构建的完整索引结构，供 Step 0 通过 correlation 查 launch、供 Path B 从 backward autograd event 跳转到 forward thread、供 Path B/C 查找时间覆盖链。结构见第 5 节。 |

### 1.4 返回值

返回 `list[KernelStackEntry]`。每个 CUDA kernel 对应一条 `KernelStackEntry`。

---

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

| 字段 | 含义 |
|---|---|
| `file` | frame 所在源码文件路径。Path A 来自 `stack_traces` 字符串；Path B/C 来自 `python_function` 事件的源码信息。 |
| `line` | frame 所在行号。无法解析时建议使用 `0`，但应尽量从 trace 字段中恢复。 |
| `func` | 函数名、方法名或 `nn.Module` 名称。Path B/C 中应包含对应 `nn.Module` 的可读名称。 |

### 2.2 `KernelStackEntry`

```python
@dataclass
class KernelStackEntry:
    name: str
    dur_us: float
    phase: str
    path: str
    chains: list[FrameInfo]
    unmatched: bool
```

字段说明：

| 字段 | 含义 |
|---|---|
| `name` | kernel name，来自 kernel event 的 `name` 字段。 |
| `dur_us` | kernel duration，单位为 microsecond，来自 kernel event 的 `dur` 字段。 |
| `phase` | kernel 阶段，取值为 `"bwd"` / `"non-bwd"`。`bwd` 表示 launch tid != main_thread_tid；`non-bwd` 表示 launch tid == main_thread_tid。Step 1a 不区分 fwd/other，留给 Step 1b。 |
| `path` | 命中的路径，取值为 `"A"` / `"B"` / `"C"`。 |
| `chains` | 统一调用链格式，`innermost first`。Path A 为解析出的 stack frames；Path B/C 为覆盖 kernel launch 时间点的 `nn.Module` frame 链。 |
| `unmatched` | 是否未找到有效 `nn.Module` 覆盖链。`True` 表示当前 kernel 没有有效归属链，`chains=[]`。 |

### 2.3 顺序约定

`chains` 必须统一为 **innermost first**：

```text
chains[0] = 最内层 frame / 最具体的 nn.Module
chains[-1] = 最外层 frame / 最外层 nn.Module
```

对 Path B/C 来说，覆盖链按 `dur` 升序排序即可得到 innermost first。

---

## 3. 三条路径详细逻辑

### 3.1 主路径判断

路径判断顺序固定：

```python
Step 0（所有路径共用）:
  launch = corr_to_launch[kernel.correlation]
  is_bwd = (launch.tid != main_thread_tid)
  phase = "bwd" if is_bwd else "non-bwd"

st = kernel.get("args", {}).get("stack", {})
traces = st.get("stack_traces", []) if isinstance(st, dict) else []
if traces:  # Path A
    # Path A
elif phase == "bwd":
    # Path B
else:
    # Path C
```

注意：

- 只要 `args["stack"]["stack_traces"]` 存在且非空，优先走 Path A。
- `phase` 由 Step 0 统一确定：`bwd` 表示 launch tid != main_thread_tid；`non-bwd` 表示 launch tid == main_thread_tid。
- Path C 是无 `stack_traces` 且 `phase="non-bwd"` 时的默认路径。

### 3.2 公共索引

为了避免每个 kernel 都全量扫描 `events`，实现时建议在函数开始阶段预构建以下索引：

```python
runtime_by_corr: dict[int, list[dict]]
cpu_ops_by_tid: dict[int, list[dict]]
python_funcs_by_tid: dict[int, list[dict]]
```

建议构建逻辑：

```python
runtime_by_corr = defaultdict(list)
cpu_ops_by_tid = defaultdict(list)
python_funcs_by_tid = defaultdict(list)

for ev in events:
    cat = ev.get("cat")
    tid = ev.get("tid")
    args = ev.get("args") or {}

    if cat == "cuda_runtime" and "correlation" in args:
        runtime_by_corr[args["correlation"]].append(ev)

    if cat == "cpu_op":
        cpu_ops_by_tid[tid].append(ev)

    if cat == "python_function":
        python_funcs_by_tid[tid].append(ev)

for bucket in cpu_ops_by_tid.values():
    bucket.sort(key=lambda e: (e.get("ts", 0), e.get("dur", 0)))

for bucket in python_funcs_by_tid.values():
    bucket.sort(key=lambda e: (e.get("ts", 0), e.get("dur", 0)))
```

### 3.3 公共辅助函数

#### 3.3.1 判断时间覆盖

```python
def covers(ev: dict, ts: float) -> bool:
    start = ev.get("ts")
    dur = ev.get("dur", 0)
    if start is None:
        return False
    return start <= ts <= start + dur
```

建议使用闭区间 `start <= ts <= start + dur`，与 profiler 时间包含语义保持一致。

#### 3.3.2 通过 correlation 找 launch 事件

```python
def find_launch_event(kernel: dict) -> dict | None:
    args = kernel.get("args") or {}
    corr = args.get("correlation")
    if corr is None:
        return None

    candidates = runtime_by_corr.get(corr, [])
    if not candidates:
        return None

    # 正常情况下一个 correlation 对应一个 cuda_runtime launch。
    # 若存在多个候选，优先选择最接近 kernel launch 语义的事件。
    return choose_best_cuda_runtime_event(candidates)
```

`choose_best_cuda_runtime_event` 的最小可行策略：返回第一个候选。

如需增强，可优先选择 name 中包含 `cudaLaunchKernel` / `cudaGraphLaunch` / `cuLaunchKernel` 的事件。

#### 3.3.3 获取 CPU 覆盖链

```python
def get_covering_cpu_ops(tid: int, ts: float) -> list[dict]:
    chain = [ev for ev in cpu_ops_by_tid.get(tid, []) if covers(ev, ts)]
    return sorted(chain, key=lambda e: e.get("dur", 0))  # innermost first
```

#### 3.3.4 获取正向 `nn.Module` 覆盖链

```python
def get_covering_nn_modules(tid: int, ts: float) -> list[dict]:
    chain = []
    for ev in python_funcs_by_tid.get(tid, []):
        if not covers(ev, ts):
            continue
        if "nn.Module:" not in str(ev.get("name", "")):
            continue
        chain.append(ev)

    return sorted(chain, key=lambda e: e.get("dur", 0))  # innermost first
```

#### 3.3.5 将 `python_function` 事件转为 `FrameInfo`

```python
def python_module_event_to_frame(ev: dict) -> FrameInfo:
    args = ev.get("args") or {}
    return FrameInfo(
        file=extract_file_from_python_event(ev),
        line=extract_line_from_python_event(ev),
        func=extract_func_from_python_event(ev),
    )
```

字段来源建议：

1. 优先从 `args` 中读取 `file` / `line` / `func` 类字段。
2. 若 trace 使用 `Call stack` / `Code Location` / `Python id` 关联信息，可沿用现有 analyzer 中的解析逻辑。
3. 对 Path B/C 的测试约束是 `chains[0].func` 包含对应 `nn.Module` 名，因此 `func` 至少应保留 `event.name` 中的 `nn.Module:` 后半部分或完整 `event.name`。

---

## 3.4 Path A：kernel 有 `stack_traces`

### 3.4.1 适用条件

```python
kernel.get("args", {}).get("stack", {}).get("stack_traces", []) 非空
```

Path A 同时适用于 forward 和 backward kernel。

### 3.4.2 输入格式

`stack_traces` 是多行字符串，每行形如：

```text
  File "xxx.py", line N, in func_name
```

### 3.4.3 输出规则

```text
chains = parsed stack frames, innermost first
path = "A"
unmatched = False
phase = 由 Step 0 统一确定（"bwd" 或 "non-bwd"）
```

### 3.4.4 stack trace 解析伪代码

```python
def parse_stack_traces(stack_text: str) -> list[FrameInfo]:
    frames = []
    for line in stack_text.splitlines():
        # 示例：  File "xxx.py", line N, in func_name
        m = STACK_FRAME_PATTERN.match(line)
        if not m:
            continue
        frames.append(FrameInfo(
            file=m.group("file"),
            line=int(m.group("line")),
            func=m.group("func"),
        ))

    # profiler stack_traces 的原始顺序需要根据实际 trace 确认。
    # Step 1a 对外保证 innermost first。
    return normalize_to_innermost_first(frames)
```

说明：

- 这里的正则只用于解析标准 profiler stack trace 文本行，不用于源码解析，不违反 AST-first 源码解析约束。
- Step 1a 不做 frame 过滤，framework frame、runtime frame、用户 frame 全部保留。

### 3.4.5 phase 来源

Path A 不再做独立 phase 推断，phase 由 Step 0 统一确定：

```python
launch = corr_to_launch[kernel.correlation]
is_bwd = (launch.tid != main_thread_tid)
phase = "bwd" if is_bwd else "non-bwd"
```

Step 1a 不区分 fwd/other，留给 Step 1b。

---

## 3.5 Path B：无 `stack_traces` 的 backward kernel

### 3.5.1 适用条件

```python
not kernel.get("args", {}).get("stack", {}).get("stack_traces", [])
and phase == "bwd"
```

Path B 的判断依据：Step 0 中 kernel 对应 launch 点所在 CPU 线程不是 `main_thread_tid`，即 `phase == "bwd"`。

### 3.5.2 精确步骤

以 `gather_backward_kernel corr=790124` 的验证结果为基准，Path B 步骤如下：

```text
Step 1. kernel.correlation = C
Step 2. 找 cat=cuda_runtime, correlation=C -> launch_ev
        取 launch_ev.tid = bwd_tid, launch_ev.ts = launch_ts
Step 3. 在 bwd_tid 上找所有 cat=cpu_op 覆盖 launch_ts
        覆盖链按 dur 升序排序，得到 innermost first
Step 4. 遍历覆盖链，找 ts 与 bwd_tid 上 fwdbwd ph=f 精确对齐的 cpu_op
        命中后取 fwdbwd ph=f 的 id
Step 5. 找 cat=fwdbwd, ph=s, id=同一个 id
        得到正向 tid 和正向 ts_fwd
Step 6. 在正向 tid 上找覆盖 ts_fwd 的 cat=python_function，且 name 含 "nn.Module:"
        按 dur 升序排序，转成 FrameInfo 链
Step 7. 输出 path="B"，phase="bwd"，chains=FrameInfo 列表
        若 Step 4/5/6 失败，则 unmatched=True，chains=[]
```

### 3.5.3 phase 判断

Path B 不再通过 `cpu_op` name 或 `autograd::engine::evaluate_function:` 判断 backward，而是直接复用 Step 0：

```python
is_bwd = (launch.tid != main_thread_tid)
phase = "bwd" if is_bwd else "non-bwd"
```

Path B 仅在 `phase == "bwd"` 时进入。

### 3.5.4 查找 backward `fwdbwd ph=f` 事件

已验证事实：PyTorch profiler 在 autograd function 开始时打 `fwdbwd ph=f`，其 `ts` 严格等于对应 backward `cpu_op.ts`。

```python
def find_fwdbwd_f_for_backward_chain(
    bwd_tid: int,
    cpu_chain: list[dict],
    fwdbwd_index: dict,
) -> dict | None:
    fwdbwd_f_by_tid_ts = fwdbwd_index.get("fwdbwd_f_by_tid_ts", {})
    for cpu_op in cpu_chain:
        op_ts = cpu_op.get("ts")
        if op_ts is None:
            continue
        f_ev = fwdbwd_f_by_tid_ts.get((bwd_tid, op_ts))
        if f_ev is not None:
            return f_ev
    return None
```

由于 `fwdbwd_f_by_tid_ts` 在预构建阶段生成，Path B Step 4 为 O(depth)：

```python
f_ev = fwdbwd_index["fwdbwd_f_by_tid_ts"].get((bwd_tid, cpu_op["ts"]))
```

### 3.5.5 通过 `fwdbwd ph=s` 跳回正向线程

```python
def find_forward_start_from_fwdbwd_f(f_ev: dict, fwdbwd_index: dict) -> dict | None:
    flow_id = f_ev.get("id")
    if flow_id is None:
        return None

    return fwdbwd_index.get("fwdbwd_s_by_id", {}).get(flow_id)
```

返回的 `s_ev` 中：

```text
s_ev.tid = 正向 CPU 线程
s_ev.ts  = 正向开始时间点
```

### 3.5.6 Path B 主伪代码

```python
def build_entry_path_b(kernel: dict) -> KernelStackEntry:
    launch_ev = find_launch_event(kernel)
    if launch_ev is None:
        return unmatched_entry(kernel, path="B", phase="bwd")

    bwd_tid = launch_ev.get("tid")
    launch_ts = launch_ev.get("ts")
    if bwd_tid is None or launch_ts is None:
        return unmatched_entry(kernel, path="B", phase="bwd")

    cpu_chain = get_covering_cpu_ops(bwd_tid, launch_ts)
    if not cpu_chain:
        return unmatched_entry(kernel, path="B", phase="bwd")

    f_ev = find_fwdbwd_f_for_backward_chain(bwd_tid, cpu_chain, fwdbwd_index)
    if f_ev is None:
        return unmatched_entry(kernel, path="B", phase="bwd")

    s_ev = find_forward_start_from_fwdbwd_f(f_ev, fwdbwd_index)
    if s_ev is None:
        return unmatched_entry(kernel, path="B", phase="bwd")

    fwd_tid = s_ev.get("tid")
    ts_fwd = s_ev.get("ts")
    if fwd_tid is None or ts_fwd is None:
        return unmatched_entry(kernel, path="B", phase="bwd")

    module_chain = get_covering_nn_modules(fwd_tid, ts_fwd)
    if not module_chain:
        return unmatched_entry(kernel, path="B", phase="bwd")

    return KernelStackEntry(
        name=kernel.get("name", ""),
        dur_us=float(kernel.get("dur", 0.0) or 0.0),
        phase="bwd",
        path="B",
        chains=[python_module_event_to_frame(ev) for ev in module_chain],
        unmatched=False,
    )
```

### 3.5.7 `Fwd thread id` 字段的定位

`cpu_op` 在 backward thread 上可能带有 `Fwd thread id` 字段。该字段可用于辅助校验，例如：

```python
if fwd_thread_id_from_cpu_op is not None:
    assert_or_warn(fwd_thread_id_from_cpu_op == s_ev.get("tid"))
```

但主路径仍以 `fwdbwd ph=f/ph=s` flow 为准，不依赖 `Fwd thread id`。

---

## 3.6 Path C：无 `stack_traces` 的正向 kernel

### 3.6.1 适用条件

```python
not kernel.get("args", {}).get("stack", {}).get("stack_traces", [])
and phase == "non-bwd"
```

### 3.6.2 处理步骤

```text
Step 1. kernel.correlation = C
Step 2. 找 cat=cuda_runtime, correlation=C -> launch_ev
        取 launch_ev.tid, launch_ev.ts
Step 3. 在 launch_ev.tid 上找覆盖 launch_ev.ts 的 cat=python_function，且 name 含 "nn.Module:"
        按 dur 升序排序，得到 innermost first
Step 4. 转为 FrameInfo 列表
        chains = FrameInfo 列表，phase = "non-bwd"
        如 Step 3 找不到，则 unmatched=True，chains=[]
```

### 3.6.3 Path C 主伪代码

```python
def build_entry_path_c(kernel: dict) -> KernelStackEntry:
    launch_ev = find_launch_event(kernel)
    if launch_ev is None:
        return unmatched_entry(kernel, path="C", phase="non-bwd")

    tid = launch_ev.get("tid")
    launch_ts = launch_ev.get("ts")
    if tid is None or launch_ts is None:
        return unmatched_entry(kernel, path="C", phase="non-bwd")

    module_chain = get_covering_nn_modules(tid, launch_ts)
    if not module_chain:
        return unmatched_entry(kernel, path="C", phase="non-bwd")

    return KernelStackEntry(
        name=kernel.get("name", ""),
        dur_us=float(kernel.get("dur", 0.0) or 0.0),
        phase="non-bwd",
        path="C",
        chains=[python_module_event_to_frame(ev) for ev in module_chain],
        unmatched=False,
    )
```

---

## 3.7 主函数伪代码

```python
def build_kernel_stack_cost_table(
    events: list[dict],
    source_files: dict[str, str],
    fwdbwd_index: dict,
) -> list[KernelStackEntry]:
    # source_files intentionally unused in Step 1a.

    runtime_by_corr = build_runtime_by_correlation(events)
    cpu_ops_by_tid = build_cpu_ops_by_tid(events)
    python_funcs_by_tid = build_python_funcs_by_tid(events)
    fwdbwd_f_by_tid_ts = build_fwdbwd_f_by_tid_ts(fwdbwd_index)

    rows: list[KernelStackEntry] = []

    for ev in events:
        if not is_cuda_kernel_event(ev):
            continue

        args = ev.get("args") or {}
        st = args.get("stack", {}) if isinstance(args, dict) else {}
        traces = st.get("stack_traces", []) if isinstance(st, dict) else []

        Step 0（所有路径共用）:
            launch = corr_to_launch[kernel.correlation]
            is_bwd = (launch.tid != main_thread_tid)
            phase = "bwd" if is_bwd else "non-bwd"

        if traces:
            rows.append(build_entry_path_a(ev))
        elif phase == "bwd":
            rows.append(build_entry_path_b(ev))
        else:
            rows.append(build_entry_path_c(ev))

    return rows
```

`is_cuda_kernel_event` 建议沿用现有 analyzer 中对 GPU kernel 事件的判断逻辑，通常可基于 `cat` / `ph` / `dur` / kernel event 字段判断，不在 Step 1a 中另起一套不兼容规则。

---

## 4. 路径判断流程图

```mermaid
flowchart TD
    A[遍历 events 中的 CUDA kernel] --> S0[Step 0: correlation -> launch\nlaunch.tid != main_thread_tid ?\nphase=bwd/non-bwd]
    S0 --> B{args["stack"]["stack_traces"]\n存在且非空?}
    B -- 是 --> PA[Path A\n解析 stack_traces\nphase 由 Step 0 确定]
    B -- 否 --> C{phase == "bwd"?}
    C -- 是 --> PB[Path B\n通过 backward cpu_op + fwdbwd flow\n跳回正向 nn.Module 链]
    C -- 否 --> PC[Path C\n在 launch CPU 线程上\n查找 nn.Module 覆盖链]

    PA --> OA[KernelStackEntry\npath=A\nchains=parsed frames\nunmatched=False]
    PB --> OB{找到正向 nn.Module 链?}
    PC --> OC{找到 nn.Module 链?}

    OB -- 是 --> OB1[KernelStackEntry\npath=B\nphase=bwd\nchains=module frames\nunmatched=False]
    OB -- 否 --> OB2[KernelStackEntry\npath=B\nphase=bwd\nchains=[]\nunmatched=True]

    OC -- 是 --> OC1[KernelStackEntry\npath=C\nphase=non-bwd\nchains=module frames\nunmatched=False]
    OC -- 否 --> OC2[KernelStackEntry\npath=C\nphase=non-bwd\nchains=[]\nunmatched=True]
```

ASCII 版本：

```text
CUDA kernel
  |
  |-- Step 0: correlation -> cuda_runtime launch
  |       |
  |       `-- launch.tid != main_thread_tid ? phase=bwd : phase=non-bwd
  |
  |-- has args["stack"]["stack_traces"] ?
  |       |
  |       `-- yes -> Path A -> parse stack_traces -> chains=frames
  |
  `-- no
          |
          |-- phase == "bwd" ?
          |       |
          |       `-- yes -> Path B
          |                 correlation -> cuda_runtime launch
          |                 -> bwd cpu_op covering chain
          |                 -> fwdbwd ph=f at cpu_op.ts
          |                 -> fwdbwd ph=s same id
          |                 -> forward tid/ts
          |                 -> covering nn.Module python_function chain
          |
          `-- no -> Path C
                    correlation -> cuda_runtime launch
                    -> launch tid/ts
                    -> covering nn.Module python_function chain
```

---

## 5. `fwdbwd_index` 预构建方式

### 5.1 输入事件格式

`fwdbwd` 事件结构示例：

```python
{
    "ph": "f" | "s",
    "id": 491,
    "pid": ...,
    "tid": ...,
    "ts": ...,
    "cat": "fwdbwd",
    "name": ...,
    "bp": ...,
    "uid": ...,
}
```

字段特点：

- `ph="f"`：backward 侧 flow 终点，`tid` 是 autograd engine C++ 线程，`ts` 与 backward `cpu_op.ts` 精确对齐。
- `ph="s"`：forward 侧 flow 起点，`tid` 是正向 CPU 线程，`ts` 是正向开始时间点。
- `id`：连接同一组 `s/f` flow 事件。
- 无 `args`，因此不能依赖 `args` 字段。

### 5.2 基础索引结构

```python
def build_fwdbwd_flow_index(events: list[dict]) -> dict:
    """构建完整 fwdbwd_index。"""
    return {
        "corr_to_launch":       {correlation: launch_event},
        "fwdbwd_f_by_tid_ts":   {(tid, ts): event},
        "fwdbwd_s_by_id":       {id: event},
        "tid_cpuop_sorted":     {tid: [events sorted by ts]},
        "tid_nn_module_sorted": {tid: [events sorted by ts]},
        "main_thread_tid":      int,
    }
```

构建要点：

- `corr_to_launch` 收集 `cat="cuda_runtime"` 且 `args.correlation` 存在的 launch 事件。
- `fwdbwd_f_by_tid_ts` 收集 `cat="fwdbwd", ph="f"` 事件，key 为 `(tid, ts)`。
- `fwdbwd_s_by_id` 收集 `cat="fwdbwd", ph="s"` 事件，key 为 `id`。
- `tid_cpuop_sorted` 收集 `cat="cpu_op"`，按 `ts` 排序。
- `tid_nn_module_sorted` 收集 `cat="python_function"` 且 `name` 含 `nn.Module`，按 `ts` 排序。
- `main_thread_tid` 取 `nn.Module` python_function 数量最多的 tid。

最终结构：

```python
fwdbwd_index = {
    "corr_to_launch":       {correlation: launch_event},        # O(1) lookup
    "fwdbwd_f_by_tid_ts":   {(tid, ts): event},                 # Path B Step 4
    "fwdbwd_s_by_id":       {id: event},                        # Path B Step 5
    "tid_cpuop_sorted":     {tid: [events sorted by ts]},        # 二分时间包含
    "tid_nn_module_sorted": {tid: [events sorted by ts]},        # 二分时间包含
    "main_thread_tid":      int,                                 # phase 判断用
}
```

### 5.3 Path B 优化索引

为了让 Path B Step 4 不需要反复扫描所有 flow id，建议从 `fwdbwd_index` 派生：

```python
def build_fwdbwd_f_by_tid_ts(fwdbwd_index: dict) -> dict[tuple[int, float], dict]:
    return fwdbwd_index.get("fwdbwd_f_by_tid_ts", {})
```

Path B Step 4 可使用：

```python
for cpu_op in cpu_chain:
    f_ev = fwdbwd_index["fwdbwd_f_by_tid_ts"].get((bwd_tid, cpu_op.get("ts")))
    if f_ev is not None:
        break
```

### 5.4 完整性建议

构建 `fwdbwd_index` 时不强制要求每个 id 同时具备 `s` 和 `f`。原因：

- trace 可能被截断。
- profiler event 可能不完整。
- Step 1a 应尽量容错，失败时输出 `unmatched=True`，而不是中断整个表构建。

但可以记录 debug 计数：

```text
fwdbwd total ids
fwdbwd complete s/f pairs
fwdbwd missing s
fwdbwd missing f
```

---

## 6. 测例更新方案：T-1a-3 / T-1a-4

原测例断言 `mod_name` 字段。Step 1a 新结构中不再输出单一 `mod_name`，而是输出统一的 `chains: list[FrameInfo]`。因此测例需要改为断言 `chains`。

### 6.1 T-1a-3：Path C，无 stack 正向 kernel

#### 场景

构造一个无 `stack_traces` 的正向 kernel：

1. kernel event 含 `correlation=C`。
2. 存在 `cat=cuda_runtime` 且 `args.correlation=C` 的 launch 事件。
3. launch 事件所在 `tid` 上存在覆盖 `launch_ts` 的 `python_function` 事件。
4. 该 `python_function.name` 含 `nn.Module:`。
5. backward `cpu_op` 覆盖链中不存在 `autograd::engine::evaluate_function:`，因此进入 Path C。

#### 新断言

```python
entry = find_entry_for_kernel(result, kernel_name)

assert entry.path == "C"
assert entry.phase == "non-bwd"
assert entry.unmatched is False
assert entry.chains
assert "<expected nn.Module name>" in entry.chains[0].func
```

可选增强断言：

```python
assert isinstance(entry.chains[0].file, str)
assert isinstance(entry.chains[0].line, int)
assert entry.chains[0].line > 0
```

#### 关键点

- 不再断言 `entry.mod_name`。
- `chains[0]` 必须是 innermost module。
- 如果 synthetic trace 中构造了多层 `nn.Module` 覆盖关系，应额外断言 `dur` 更短的 module 排在 `chains[0]`。

### 6.2 T-1a-4：Path B，无 stack 反向 kernel

#### 场景

构造一个无 `stack_traces` 的 backward kernel：

1. kernel event 含 `correlation=C`。
2. 存在 `cat=cuda_runtime` 且 `args.correlation=C` 的 launch 事件，位于 backward `bwd_tid`。
3. `bwd_tid` 上存在覆盖 `launch_ts` 的 `cpu_op` 链。
4. Step 0 判定 launch 线程不是 `main_thread_tid`，因此 `phase="bwd"`。
5. 覆盖链中某个 `cpu_op.ts` 与 `fwdbwd ph=f` 事件精确相等。
6. 同一 `id` 存在 `fwdbwd ph=s` 事件，给出正向 `tid` 与 `ts_fwd`。
7. 正向 `tid` 上存在覆盖 `ts_fwd` 的 `python_function`，且 name 含 `nn.Module:`。

#### 新断言

```python
entry = find_entry_for_kernel(result, kernel_name)

assert entry.path == "B"
assert entry.phase == "bwd"
assert entry.unmatched is False
assert entry.chains
assert "<expected nn.Module name>" in entry.chains[0].func
```

可选增强断言：

```python
assert len(entry.chains) >= 1
assert entry.chains[0].line == expected_inner_module_line
```

#### 关键点

- 断言 `phase="bwd"` 是 T-1a-4 必须覆盖的核心语义。
- 断言 `chains[0].func` 包含对应正向 `nn.Module` 名，验证 Path B 已从 backward kernel 成功跳回正向 module 层级。
- 如 synthetic trace 构造了多层正向 module，需验证 `chains` 为 innermost first。

---

## 7. 边界情况处理

### 7.1 kernel 缺少 correlation

#### 场景

无 `stack_traces` 的 kernel 缺少：

```python
kernel.args["correlation"]
```

#### 处理

- Path B/C 都无法找到 `cuda_runtime` launch 事件。
- 输出 unmatched entry。

```python
KernelStackEntry(
    name=kernel_name,
    dur_us=dur,
    phase="non-bwd" 或路径默认 phase,
    path="B" 或 "C",
    chains=[],
    unmatched=True,
)
```

建议：

- Path B 中缺 correlation：`phase="bwd"`，因为只有判断为 backward 后才会进入 Path B。
- Path C 中缺 correlation：`phase="non-bwd"`，因为 Path C 默认是 forward fallback。
- 如果连路径判断都依赖 correlation 且无法判断 backward，则走 Path C unmatched。

### 7.2 找不到 `cuda_runtime` launch 事件

#### 场景

kernel 有 correlation，但：

```python
runtime_by_corr[correlation]
```

为空。

#### 处理

- 不抛异常。
- 输出 `unmatched=True`。
- `chains=[]`。

### 7.3 多个 `cuda_runtime` 事件共享同一 correlation

#### 场景

理论上一个 kernel correlation 对应一个 launch 事件，但 trace 异常或特殊运行时可能出现多个候选。

#### 处理策略

建议优先级：

1. 优先 name 含 `cudaLaunchKernel` / `cuLaunchKernel` / `cudaGraphLaunch` 的事件。
2. 若仍有多个，选择 `ts` 最接近 kernel event 的候选。
3. 若无法判断，选择第一个候选，并记录 debug 信息。

### 7.4 Path B 找不到 backward `cpu_op` 覆盖链

#### 场景

launch event 找到了，但 backward tid 上无覆盖 `launch_ts` 的 `cpu_op`。

#### 处理

```python
path="B"
phase="bwd"
chains=[]
unmatched=True
```

### 7.5 Path B 找不到 `fwdbwd ph=f` 匹配

#### 场景

backward `cpu_op` 覆盖链存在，但没有任何 `cpu_op.ts` 与同 tid 的 `fwdbwd ph=f.ts` 精确相等。

#### 处理

```python
path="B"
phase="bwd"
chains=[]
unmatched=True
```

不建议在 Step 1a 中使用模糊时间窗口兜底。当前已验证 profiler 在该场景下 `ph=f.ts == cpu_op.ts`，保持精确匹配更可控。

### 7.6 Path B 找不到 `fwdbwd ph=s`

#### 场景

找到 `ph=f`，但同 `id` 的 `ph=s` 缺失。

#### 处理

```python
path="B"
phase="bwd"
chains=[]
unmatched=True
```

### 7.7 Path B/C 正向 `nn.Module` 链为空

#### 场景

已找到正向 `tid/ts`，但没有覆盖该时间点的 `python_function`，或事件名不含 `nn.Module:`。

#### 处理

```python
chains=[]
unmatched=True
```

说明：

- 这代表 Step 1a 未找到有效 nn.Module 覆盖。
- 不在 Step 1a 中回退到普通 `python_function` 或 `cpu_op` 名称，避免混淆后续 Step 1b 的模型结构匹配。

### 7.8 Path A stack trace 为空或无法解析

#### 场景

`stack_traces` 字段存在，但解析后没有任何合法 frame。

#### 处理建议

严格按路径判断仍属于 Path A，但需要标记 unmatched：

```python
path="A"
phase=Step 0 统一确定（"bwd" 或 "non-bwd"）
chains=[]
unmatched=True
```

如果 `stack_traces` 是空字符串或空列表，应视作不存在，继续进入 Path B/C 判断。

### 7.9 duration 缺失

#### 场景

kernel 缺少 `dur`。

#### 处理

```python
dur_us = float(kernel.get("dur", 0.0) or 0.0)
```

不要因单条 kernel 缺少 duration 中断整个构建。

### 7.10 event 字段类型异常

#### 场景

`args` 不是 dict，`tid` / `ts` 缺失，`line` 不是 int 等。

#### 处理

- 对 `args` 使用：`args = ev.get("args") or {}`。
- 对 line 做安全转换。
- 无法恢复的字段使用默认值，但保持数据结构完整。

---

## 8. 与 Step 1b 的接口约定

### 8.1 Step 1a 的职责边界

Step 1a 只负责构建 kernel 级调用链成本表：

```text
kernel event -> KernelStackEntry
```

它不负责：

1. frame 过滤。
2. 用户模型源码结构匹配。
3. framework / runtime frame 删除。
4. module name 规范化到 DAG node id。
5. 多 kernel 聚合。
6. source line evidence 精修。

### 8.2 Step 1b 可依赖的输入

Step 1b 可以依赖 `KernelStackEntry` 的以下稳定语义：

| 字段 | Step 1b 可依赖语义 |
|---|---|
| `name` | 原始 kernel name。 |
| `dur_us` | kernel 耗时，单位 us。 |
| `phase` | `fwd` / `bwd` / `unknown`。Path B 必须是 `bwd`，Path C 必须是 `fwd`。 |
| `path` | 调用链来源路径：`A` / `B` / `C`。 |
| `chains` | innermost first 的 `FrameInfo` 列表。Step 1b 可从内到外匹配最具体的用户模型 frame。 |
| `unmatched` | 若为 `True`，Step 1b 不应强行归属到某个 nn.Module；可进入 unknown bucket 或 debug bucket。 |

### 8.3 `source_files` 的延迟使用

`source_files` 在 Step 1a 中保留但不使用。Step 1b 可使用它做：

1. `FrameInfo.file` 与模型源码文件集合的匹配。
2. framework frame 与用户 frame 区分。
3. class / method / line range 的结构化匹配。
4. 将 Path A 的原始 stack frame 映射到用户模型 module。
5. 将 Path B/C 的 `nn.Module` frame 进一步映射到 DAG node。

### 8.4 unmatched 的后续处理约定

当 Step 1a 输出：

```python
entry.unmatched is True
entry.chains == []
```

Step 1b 应：

1. 保留该 kernel 的成本，不丢弃。
2. 不伪造 module 归属。
3. 将其归入 unknown / unmatched bucket。
4. 在 debug summary 中统计 unmatched kernel 数量和耗时占比。

### 8.5 frame 顺序约定

Step 1b 必须假设：

```python
entry.chains[0]
```

是最内层 frame。匹配时建议从 `chains[0]` 向外扫描，优先匹配最具体的用户模型 frame。

### 8.6 path 语义约定

| path | chains 来源 | 典型用途 |
|---|---|---|
| `A` | kernel 自带 `stack_traces` | 增强 trace 下直接从 kernel stack 映射源码热点。 |
| `B` | backward kernel 通过 `fwdbwd` flow 跳回正向 `nn.Module` 链 | 无 stack trace 的 backward kernel 归属恢复。 |
| `C` | forward kernel launch 线程上的正向 `nn.Module` 覆盖链 | 无 stack trace 的 forward kernel 归属恢复。 |

---

## 9. 实现注意事项

### 9.1 不做 frame 过滤

Step 1a 必须保留 Path A 中的所有 frames，包括 framework 级调用。过滤推迟到 Step 1b。

### 9.2 不使用 `source_files`

Step 1a 中不要因为传入了 `source_files` 就提前筛选用户文件。这样可以避免 Step 1a 和 Step 1b 职责耦合。

### 9.3 输出顺序稳定

主循环必须按 `events` 原始顺序遍历 kernel，并按遍历顺序 append 到结果列表。

### 9.4 不引入正则源码解析

Step 1a 允许用简单 parser 解析 profiler `stack_traces` 文本格式，但不得引入针对用户源码的正则解析逻辑。源码结构解析属于 analyzer 既有 AST-first 路径或 Step 1b 之后的职责。

### 9.5 Debug 统计建议

建议在实现中保留可选 debug counters：

```text
total kernels
path A matched / unmatched
path B matched / unmatched
path C matched / unmatched
missing correlation
missing cuda_runtime launch
missing fwdbwd f
missing fwdbwd s
empty nn.Module chain
```

这些统计不属于 `KernelStackEntry` 输出结构，但对后续验证和回归定位很有帮助。

---

## 10. 验收标准

Step 1a 完成后应满足：

1. 输出条数等于输入 `events` 中 CUDA kernel 事件数量。
2. 输出顺序与输入 kernel 出现顺序一致。
3. Path A：有非空 `stack_traces` 的 kernel 进入 Path A，并输出 `chains`。
4. Path B：无 `stack_traces` 的 backward kernel 能通过 `fwdbwd ph=f/ph=s` 找回正向 `nn.Module` 链。
5. Path C：无 `stack_traces` 的正向 kernel 能在 launch CPU 线程上找到正向 `nn.Module` 链。
6. T-1a-3 改为断言 `chains[0].func` 包含对应 `nn.Module` 名，且 `unmatched=False`。
7. T-1a-4 改为断言 `chains[0].func` 包含对应 `nn.Module` 名，`phase="bwd"`，且 `unmatched=False`。
8. correlation、fwdbwd、nn.Module 链缺失时不抛异常，输出 `unmatched=True`。
9. Step 1a 不做 frame 过滤，不使用 `source_files` 做提前筛选。
