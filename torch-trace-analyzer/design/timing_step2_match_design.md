# Timing Step 2：Kernel Chain → Instance 候选匹配设计

## 1. Overview

Timing 4 步 Pipeline 的目标是把 runtime trace 中的 kernel 开销，稳定、可解释地归属到静态 DAG 的具体 `InstanceKey` 上。

Step 2 位于 Pipeline 中间层：

1. **Step 1：构建 KernelChain**
   - 输入：trace kernel event + stack frames。
   - 输出：每条 kernel 对应一个 `KernelChain`。
   - 本轮需要补充：从 trace 事件名中提取 ModuleList 实例后缀 `_N`，写入 `KernelChain.instance_suffix`。
2. **Step 2：KernelChain → Instance 候选匹配**
   - 输入：Step 1 输出的 `KernelChain`、静态 DAG、预构建源码位置倒排索引 `loc_index`。
   - 输出：每条 kernel chain 对应的归属结果：具体 `InstanceKey`，或父 group，或 `__unattributed__`。
3. **Step 3：kernel attribution table 聚合**
   - 输入：Step 2 的逐 kernel 归属结果。
   - 输出：按 `InstanceKey` 聚合的 forward/backward/kernel 耗时表。
4. **Step 4：instance timing rollup + panel data**
   - 输入：Step 3 聚合表 + DAG 层级结构。
   - 输出：前端 timing 面板所需的实例级与 group 级 timing 数据。

Step 2 的核心要求：

- 以 runtime stack frame 的源码位置为主线，匹配静态 DAG 中的 module 实例。
- ModuleList / Sequential 等列表容器场景中，不扩展 `InstanceKey` 结构，而是使用 `attr_name` 和 trace 事件名里的 `_N` 后缀进行实例消歧。
- runtime trace 是可信输入。如果 trace 明确带有 `_N`，但静态 DAG 找不到对应 `blocks[N]`，说明 DAG 解析存在缺口；此时应 **LOG ERROR + 归父 group**，保证时间数据完整性，同时用 error 暴露 DAG 缺口。

---

## 2. 数据结构

### 2.1 InstanceKey（已有，不扩展）

```python
@dataclass
class InstanceKey:
    class_name: str
    instance_suffix: str  # e.g. "_3" for blocks[3], "" for non-list
```

字段说明：

- `class_name`：静态 DAG 中的模块类名，例如 `FSDPBlock`、`DenseTower`。
- `instance_suffix`：同类多实例的后缀。
  - ModuleList / Sequential 展开实例：`"_3"` 表示第 3 个实例。
  - 非列表实例或父 group：`""`。

本设计不扩展 `InstanceKey`。列表容器不同实例的区分，依赖静态 DAG 中的 `attr_name` 语义，例如 `blocks[3]`。

### 2.2 KernelChain（Step 1 输出）

```python
@dataclass
class KernelChain:
    kernel_name: str
    duration_us: float
    frames: list[StackFrame]
    instance_suffix: str
```

字段说明：

- `kernel_name`：kernel 名称。
- `duration_us`：kernel duration，单位微秒。
- `frames`：kernel 对应的 Python stack frames，保持从外层到内层，或在实现中明确统一顺序。
- `instance_suffix`：Step 1 从 trace 事件名中提取的实例后缀。
  - 例如 trace 事件名包含 `FSDPBlock_3`，提取为 `"_3"`。
  - 无后缀时为 `""`。

### 2.3 loc_index（Step 2 倒排索引）

```python
loc_index: dict[tuple[str, int], list[tuple[str, str]]]
# key: (basename(file), line)
# value: [(class_name, attr_name), ...]
```

字段说明：

- key：源码位置。
  - `basename(file)`：文件 basename，避免 trace 中绝对路径和本地解压路径不一致导致无法匹配。
  - `line`：源码行号。
- value：该源码位置对应的静态 DAG 候选列表。
  - `class_name`：候选模块类名。
  - `attr_name`：候选模块在父模块中的属性路径，例如：
    - `dense_tower`
    - `blocks`
    - `blocks[3]`
    - `encoder.layers[2]`

`loc_index` 是 Step 2 的预计算结构，用于把逐 kernel 的匹配从扫描全 DAG，优化为按源码位置 O(1) 查候选。

---

## 3. 倒排索引构建：build_loc_index(dag)

### 3.1 输入

- 静态 DAG，包含每个 module/group/instance 的源码映射信息。
- 对每个候选节点，至少需要可获得：
  - `class_name`
  - `attr_name`
  - `source file`
  - `line`

### 3.2 输出

```python
loc_index: dict[tuple[str, int], list[tuple[str, str]]]
```

### 3.3 算法

伪代码：

```python
def build_loc_index(dag):
    loc_index = defaultdict(list)

    for class_node in dag.classes:               # C
        for attr in class_node.attrs:            # A per class
            loc = attr.call_loc or attr.forward_loc or attr.source_loc
            if not loc:
                continue

            file_base = basename(loc.file)
            line = loc.line
            key = (file_base, line)

            loc_index[key].append((class_node.class_name, attr.attr_name))

    return dict(loc_index)
```

### 3.4 构建规则

1. **索引源码调用点优先**
   - 优先使用 forward 调用点，例如 `self.blocks[i](x)` 所在行。
   - 如果没有明确调用点，再考虑静态节点声明或可解释的 fallback source location。
2. **同一源码行允许多个候选**
   - 例如一行里可能包含多个 module call，或父 group 与子 module 都映射到同一行。
   - 因此 value 必须是 list，而不是单个候选。
3. **basename(file) 归一化**
   - trace stack frame 的路径可能来自 runtime 环境，静态 DAG 的路径可能来自本地解压目录。
   - 使用 basename 可降低路径前缀差异带来的误判。

### 3.5 复杂度

设：

- `C`：DAG 中 class/group 数量。
- `A`：每个 class/group 的平均 attr 候选数量。

则索引构建复杂度为：

```text
O(C · A)
```

该成本只在 Step 2 开始前支付一次。

---

## 4. 匹配算法：match_kernel_to_instance(chain, loc_index, dag)

### 4.1 输入

- `chain: KernelChain`
- `loc_index: dict[tuple[str, int], list[tuple[str, str]]]`
- `dag`：静态 DAG，需支持：
  - 根据 `(class_name, attr_name)` 查询父 group 的 `InstanceKey`。
  - 根据 `(class_name, attr_name, instance_suffix)` 查询精确列表实例的 `InstanceKey`。
  - 判断 `blocks[N]` / `attr_name[N]` 是否存在。

### 4.2 输出

```python
@dataclass
class MatchResult:
    instance_key: InstanceKey | str
    reason: str
    matched_frame: StackFrame | None
    candidates: list[tuple[str, str]]
```

其中：

- `instance_key`：
  - 精确命中时返回具体 `InstanceKey`。
  - `_N` 存在但静态节点缺失时，返回父 group 的 `InstanceKey`。
  - 完全未命中时返回 `"__unattributed__"`。
- `reason`：匹配原因，方便日志和测试断言。
- `matched_frame`：最终采用的最深命中帧。
- `candidates`：该帧对应的候选列表。

### 4.3 核心匹配表

| 场景 | 处理 |
|---|---|
| 事件名没有 `_N` 后缀 | 正常匹配，命中父 group 就归父 group |
| 有 `_N`，找到对应 `blocks[N]` | 精确归属 |
| 有 `_N`，但 `blocks[N]` 不存在 | LOG ERROR + 归父 group |

### 4.4 算法步骤

1. 遍历 `chain.frames`，用 `(basename(frame.file), frame.line)` 查询 `loc_index`。
2. 找到所有命中帧后，取最深，也就是最内层命中的 frame。
3. 对最深 frame 的候选列表进行消歧：
   - 如果 `chain.instance_suffix == ""`：
     - 使用候选对应的父 group / 非列表实例 `InstanceKey`。
     - 如果候选本身已经是具体实例，也可直接返回该实例。
   - 如果 `chain.instance_suffix != ""`：
     - 解析 `N`，例如 `"_3" -> 3`。
     - 对候选 `attr_name` 查找对应的 `attr_name[N]`。
     - 找到则返回精确实例 `InstanceKey`。
     - 找不到则 LOG ERROR，并返回父 group `InstanceKey`。
4. 如果没有任何 frame 命中 `loc_index`，返回 `__unattributed__`。

### 4.5 伪代码

```python
def match_kernel_to_instance(chain, loc_index, dag):
    matched = []

    for depth, frame in enumerate(chain.frames):
        key = (basename(frame.file), frame.line)
        candidates = loc_index.get(key)
        if not candidates:
            continue
        matched.append((depth, frame, candidates))

    if not matched:
        return MatchResult(
            instance_key="__unattributed__",
            reason="no_stack_frame_hit_loc_index",
            matched_frame=None,
            candidates=[],
        )

    # 取最深命中帧。要求 frames 顺序在 Step 1 或本函数内明确标准化。
    depth, frame, candidates = max(matched, key=lambda item: item[0])

    # 候选列表内优先选择可解析到 InstanceKey 的候选。
    for class_name, attr_name in candidates:
        parent_key = dag.lookup_parent_instance_key(class_name, attr_name)

        if chain.instance_suffix == "":
            instance_key = dag.lookup_instance_key(class_name, attr_name)
            if instance_key is None:
                instance_key = parent_key

            return MatchResult(
                instance_key=instance_key,
                reason="matched_without_instance_suffix",
                matched_frame=frame,
                candidates=candidates,
            )

        index = parse_instance_suffix(chain.instance_suffix)  # "_3" -> 3
        child_attr_name = make_indexed_attr_name(attr_name, index)  # blocks + 3 -> blocks[3]
        child_key = dag.lookup_instance_key(class_name, child_attr_name)

        if child_key is not None:
            return MatchResult(
                instance_key=child_key,
                reason="matched_with_instance_suffix",
                matched_frame=frame,
                candidates=candidates,
            )

        LOG.error(
            "ModuleList index mismatch: kernel=%s suffix=%s class=%s attr=%s indexed_attr=%s frame=%s:%s; "
            "runtime trace has this instance but static DAG does not. Attribute cost to parent group.",
            chain.kernel_name,
            chain.instance_suffix,
            class_name,
            attr_name,
            child_attr_name,
            basename(frame.file),
            frame.line,
        )

        return MatchResult(
            instance_key=parent_key,
            reason="instance_suffix_index_missing_fallback_to_parent_group",
            matched_frame=frame,
            candidates=candidates,
        )

    return MatchResult(
        instance_key="__unattributed__",
        reason="matched_frame_but_no_resolvable_candidate",
        matched_frame=frame,
        candidates=candidates,
    )
```

### 4.6 边界处理

#### B1. chain 里没有任何帧命中 loc_index

处理：

- 返回 `__unattributed__`。
- 原因通常是纯 PyTorch 内部 kernel，或 stack trace 缺少用户源码帧。

#### B2. 多帧命中 loc_index

处理：

- 取最深帧。
- 目的：避免父 group 和子 module 同时命中时，把同一 kernel 重复计给父 group。

#### B3. 有 `_N` 且找到对应 `blocks[N]`

处理：

- 返回精确 `InstanceKey(class_name, "_N")`。
- 这是 ModuleList 场景的主路径。

#### B4. 有 `_N` 但 `blocks[N]` 不存在

处理：

- LOG ERROR，错误类型应明确标记为 DAG 解析问题。
- 不归 `__unattributed__`。
- 回退归父 group，保证真实 runtime 开销不会丢失。

原因：

- runtime trace 是可信的，`FSDPBlock_99` 这类信息说明 runtime 中确实存在对应实例。
- 静态 DAG 找不到 `blocks[99]`，说明源码解析、容器展开或实例裁剪存在缺口。
- 归父 group 可以保持总耗时完整；ERROR 日志用于暴露 DAG 缺口并驱动后续修复。

#### B5. 事件名没有 `_N` 后缀

处理：

- 正常匹配当前命中候选。
- 如果命中位置是父 group，就归父 group。
- 不做 `_N` 推断，不从 annotation 或其他弱信号猜测实例编号。

---

## 5. 时间复杂度分析

### 5.1 参数定义

- `K`：kernel chain 数量。
- `F`：每条 kernel chain 的平均 stack frame 数量。
- `C`：静态 DAG 中 class/group 数量。
- `A`：每个 class/group 的平均 attr 候选数量。
- `I`：单个源码位置命中的平均候选数量，即 `loc_index[(file, line)]` 的平均 list 长度。

典型情况下：

- `K` 很大，可能是几千到几十万。
- `F` 通常是几十级以内。
- `C` 和 `A` 与模型结构规模相关。
- `I` 通常很小，大多数源码行只对应 1～3 个候选。

### 5.2 朴素匹配复杂度

如果每条 kernel 的每个 frame 都扫描完整 DAG：

```text
O(K · F · (C + A))
```

或在实现上近似为：

```text
O(K · F · C · A)
```

问题：

- `K` 是最大量级项。
- 对每个 kernel 反复扫描 DAG 会把静态结构成本放大到 runtime kernel 数量级。

### 5.3 倒排索引优化后复杂度

预构建：

```text
O(C · A)
```

逐 kernel 匹配：

```text
O(K · F · I)
```

由于 `I` 通常远小于 `C · A`，优化后可以把 Step 2 的主要成本控制在 stack frame 遍历和少量候选消歧上。

总体复杂度：

```text
O(C · A + K · F · I)
```

---

## 6. 测试设计

本节只描述 mock 数据与断言，不写实现代码。

### 6.1 测试公共 mock 结构

#### mock DAG

构造一个最小 DAG：

```text
Root
└── Encoder(parent group)
    ├── blocks[0]: FSDPBlock, InstanceKey("FSDPBlock", "_0")
    ├── blocks[1]: FSDPBlock, InstanceKey("FSDPBlock", "_1")
    ├── blocks[2]: FSDPBlock, InstanceKey("FSDPBlock", "_2")
    ├── blocks[3]: FSDPBlock, InstanceKey("FSDPBlock", "_3")
    └── dense: DenseTower, InstanceKey("DenseTower", "")
```

父 group：

```python
InstanceKey("Encoder", "")
```

#### mock loc_index

```python
loc_index = {
    ("model.py", 100): [("Encoder", "blocks")],
    ("model.py", 120): [("FSDPBlock", "blocks[3]")],
    ("model.py", 130): [("DenseTower", "dense")],
    ("model.py", 200): [("Encoder", "encoder")],
}
```

实际测试可按 case 单独裁剪，避免相互干扰。

---

### T1 正常匹配：单帧命中，归属到精确 InstanceKey

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="aten::linear",
    duration_us=100.0,
    frames=[StackFrame(file="/runtime/model.py", line=130)],
    instance_suffix="",
)
```

loc_index：

```python
("model.py", 130) -> [("DenseTower", "dense")]
```

#### 断言

- 返回 `InstanceKey("DenseTower", "")`。
- `reason == "matched_without_instance_suffix"`。
- 不产生日志 ERROR。

---

### T2 ModuleList `_N` 精确消歧：`FSDPBlock_3` → `blocks[3]`

#### 场景

源码中存在：

```python
for block in self.blocks:
    x = block(x)
```

trace 事件名中出现 `FSDPBlock_3`，Step 1 提取：

```python
instance_suffix="_3"
```

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="triton_flash_attn",
    duration_us=250.0,
    frames=[StackFrame(file="/runtime/model.py", line=100)],
    instance_suffix="_3",
)
```

loc_index：

```python
("model.py", 100) -> [("Encoder", "blocks")]
```

mock DAG 中存在：

```python
blocks[3] -> InstanceKey("FSDPBlock", "_3")
```

#### 断言

- 返回 `InstanceKey("FSDPBlock", "_3")`。
- `reason == "matched_with_instance_suffix"`。
- 不归父 group。
- 不产生日志 ERROR。

---

### T3 多帧命中取最深：父子两级 frame 都命中，归到子级

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="aten::matmul",
    duration_us=180.0,
    frames=[
        StackFrame(file="/runtime/model.py", line=200),  # 父 group Encoder
        StackFrame(file="/runtime/model.py", line=130),  # 子模块 DenseTower
    ],
    instance_suffix="",
)
```

loc_index：

```python
("model.py", 200) -> [("Encoder", "encoder")]
("model.py", 130) -> [("DenseTower", "dense")]
```

#### 断言

- 返回 `InstanceKey("DenseTower", "")`。
- `matched_frame.line == 130`。
- 父 group `InstanceKey("Encoder", "")` 不累计该 kernel 的独立归属。
- 不产生日志 ERROR。

---

### T4 `_N` index 对不上：LOG ERROR + 归父 group

#### 场景

runtime trace 事件名包含 `FSDPBlock_99`，Step 1 提取：

```python
instance_suffix="_99"
```

但静态 DAG 只解析出了 `blocks[0]` ～ `blocks[3]`。

这说明 runtime 中存在该实例，但静态 DAG 没有对应节点，属于 DAG 解析缺口。

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="triton_layernorm",
    duration_us=300.0,
    frames=[StackFrame(file="/runtime/model.py", line=100)],
    instance_suffix="_99",
)
```

loc_index：

```python
("model.py", 100) -> [("Encoder", "blocks")]
```

mock DAG：

```text
blocks[0], blocks[1], blocks[2], blocks[3] 存在
blocks[99] 不存在
父 group Encoder 存在：InstanceKey("Encoder", "")
```

#### 断言

- 产生日志 ERROR，内容包含：
  - `ModuleList index mismatch`
  - `suffix=_99`
  - `indexed_attr=blocks[99]`
  - `runtime trace has this instance but static DAG does not`
- 返回父 group：`InstanceKey("Encoder", "")`。
- `reason == "instance_suffix_index_missing_fallback_to_parent_group"`。
- 不返回 `__unattributed__`。

---

### T5 完全未命中：纯 PyTorch 内部 kernel，归 `__unattributed__`

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="aten::empty",
    duration_us=5.0,
    frames=[
        StackFrame(file="/usr/local/lib/python/site-packages/torch/nn/functional.py", line=1234),
        StackFrame(file="/usr/local/lib/python/site-packages/torch/autograd/function.py", line=567),
    ],
    instance_suffix="",
)
```

loc_index 中没有任何对应 key。

#### 断言

- 返回 `"__unattributed__"`。
- `reason == "no_stack_frame_hit_loc_index"`。
- `matched_frame is None`。
- 不产生日志 ERROR。

---

### T6 父 group 归属：最深帧停在父 module 调用行，归父 group

#### 场景

stack chain 中最深的用户源码 frame 只到父 module 调用行，没有进入更细子模块源码行。

这类情况不应强行猜测子实例，应归父 group。

#### mock 数据

KernelChain：

```python
KernelChain(
    kernel_name="aten::native_layer_norm_backward",
    duration_us=220.0,
    frames=[StackFrame(file="/runtime/model.py", line=200)],
    instance_suffix="",
)
```

loc_index：

```python
("model.py", 200) -> [("Encoder", "encoder")]
```

mock DAG：

```python
("Encoder", "encoder") -> InstanceKey("Encoder", "")
```

#### 断言

- 返回 `InstanceKey("Encoder", "")`。
- `reason == "matched_without_instance_suffix"`。
- 不尝试 `_N` 消歧。
- 不产生日志 ERROR。

---

## 7. 实现设计

### 7.1 Step 1：补充 instance_suffix 提取

在 Step 1 构建 `KernelChain` 时，从 trace 事件名或 module event name 中提取 `_N`：

```python
def extract_instance_suffix(event_name: str) -> str:
    # 仅识别明确的末尾 _N，例如 FSDPBlock_3 -> _3
    # 不做模糊猜测，不从源码 annotation 推断。
    ...
```

要求：

- 只接受明确数字后缀。
- 无后缀返回 `""`。
- 提取结果写入 `KernelChain.instance_suffix`。

### 7.2 Step 2a：构建 loc_index

新增或改造：

```python
def build_loc_index(dag) -> dict[tuple[str, int], list[tuple[str, str]]]:
    ...
```

要求：

- 只构建一次。
- 使用 `basename(file)` + `line` 作为 key。
- value 保留 list，以支持同一行多个候选。

### 7.3 Step 2b：逐 kernel chain 匹配

新增或改造：

```python
def match_kernel_to_instance(chain, loc_index, dag) -> MatchResult:
    ...
```

要求：

- 多帧命中取最深。
- `_N` 非空时，只做精确 index 匹配。
- `_N` 对不上时，LOG ERROR + 归父 group。
- 完全未命中时，归 `__unattributed__`。

### 7.4 日志要求

`_N` 对不上时必须使用 ERROR 级别日志，且信息足够定位 DAG 缺口：

```text
ModuleList index mismatch: kernel=<kernel_name> suffix=<_N> class=<class_name> attr=<attr_name> indexed_attr=<attr_name[N]> frame=<file>:<line>; runtime trace has this instance but static DAG does not. Attribute cost to parent group.
```

日志语义：

- 这是 DAG 解析问题，不是 runtime trace 问题。
- ERROR 用于在测试和人工审查中暴露缺口。
- 归父 group 用于保证 timing 数据完整。

### 7.5 与 Step 3 的接口

Step 2 输出的 `MatchResult.instance_key` 进入 Step 3 聚合：

- 精确实例：按该 `InstanceKey` 聚合。
- 父 group fallback：按父 group `InstanceKey` 聚合。
- `__unattributed__`：进入 unattributed bucket，不计入具体 DAG 节点。

---

## 8. 验收标准

1. Step 2 文档中的 6 个测试场景均可落地为单元测试。
2. T4 明确验证：`_N` 存在但静态 index 缺失时，必须 LOG ERROR + 归父 group，不得归 `__unattributed__`。
3. 匹配算法不扩展 `InstanceKey`。
4. 无 `_N` 时不做实例编号猜测，命中父 group 即归父 group。
5. 多帧命中只取最深帧，避免父子重复计时。
6. 复杂度从朴素扫描优化为：

```text
O(C · A + K · F · I)
```
