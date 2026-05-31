<callout icon="bulb" bgc="5">  
  本文先整理 2026-05-29 16:00-18:00 左右已经讨论过的 **ast-timing 联合修复方案**，当前阶段只做方案沉淀，不做代码修改。工作分支已从最新 `origin/master` 拉起为 `feat/ast-timing-fix`。内部 storage 分支因现有未提交改动阻塞切换，已停止对 storage 的分支操作，未覆盖已有工作。  
</callout>

<callout icon="warning" bgc="2">  
  **2026-05-29 23:49 NaN 补充（核心概念修正）**：方案草案在 group 语义上存在偏差，需在整个方案中贯穿以下核心定义再继续讨论：  
  
  **Group 的语义单元 = `(attr, callsite)` 的组合，即"一个 attr 在一个地方调用了一次"。**
  
  - torch 的 Module call（即 DAG 的 group）本质上不是构造点，也不是调用点，而是"调用 × 构造"产生的组合。
  - 同一个 attr 可能被多次调用（for 循环），每次调用是独立 group。
  - 在 for 循环中可能出现：同一行匹配多个对象（一行多 group），或同一行匹配一个对象；  
    但也可能出现**多行匹配同一个 group**（一个 attr 的一次调用跨多行时，这些行都应归到同一 group）。
  - 因此 loc 不是 group 的主键，`(attr_identity, call_occurrence_index)` 才是 group 的稳定主键。  
    loc 是匹配的线索，不是 group 的唯一身份标识。
</callout>

## 目标与问题边界

本轮联合修复聚焦两类 timing 归因问题：

1. **同一个实例或同一类实例在多个 loc 被调用时**，不能只根据 class name 或粗粒度 stack frame 做归因，需要从源码 loc 精确定位到父 Module 中的 attr name，再由 attr name 解析到 child instance。
2. **kernel 时间归档时**，不能只看调用栈的某一层，也不能把未命中的内部 tensor op 拆成行级弱分组；需要从 kernel 的调用栈自底向上选择最深有效命中，并把时间归到正确的 Module 实例或 self cost。

这两个问题本质上都要求：**AST 静态结构与 timing stack 匹配共用一套稳定的实例语义**。

---

## 核心语义定义（2026-05-29 NaN 确认）

### Group 的语义单元

在讨论"loc 匹配"和"时间归档"之前，必须先对 group 的概念边界形成共识：

**一个 group = 一个 attr 在一个地方的一次调用（`attr × callsite × occurrence`）。**

这与直觉上的"构造点"或"调用点"都不同：

| 概念 | 含义 | 是否等于 group |
|---|---|---|
| 构造点 `self.a = Block()` | attr 被赋值 | ❌ 一个 attr 可以只构造一次但被调用多次 |
| 调用点 loc `self.a(x)` | 某行源码 | ❌ 同一行可能对应多个 group（for 循环里 `blocks[i](x)`） |
| **attr × callsite × occurrence** | 一个 attr 在一处被调用了一次 | ✅ group 的真实语义 |

**三种 loc 与 group 的映射关系（均合法）：**

```
一行 → 多 group：for i in range(N): self.blocks[i](x)  # 同一行调用了 N 个不同实例
一行 → 一 group：x = self.a(x)                          # 常规情况
多行 → 一 group：y = self.a(                             # 一次调用跨多行
                     x, mask
                 )
```

**对 loc index 设计的影响：**

- loc 只能作为匹配线索（"这个 kernel 的栈帧落在哪行"），不能作为 group 的主键。
- group 的主键必须是 `(attr_identity, call_occurrence_index)`，其中：
  - `attr_identity` = `(parent_class, attr_name)` 在 static tree 中的唯一标识
  - `call_occurrence_index` = 运行时该 attr 被第几次调用（来自 trace timeline 顺序，或 `_N` 后缀）
- loc 匹配到 `(parent_class, attr_name)` 后，还需要结合 trace 事件时序确定是"第几次调用"，才能最终确定归属哪个 group。

---

## 已确认的核心方向

### 1. loc 的第一目标是 `parent_class.attr_name`，不是 child class

已讨论的关键结论是：源码 loc 不能直接解析成 child class。正确顺序应为：

1. 用 `(basename(file), line)` 查 loc index。
2. 得到一个或多个候选：`(parent_class, attr_name)`。
3. 再回到 `static_module_tree[parent_class]`，通过 `attrs[attr_name]` 解析 child class / child instance。

原因是同一个 child class 可以在父类里出现多次，例如：

```python
class Parent(nn.Module):
    def __init__(self):
        self.a = Block()
        self.b = Block()

    def forward(self, x):
        x = self.a(x)   # loc A
        x = self.b(x)   # loc B
        return x
```

如果 loc 直接定位到 `Block`，就无法区分 `a` 和 `b`。但如果 loc 先定位到 `Parent.a` / `Parent.b`，再解析 child class，就能保留实例身份。

<callout icon="thought_balloon" bgc="3">  
  **归因语义建议：** loc index 的 value 不应是单值，也不应是 child class，而应是候选列表：`loc_index[(file, line)] -> [(parent_class, attr_name), ...]`。同一行存在多个 Module 调用时，由后续消歧逻辑处理。  
</callout>

### 2. loc index 应基于 `static_module_tree` 构建，不应基于 `class_map`

已有问题来自 timing 流程把 `class_map` 传给 `build_loc_index()`。`class_map` 只有类范围和方法范围，缺少 loc 到 attr 所需的结构信息，例如：

- `attrs`
- `first_call_loc`
- `attr_def_loc`
- `dep_edge_locs`

因此它容易触发 `<self>` fallback，把 forward 内许多源码行注册成 `(Class, <self>)`，导致一个真实 Module 被拆成大量行级 timing group。

正确方向是：

```python
loc_index = build_loc_index(static_module_tree)
```

`static_module_tree` 中保留了父 Module、child attr、attr 定义点、forward 首次调用点、依赖边 loc 等信息，更适合作为 timing loc index 的输入。

### 3. `<self>` fallback 只能作为极窄兜底

`<self>` fallback 不是完全禁止，但必须限制范围：

- 当某个 class 完全没有可用的 `attrs / first_call_loc / attr_def_loc / dep_edge_locs` 时，才允许注册 `<self>`。
- 如果 class 已经有 child attr 结构，则不能把 forward 中每一行都注册为 `<self>`。

否则 timing 会出现大量弱归因分组，且同一个真实实例被拆散。

---

## kernel 调用栈自底向上匹配方案

### 1. 遍历全栈，选择最深命中 frame

已讨论的 Step1 stack frames 顺序是 **outer → inner**。因此 matching 时可以遍历全栈，收集所有命中 loc index 的 frame，然后选择 depth 最大的命中。

伪代码：

```python
matched = []
for depth, frame in enumerate(chain.frames or []):
    key = (basename(frame.file), frame.line)
    candidates = loc_index.get(key)
    if candidates:
        matched.append((depth, frame, candidates))

if not matched:
    return "__unattributed__"

# frames 是 outer -> inner，depth 最大代表最内层命中
best_depth, best_frame, candidates = max(matched, key=lambda item: item[0])
```

这里的“自底向上”有两层含义：

1. **调用栈匹配层面：** 优先使用最内层、最接近 kernel 的有效 loc 命中。
2. **timing 汇总层面：** child timing 自底向上 rollup 到 parent，同时 parent 保留自身 self cost。

### 2. 内部 tensor op 未命中时，应冒泡到外层 Module 调用点

典型场景：

```text
Outer.forward:L10  -> self.inner(x)
Inner.forward:L50  -> torch.cat(...)
```

如果 `Inner.forward:L50` 是纯 tensor op，不在 loc index 中，而 `Outer.forward:L10` 命中 `(Outer, inner)`，则该 kernel 不应归为 `__unattributed__`，也不应生成 `Inner.forward:L50` 行级弱分组。

正确行为：

1. 内层 `torch.cat` 行没有命中 loc index。
2. 继续向外找到 `Outer.forward:L10`。
3. `Outer.forward:L10` 命中 `(Outer, inner)`。
4. 回查 `static_module_tree[Outer].attrs[inner]`，得到 child class `Inner`。
5. kernel 时间归入 `Inner` 这个实例整体，作为 `Inner` 的 self cost。
6. 后续 rollup 时，`Inner` 的 inclusive cost 汇总到 `Outer`。

<callout icon="bulb" bgc="4">  
  这不是退化，而是期望语义：纯 tensor op 属于当前 Module 实例内部实现，应计入该实例 self cost，而不是制造行级虚拟 Module。  
</callout>

### 3. 全栈无命中才进入 `__unattributed__`

如果 kernel 的所有 stack frame 都无法命中 loc index，才归入 `__unattributed__`。不要为了提高覆盖率强行猜测 Module，否则会制造错误归因。

---

## 实例 key 与 Step2/Step3 接口统一

### 1. Step2 输出必须统一为四元组 key

当前需要避免混用 dataclass `InstanceKey` 与 tuple key。Step3 rollup 需要稳定解包：

```python
class_name, callsite_file, callsite_line, ancestors = key
```

因此 Step2 的可归因输出建议统一为：

```python
(child_class, callsite_file, callsite_line, ancestors)
```

字段含义：

- `child_class`：由 `static_module_tree[parent_class].attrs[attr_name]` 解析得到。
- `callsite_file`：命中的源码文件 basename。
- `callsite_line`：命中的源码行号。
- `ancestors`：从 root 到 parent 的调用链，按 outer → inner 排列。

若 child class 无法解析，也不能退回 dataclass key，而应返回 parent 语义的四元组 key，并记录原因。

### 2. ModuleList / Sequential 的 `_N` 后缀处理

已有方向：

- trace 事件名或 stack chain 带 `_N` 后缀时，优先解析为 `attr[N]`。
- 如果 `static_module_tree` 中存在对应展开实例，则归到精确实例。
- 如果不存在，记录 error / reason，并归到父 group；父 group 仍必须使用四元组 key。

---

## 建议落地结构

### Step A：从 AST 静态树传递到 timing pipeline

主流程需要保留并传递 `static_module_tree`：

```python
static_module_tree, roots, _ = build_static_module_tree(
    source_files,
    conditional_mode="train",
)

timing_data = build_timing_data_from_trace(
    events,
    source_files,
    mod_info,
    step_dur_us,
    profiler_steps,
    src_info=src_info,
    roots=roots,
    static_module_tree=static_module_tree,
)
```

然后逐层传到：

1. `build_timing_data_from_trace()`
2. `build_instance_timing_pipeline()`
3. `build_kernel_attribution_table()`
4. `build_loc_index()` / `match_kernel_to_instance()`

### Step B：基于 `static_module_tree` 构建 loc index

建议优先级：

1. 优先注册 `first_call_loc[attr_name]`。
2. 没有 first call 时，再注册 `attr_def_loc[attr_name]`。
3. 可补充注册 `dep_edge_locs[attr_name]` 中明确代表调用边的 loc。
4. 只有完全没有结构信息时，才注册 `<self>` fallback。

目标结构：

```python
loc_index[(basename(file), line)].append((parent_class, attr_name))
```

### Step C：attr 命中后解析实例 key

建议新增或复用 helper，语义如下：

```python
def resolve_attr_instance_key(static_module_tree, parent_class, attr_name, frame, chain):
    callsite_file = basename(frame.file)
    callsite_line = int(frame.line)

    child_class = lookup_child_class(static_module_tree, parent_class, attr_name)
    ancestors = build_ancestor_chain(static_module_tree, chain.frames, stop_class=parent_class)

    if child_class:
        return (child_class, callsite_file, callsite_line, tuple(ancestors))

    return (parent_class, callsite_file, callsite_line, tuple(ancestors))
```

需要支持的 attr 形态：

- 普通 attr：`self.block`
- 容器 attr：`self.blocks`
- 展开 attr：`self.blocks[3]`
- Sequential / ModuleList 展开的 `_N` 后缀

### Step D：bottom-up rollup

Step3 应基于统一四元组 key 做 rollup：

- child attr 的 inclusive kernel time 自底向上累加到 parent。
- parent group 保留自身 self cost。
- 纯 tensor op 归当前实例 self cost。
- 全栈无命中的 kernel 保留在 `__unattributed__`。

---

## 待确认点

1. **stack frame 顺序是否所有入口都稳定为 outer → inner。** 如果存在 trace 来源差异，需要 Step1 统一方向或 Step2 做显式归一化。
2. **`static_module_tree.attrs` 的字段格式。** 需要确认 child class 字段名、container 展开格式、file basename 归一化策略。
3. **同一 loc 多候选时的消歧优先级。** 建议顺序：精确 indexed attr > 普通 attr > container attr > `<self>`。
4. **ancestors 的构造来源。** 可以来自 stack frames 反查，也可以来自静态 DAG parent/child 关系；需要保证与 Step3 rollup 口径一致。
5. **`dep_edge_locs` 是否直接进入 loc index。** 如果某些 dep edge loc 只是数据依赖位置而非真实 Module callsite，需要避免过度注册。
6. **AST 分支与 timing 分支的边界。** 当前方案是 ast-timing 联合修复，但落地时仍要明确哪些 helper 属于 AST 静态结构，哪些属于 timing 归因，不要互相污染职责。

---

## 建议测例设计草案

<callout icon="first_place_medal" bgc="3">  
  按既定修复流程，下面只是测例设计草案。后续如果进入编码阶段，需要先让 NaN 确认测例设计，再写测试和修复代码。  
</callout>

### T-LOC-1：消除 `<self>` 行级 fallback 泛滥

**Mock 结构：** 一个 Parent 有两个 child attr，forward 中还有多行 tensor op。

**断言：**

- loc index 只注册真实 child attr callsite。
- 不因为 forward 行数增长而增加 `<self>` loc。
- 有 attrs 的 class 不产生 forward 每行 `<self>`。

### T-LOC-2：同一 child class 多 attr / 多 loc

**Mock 结构：** `self.a = Block()`、`self.b = Block()`，forward 分别在不同 loc 调用。

**断言：**

- loc A 定位到 `(Parent, a)`。
- loc B 定位到 `(Parent, b)`。
- 不只返回 `Block` class。

### T-LOC-3：内部 tensor op 冒泡归当前实例 self cost

**Mock stack：**

```text
Outer.forward:L10 self.inner(x)
Inner.forward:L50 torch.cat(...)
```

**断言：**

- L50 未命中 loc index 时，不生成行级 weak group。
- L10 命中 `(Outer, inner)` 后归到 `Inner` 四元组 key。
- reason 标记为外层 callsite 冒泡命中。

### T-LOC-4：全栈无命中归 `__unattributed__`

**Mock stack：** 所有 frame 都不在 loc index。

**断言：**

- 归因为 `__unattributed__`。
- reason 为 `no_stack_frame_hit_loc_index` 或等价原因。

### T-LOC-5：ModuleList `_N` 后缀

**Mock 结构：** `self.blocks = nn.ModuleList([...])`，trace chain 带 `_3`。

**断言：**

- 存在 `blocks[3]` 时归精确实例。
- 不存在 `blocks[3]` 时记录 error，并归父 group 四元组 key。

---

## 当前分支和工作区状态

<table header-row="true" header-col="false" col-widths="180,260,360">  
    <tr>  
        <td>项目</td>  
        <td>状态</td>  
        <td>说明</td>  
    </tr>  
    <tr>  
        <td>torch-trace-analyzer</td>  
        <td>`feat/ast-timing-fix`</td>  
        <td>已从最新 `origin/master` 拉起。当前只新增方案整理文档，未修改代码。</td>  
    </tr>  
    <tr>  
        <td>storage</td>  
        <td>`feat/ast-step2-tests`</td>  
        <td>存在已有未提交改动，切换新测试分支会覆盖 `testset/test_dag_rules.py`，因此已停止操作，未 stash、未覆盖、未修改。</td>  
    </tr>  
</table>

## 下一步建议

1. 先确认这版方案是否符合今天下午讨论的口径。
2. 若确认进入编码，先补正式测例设计并等待确认。
3. 确认后再在 `feat/ast-timing-fix` 中做最小实现，并补 `testset/test_dag_rules.py` 回归测例。
4. 修复完成后必须跑全量 `python3 testset/test_dag_rules.py`，目标为 7/7 PASS。
