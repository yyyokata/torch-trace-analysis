# Functional Grouping A/B 设计文档（方案 Q：CallLoc 扩展为完整调用链）

## 0. 任务边界

本次只输出设计文档，不写任何实现代码。

设计目标基于 NaN 已确认的方案 Q：**把 `CallLoc` 从“单帧调用点”扩展为“完整 user frame 调用栈”**，再让 `normalize` 阶段据此完成 A 类嵌套分组，并在其后做 B 类同行 functional 合并。

本文所有结论都直接基于以下源码：

- `scripts/attr_types.py`
- `storage/mock/dag_session_utils.py`
- `storage/mock/dag_session_tracer.py`
- `scripts/dag_serializer.py`
- `scripts/dag_normalizer.py`
- 补充参考：`scripts/dag_types.py`、`storage/mock/dag_session.py`、`storage/testset/unit/test_dag_normalizer.py`、`storage/testset/unit/dag_session_test_infra.py`

---

## 1. 现状诊断

### 1.1 当前 `CallLoc` 只有单帧

`scripts/attr_types.py:9-13` 当前定义为：

```python
@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int
```

结论：

1. 当前 `CallLoc` 只有最内层一个调用点；
2. 没有函数名；
3. 没有 caller 链；
4. 因此无法区分：
   - `forward -> _helper_a`
   - `forward -> _helper_a -> _helper_b`
   这两种嵌套关系。

### 1.2 当前 `_extract_loc_from_frames()` 只取第一个命中的 user frame

`storage/mock/dag_session_utils.py:24-36` 当前逻辑：

```python
def _extract_loc_from_frames(...):
    for frame in reversed(frames[:-1]):
        if any(marker in frame.filename for marker in _SKIP_STACK_FILE_MARKERS):
            continue
        return CallLoc(file=frame.filename, line=frame.lineno, col=0)
```

而 `_SKIP_STACK_FILE_MARKERS` 在 `storage/mock/dag_session_utils.py:15-21` 定义为：

- `/site-packages/torch/`
- `/site-packages/bytedance/lagrange_torch/`
- `mock/dag_session.py`
- `mock/dag_session_utils.py`

结论：

1. 当前“是不是 user frame”的判断方式，**不是白名单根目录**，而是**按文件名包含这些 skip marker 反向过滤**；
2. 当前遍历方向是 `reversed(frames[:-1])`，即从更内层往外找；
3. 一旦命中第一个 user frame 就立刻返回；
4. 因此只保留了“最内层 user frame 的 file/line”，完整调用链已经在这里丢失。

### 1.3 functional 节点直接复用这个单帧 `CallLoc`

`storage/mock/dag_session_tracer.py` 中：

- source-less constant：`169-183`
- factory op：`199-212`
- functional record：`217-231`
- functional node materialize：`237-255`

关键点：

1. functional 节点 `call_loc` 来自 `_extract_loc_from_frames()`；
2. `FunctionalAttr.def_loc` 当前直接写成 `record.call_loc`（`239-240`）；
3. 这说明 functional 节点现在只有**调用点**，没有 helper `def` 位置信息。

### 1.4 serializer 当前只序列化 `{file, line, col}`

`scripts/dag_serializer.py:18-21`：

```python
def _serialize_call_loc(loc: CallLoc | None) -> dict | None:
    if loc is None:
        return None
    return {"file": loc.file, "line": loc.line, "col": loc.col}
```

`scripts/dag_serializer.py:122-191` 和 `194-209` 中，node / edge evidence 都统一走 `_serialize_call_loc()`。

结论：

1. HTML 当前拿到的 `call_loc` 只有三元组；
2. 如果要让前端也能看到完整调用链，只能在 `_serialize_call_loc()` 做增量扩展；
3. 不应重构掉已有的 `file/line/col` 字段，因为现有消费方已经依赖这三个字段。

### 1.5 normalize 当前只有 container 逻辑，没有 A/B hook

`scripts/dag_normalizer.py` 当前 `_normalize_single_dag()`（`51-101`）只做 container 归属修正：

1. 找相关 container；
2. 确保 container node 存在；
3. 补 `is_container`；
4. 将 member 归入 `inner_dag.direct_nodes`；
5. 对 native container，把 member-member 原始边筛入 `inner_dag.edges`；
6. **不重写外层数据流边**（`75-76` 已明确写死）。

这就是 A/B 的正确 hook 点：

- **必须在 `_normalize_single_dag()` 内加新阶段**；
- 不能放前端；
- 也不应回退到构图期，因为现在缺的是 normalize 期的结构分组，不是 tracer 期的 leaf 建图。

---

## 2. CallLoc 扩展方案

## 2.1 新结构

基于用户要求，`CallLoc` 扩展为：

```python
@dataclass(frozen=True)
class CallFrame:
    file: str
    line: int
    function_name: str


@dataclass(frozen=True)
class CallLoc:
    file: str
    line: int
    col: int
    frames: tuple[CallFrame, ...] = ()
```

其中：

- `file`: 最内层 user frame 文件，保持兼容；
- `line`: 最内层 user frame 行号，保持兼容；
- `col`: 保持兼容，仍为当前实现中的 `0`；
- `frames`: 完整 user frame 调用栈，顺序是**外层 → 内层**。

约束：

1. `frames[0]` 是最外层 user frame，通常是 `forward`；
2. `frames[-1]` 是最内层 user frame；
3. `frames[-1].file == file`，`frames[-1].line == line`；
4. `frames` 只保存 user frame，不保存 torch / lagrange_torch / mock 内部帧；
5. `frames` 使用 tuple，而不是 list，理由与当前 `CallLoc(frozen=True)` 一致：保持不可变、可稳定比较。

## 2.2 是否向后兼容

**向后兼容，但兼容边界要明确。**

### 代码层兼容

如果把 `frames` 设计成带默认值的字段：

```python
frames: tuple[CallFrame, ...] = ()
```

那么当前仓库内所有仍以 `CallLoc(file, line, col)` 方式构造对象的代码，都可以继续工作，例如：

- `storage/mock/dag_session_utils.py:33`
- `storage/mock/dag_session_types.py:10`
- `storage/mock/dag_session.py:69`

即：**Python 对象构造层面是向后兼容的。**

### 序列化层兼容

旧 HTML 数据里只有：

```json
{"file": "...", "line": 123, "col": 0}
```

新 serializer 只要继续保留这三个字段，再增量加 `frames`，旧消费逻辑仍可继续使用 `file/line/col`。

即：**序列化字段层面也可以向后兼容。**

### 语义层兼容

旧数据没有 `frames` 时，A 类分组不能凭空推断调用链。

因此语义上必须明确：

- `frames` 缺失或为空 ≠ “位于 forward 顶层”；
- `frames` 缺失或为空 = “调用链信息未知”；
- normalize 在这种情况下**必须跳过 A 类按调用链分组**，而不是做任何猜测性归桶。

这符合 NaN 的要求：**不能引入静默猜测路径。**

## 2.3 `frames` 为空时 normalize 的处理

规定如下：

1. 若节点 `call_loc.frames` 为空：
   - 不参与 A 类分桶；
   - 不因为它的存在去阻止其他 bucket；
   - 它继续留在当前层 `direct_nodes`；
2. 若某个 bucket 候选成员中混入 `frames` 为空的节点：
   - 该节点不属于任何 helper bucket；
   - 不把它硬塞进最近 bucket；
3. 若整个 DAG 层所有 functional 都没有 `frames`：
   - A 不生效；
   - B 也不生效（因为 B 的入口来自 A 组和明确的 helper scope）。

这不是 fallback，而是**信息不足时显式不分组**。

## 2.4 HTML 里 `call_loc` 字段怎么变

结论：**增量加字段，不重构。**

`scripts/dag_serializer.py:18-21` 当前输出只有：

```json
{"file": ..., "line": ..., "col": ...}
```

设计改为：

```json
{
  "file": "inner.py",
  "line": 128,
  "col": 0,
  "frames": [
    {"file": "model.py", "line": 210, "function_name": "forward"},
    {"file": "model.py", "line": 132, "function_name": "_helper"},
    {"file": "model.py", "line": 128, "function_name": "_inner"}
  ]
}
```

理由：

1. `file/line/col` 已被现有 serializer / 前端使用，不能删；
2. A/B normalize 运行在后端，核心依赖的是 Python `CallLoc` 对象，不依赖 HTML；
3. 前端将来若要展示“完整调用链”，只需读取新增 `frames` 字段；
4. 这是最小改动、最稳妥的兼容方式。

---

## 3. `_extract_loc_from_frames()` 修改设计

## 3.1 user frame 的过滤条件

必须严格沿用当前源码的过滤语义，不改成别的定义。

现状来自 `storage/mock/dag_session_utils.py:15-21`：

- 文件名包含 `/site-packages/torch/` → 跳过
- 文件名包含 `/site-packages/bytedance/lagrange_torch/` → 跳过
- 文件名包含 `mock/dag_session.py` → 跳过
- 文件名包含 `mock/dag_session_utils.py` → 跳过

因此新实现的 user frame 过滤仍定义为：

> 对 `frames[:-1]` 逐帧检查，只保留 `frame.filename` **不包含**上述 `_SKIP_STACK_FILE_MARKERS` 中任一 marker 的帧。

这里必须强调：

- 当前源码是**黑名单 skip marker**模型；
- 不是“路径前缀等于模型源码根目录”模型；
- 本次设计不改变这个判定原则，只是在现有判定结果上收集完整链。

## 3.2 新的提取逻辑

现有逻辑是“从内向外找第一个 user frame，然后返回”。

新逻辑改为：

1. 输入仍是 `frames: list[traceback.FrameSummary]`；
2. 仍先忽略最后一帧 `frames[-1]`，保持与当前 `frames[:-1]` 一致；
3. 在 `frames[:-1]` 中按原始顺序收集所有 user frame，得到 `user_frames`；
4. 若 `user_frames` 为空：
   - `fallback_loc is not None` → 返回 `fallback_loc`；
   - 否则 `raise RuntimeError(error_message)`；
5. 若 `user_frames` 非空：
   - 取 `inner = user_frames[-1]` 作为最内层 user frame；
   - 构造 `CallLoc(file=inner.filename, line=inner.lineno, col=0, frames=...)`；
   - `frames` 逐个转成 `CallFrame(file=frame.filename, line=frame.lineno, function_name=frame.name)`。

## 3.3 `frames` 顺序

**固定为外层 → 内层。**

原因：

1. 用户要求 `frames[0]` 是最外层、`frames[-1]` 是最内层；
2. A 类分组要在当前 scope 下找“直接 caller / 直接 callee”关系，外到内顺序最自然；
3. 这样：
   - 顶层 `forward` 写 functional → `frames == [forward]`
   - `forward -> _helper` → `frames == [forward, _helper]`
   - `forward -> _helper_a -> _helper_b` → `frames == [forward, _helper_a, _helper_b]`
4. normalize 里取“当前 helper 的直接 caller / 当前 scope 下的直接 callee”时，不需要再反转链。

## 3.4 是否限制最大深度

**本次设计不额外加最大深度限制。**

理由必须基于源码现状说明：

1. 当前 `_extract_loc_from_frames()` 已经直接处理 `traceback.extract_stack()` 的完整结果，没有做 depth cap（`storage/mock/dag_session.py:1339-1340`, `storage/mock/dag_session_tracer.py:199`, `217`）；
2. 本次只是把“找到的全部 user frame”保留下来，不是引入新的递归采样；
3. Python 调用栈本身是有限的，`traceback.extract_stack()` 已经是现成边界；
4. 如果后续真实 trace 证明某些模型调用栈过深导致 HTML 体积异常，再作为独立优化议题处理。

因此这里不先做拍脑袋截断，避免无证据引入新行为。

---

## 4. `dag_normalizer.py` 中的 A 类合并设计

## 4.1 A 类的 hook 点

A 类必须加在 `scripts/dag_normalizer.py::_normalize_single_dag()` 内。

当前 `_normalize_single_dag()` 顺序是：

1. 收集相关 container（`57-62`）
2. 把 container node 加回 `dag.nodes` / `dag.direct_nodes`（`63-68`）
3. 标记 `is_container`（`70-73`）
4. 为 container 回填 `inner_dag.direct_nodes` 和 `inner_dag.edges`（`77-100`）

设计顺序改为：

1. 先跑现有 container normalize；
2. 再跑 A 类分组；
3. A 类生成的新 synthetic helper group 的 `inner_dag` 上递归继续跑 A；
4. 在每个 synthetic helper group 内部，再跑 B 类合并。

这样与 NaN 已确认的顺序一致：

- A 负责按调用链形成嵌套函数层级；
- B 负责在某个 helper scope 内做同行 functional 聚合；
- B 必须晚于 A。

## 4.2 当前层扫描范围

A 类只扫描**当前 DAG 层的 `direct_nodes`**。

原因：

1. serializer 也是按 `dag.direct_nodes` 输出本层节点（`scripts/dag_serializer.py:218-233`）；
2. `dag.nodes` 是 flatten 全量索引（`scripts/dag_types.py:61-69`），不适合作为“本层直属成员”的分组输入；
3. container 现在也是通过 `inner_dag.direct_nodes` 表达直属成员归属。

因此 A 类算法的输入集合应是：

- `dag.direct_nodes` 中的直属 `ModuleNode`；
- 其中重点关注 `attr` 为 `FunctionalAttr` 的节点；
- 同时必须看清楚 bucket 内是否混入了真实 `ModuleNode` / container node。

## 4.3 A 类分桶键

对于当前 DAG 层每个直属节点：

1. 读取 `node.call_loc.frames`；
2. 若 `frames` 为空或长度 `< 2`：
   - 该节点没有“当前函数之外的直接 helper caller 信息”；
   - 不参与 A 分桶；
3. 若 `frames` 长度 `>= 2`：
   - `frames[-2].function_name` 是**最内层调用点的直接 caller 函数名**；
   - 它正是用户要求的 A 类分桶键。

这里与用户给出的定义保持一致：

> “次外层” = `frames[-2].function_name`。

### 为什么这里能表达嵌套

以 `forward -> _helper_a -> _helper_b -> torch.relu` 为例：

- `torch.relu` 的 `frames` 为 `[forward, _helper_a, _helper_b]`
- 在 root scope 扫描时：
  - 直接 helper 候选应是 `_helper_a`，不是 `_helper_b`
- 因此 A 类实现不能只看 `frames[-2]` 就在 root 层直接建 `_helper_b`

正确做法是：

1. 当前 scope 需要同时持有一个 `scope_depth` / `scope_frames_prefix_len` 概念；
2. root scope 对应前缀长度 `1`，即 `[forward]`；
3. 在 root scope 下，直属 callee 是 `frames[1]`；
4. 进入 `_helper_a` group 后，scope 前缀长度变为 `2`，此时直属 callee 才是 `frames[2]`。

因此，用户给出的“`frames[-2]` 是直接 caller”这个观察成立，但真正用于 normalize 递归分组时，仍要显式维护**当前 scope 的 frame 前缀长度**。

最终设计采用：

- 在每层 normalize 递归中维护 `scope_frames: tuple[str, ...]`；
- root scope 初始为 `("forward",)`；
- 若节点 `frames` 不以 `scope_frames` 为前缀，跳过；
- 若 `len(frames) == len(scope_frames)`，说明节点直接属于当前 scope 本体，不触发 A；
- 若 `len(frames) > len(scope_frames)`，当前层直属 callee 是：
  - `frames[len(scope_frames)].function_name`

这个设计比单纯使用 `frames[-2]` 更严格，也能正确实现嵌套 helper。

## 4.4 A 类“全是 FunctionalAttr”判定

对每个 helper bucket：

1. 先收集当前 scope 下、前缀匹配该 helper 子树的所有直属节点；
2. 检查这些直属节点是否**全部**满足：
   - `isinstance(node, ModuleNode)`
   - `isinstance(node.attr, FunctionalAttr)`
3. 只要有任一节点不是 `FunctionalAttr`：
   - 不做 A 合并；
   - 该 bucket 整体保持原状。

必须这样设计的原因：

- 用户明确要求 A 类 bucket 必须“全是 FunctionalAttr（无 ModuleNode）”；
- 真实 module call、container synthetic node、后续 A/B synthetic node 都不能混进 A bucket；
- 不能因为 bucket 内大多数是 functional，就把少数 module call 静默排除掉再强行合并。

## 4.5 A 类 synthetic `ModuleNode` 的构造

A 类 group 节点直接复用 container 当前的构造模式，而不是走构图期 `_create_module_node()`。

证据：

- container 现在就是在 normalize 里直接 new `ModuleNode(...)`（`scripts/dag_normalizer.py:137-155`）；
- `_create_module_node()` 依赖 live `nn.Module` 和 `path/call_index/construct_loc`（`storage/mock/dag_session.py:786-823`），不适合 synthetic helper group。

A 类节点建议字段如下：

### attr

使用 `ModuleAttr`：

- `attr_name = helper function 名`
- `class_name = "SyntheticFunctionGroup"`
- `def_loc = helper 定义位置`
- `class_def_loc = None`
- `is_native` 语义留在 node 层，不在 attr 上表达

### node

- `call_loc = 当前 helper 的调用点`
- `metadata["is_synthetic"] = True`
- `metadata["synthetic_type"] = "function_group"`
- `is_native = False`

其中 `is_native = False` 的理由是：

1. 这个节点不是运行时 hook 到的真实 `nn.Module`；
2. 也不是 PyTorch native module；
3. 把 synthetic group 误标成 native，会混淆它与真实容器补建节点的语义。

## 4.6 A 类节点的 `call_loc`

A 类 group 的 `call_loc` 定义为：

> 当前 scope 下，该 helper bucket 第一次出现时的调用点。

也就是：

- 从 bucket 成员里取 call sequence 最早的节点；
- 该节点的 `call_loc.frames` 一定以 `scope_frames + (helper_name,)` 为前缀；
- 其 `call_loc` 可作为该 helper group 的代表调用点。

不使用 `frames[-2]` 直接重建一个新 `CallLoc`，原因是：

- 现成 leaf 已有完整 `CallLoc`；
- 直接复用 leaf 中最早出现的那个 `CallLoc` 更简单，也与真实运行顺序一致。

## 4.7 A 类节点的 `def_loc`

这里必须回答“是否需要 helper 定义首行，以及如果需要，怎么取”。NaN 已明确：整个当前框架没有 AST 参与，因此 A 类 `def_loc` 设计不能引入 AST。

### 可用信息边界

当前 `CallFrame` / `frames` 设计只保存：

- `file`；
- `line`；
- `function_name`。

其中 `frames[-1].line` 是 helper 函数体内某一行，不是 `def` 首行；`frames[-1].function_name` 是该函数名。normalize 阶段没有活 frame 对象，也没有函数对象，因此不能依赖：

- `frame.f_locals / frame.f_globals`；
- `inspect.getsourcelines(fn)`；
- AST parse / AST walk。

### 是否必须精确到 `def` 首行

A group 已经有独立的 `call_loc`：

- `call_loc` 表达当前 scope 下 helper 被调用的位置；
- 例如 `frames[-2]` 对应调用者帧，能够说明 helper group 从哪里被调用；
- 这才是展示和定位调用链时最关键的信息。

`def_loc` 的价值只是辅助展示“这个 helper 大致定义在哪个文件/函数内”。在当前 frames 已经提供 `file + function_name` 的前提下，**不值得为了精确 def 首行引入 AST**。

因此最终设计不要求 A group 的 `def_loc` 精确到 `def` 首行。

### 逐行向上扫描方案评估

可以从 `frames[-1].file` 的 `frames[-1].line` 往前逐行扫描，找第一个形如 `def <function_name>` / `async def <function_name>` 的文本行。

覆盖范围：

- method：可覆盖；
- 模块级自由函数：可覆盖；
- 本地函数：通常可覆盖；
- 嵌套 `def`：通常可覆盖最近的同名定义。

但它存在明确歧义：

1. 装饰器、多行函数签名、类型参数等语法会增加文本匹配复杂度；
2. 同名嵌套函数或同名局部函数时，纯文本扫描只能按最近文本命中，无法做结构校验；
3. 注释、字符串、条件定义、动态生成函数等场景需要额外规则排除；
4. 为了覆盖这些情况，扫描逻辑会逐步演化成“半个 parser”，这与“不引入 AST、不要过度设计”的目标冲突。

所以本文不采用逐行向上扫描作为主设计，也不把它作为 fallback。

### 最终设计

A 类 `def_loc` 直接使用 helper 最内层执行帧的文件和行号：

1. 对 helper group 取代表节点的 `call_loc.frames[-1]`，记为 `leaf_frame`；
2. 生成 `CallLoc(file=leaf_frame.file, line=leaf_frame.line, col=0, frames=())` 作为 A group 的 `attr.def_loc`；
3. 同时 synthetic group 的 `attr_name` 仍使用 helper function 名，即 `leaf_frame.function_name` / 当前 bucket helper name。

语义定义调整为：

- A group `call_loc`：helper 在调用者中的调用点，代表“哪里调用了这个 helper”；
- A group `def_loc`：helper 函数体内的代表位置，代表“这个 helper 属于哪个源码文件和函数体”，不承诺是 `def` 首行。

这里 `def_loc.frames` 设计为空，理由：

- `def_loc` 仍然是一个源码位置，不是运行时调用链；
- 当前 `class_def_loc` / `construct_loc` 也都只表达一个位置，不携带运行时 stack；
- 没必要混入不属于定义语义的数据。

### 错误处理：不允许 fallback 或静默猜测

以下情况必须直接 `raise RuntimeError`，错误信息带上 helper group 名称和可用的 call_loc 信息：

1. `call_loc` 缺失或 `call_loc.frames` 为空，但该节点已被选入 A group；
2. `leaf_frame.file` 为空；
3. `leaf_frame.line` 不是正整数；
4. `leaf_frame.function_name` 与当前 bucket helper name 不一致。

这不是 fallback：它不猜测 def 首行，不做文本搜索，不做 AST parse；只是把 tracer 已经采集到的确定执行帧位置原样写入 `def_loc`。

## 4.8 A 类成员迁移与递归

对一个成功成组的 A bucket：

1. 新建 synthetic `ModuleNode`；
2. 将 bucket 内成员 node id 移入 `new_node.inner_dag.direct_nodes`；
3. `new_node.inner_dag.nodes` 也写入这些成员；
4. `new_node.inner_dag.edges` 取当前层 `dag.edges` 中两端都在成员集合内的原始边；
5. `dag.nodes` 追加 `new_node.node_id`；
6. `dag.direct_nodes` 中删除这些成员，并在 bucket 首次出现位置插入 `new_node.node_id`；
7. 然后对 `new_node.inner_dag` 递归继续执行 A；
8. 不在 A group 的 `inner_dag` 内执行 B，B 只在 root dag 和真实 ModuleNode 的 `inner_dag` 执行。

这套流程与 container 现有模式一致：

- **边不重写**；
- **成员归属靠 `inner_dag.direct_nodes` 表达**；
- **递归在 normalize 阶段完成**。

---

## 5. B 类合并算法设计

## 5.1 B 类执行时机

B 类严格放在 A 类之后，作为真实模块内部的同行 functional 合并。

执行范围改为：**只在真实 ModuleNode 的 inner_dag 上执行 B，包括 root dag 本身**。

具体范围：

1. root dag：代表整个模型顶层，可视为 root module 的 `inner_dag`；
2. 每个真实 `ModuleNode.inner_dag`：即运行时真实存在的 nn.Module 对应子图。

明确不执行 B 的范围：

1. Container synthetic node 的 `inner_dag`；
2. A group（`SyntheticFunctionGroup`）的 `inner_dag`；
3. B group（`SyntheticCallLocLineGroup`）的 `inner_dag`；
4. 其他 synthetic group 的 `inner_dag`。

原因：

- B 的语义是“真实模块 scope 内，同一源码行产生多个 functional leaf 时做展示合并”；
- Container inner_dag 表达容器归属，不是真实函数 scope，不应再做同行 functional 归并；
- A group inner_dag 已经表达 helper function scope，继续在内部做 B 会叠加两层 synthetic 展示实体，增加复杂度；
- B group inner_dag 是 B 的结果，不能再递归执行 B。

触发条件不变：

- 同一个真实 ModuleNode inner_dag / root dag 的直属节点；
- 按 `(call_loc.file, call_loc.line)` 分桶；
- 桶内节点数 `>= 2`；
- 桶内全部是 `FunctionalAttr`；
- 不混入真实 module、container、A group、已有 B group。

因此 root `forward` 直属 functional 现在会走 B：

- 若 forward 直属多个 functional 位于同一源码行，生成 root 层 B group；
- 若 forward 直属 functional 不同行，B 不触发，它们继续保留在 root `direct_nodes`；
- A 仍然不会为 `forward` 本体生成 function group，因为 A 的条件是 `len(frames) > len(scope_frames)`，而 forward 直属 leaf 的 frames 长度等于 root scope 长度。

## 5.2 B 类分桶键

B 类 bucket 键为：

- `src_file = node.call_loc.file`
- `src_start_line = node.call_loc.line`

即按 `(src_file, src_start_line)` 分桶。

这里直接基于现有单帧字段即可，不依赖 `frames`。

## 5.3 B 类成组条件

对某个 helper scope 的 `inner_dag.direct_nodes`：

1. 只看直属节点；
2. 按 `(file, line)` 分桶；
3. 某个桶满足以下全部条件时，才生成 B group：
   - 桶内节点数 `>= 2`；
   - 桶内所有节点都是 `FunctionalAttr`；
   - 不含真实 module node；
   - 不含 container node；
   - 不含 A 类 synthetic helper group；
   - 不含已存在的 B 类 synthetic group。

只要有任一非 functional 节点混入，该桶整桶不合并。

## 5.4 B 类 synthetic `ModuleNode` 设计

B 类 group 也使用 `ModuleNode + ModuleAttr + inner_dag`。

字段设计：

- `attr_name = f"{parent_func}:{line_no}"`
- `class_name = "SyntheticCallLocLineGroup"`
- `def_loc = None`
- `class_def_loc = None`
- `call_loc = 该行对应的代表 CallLoc`
- `metadata["is_synthetic"] = True`
- `metadata["synthetic_type"] = "callloc_group"`
- `is_native = False`

其中 `parent_func` 取当前 helper scope 的函数名，而不是 leaf 自己的最内层函数名。

例如：

- 当前位于 `_helper` group 的 `inner_dag`
- 某行 132 上有多个 functional
- B group 名称就是 `_helper:132`

## 5.5 B 类边处理

B 类边处理与 A 类一致：

1. 当前层外部 `dag.edges` 不重写；
2. B group `inner_dag.edges` 只收成员之间的原始边；
3. 跨 B group 边界的数据流继续留在外层。

因此：

- B 不是数据流上的新实体；
- B 只是 normalize 后的展示分组实体。

---

## 6. A/B synthetic ModuleNode 字段设计汇总

## 6.1 A 类 function group

| 字段 | 设计 |
|---|---|
| `attr_name` | helper function 名，如 `_get_reweight` |
| `class_name` | `SyntheticFunctionGroup` |
| `call_loc` | 该 helper 在当前 scope 下首次出现节点的 `call_loc` |
| `def_loc` | 直接使用代表节点 `call_loc.frames[-1]` 的 `file/line`，表示 helper 函数体内代表位置，不承诺是 `def` 首行 |
| `class_def_loc` | `None` |
| `metadata["is_synthetic"]` | `True` |
| `metadata["synthetic_type"]` | `"function_group"` |
| `is_native` | `False` |

## 6.2 B 类 callloc group

| 字段 | 设计 |
|---|---|
| `attr_name` | `"parent_func:132"` |
| `class_name` | `SyntheticCallLocLineGroup` |
| `call_loc` | 该行代表节点的 `call_loc` |
| `def_loc` | `None` |
| `class_def_loc` | `None` |
| `metadata["is_synthetic"]` | `True` |
| `metadata["synthetic_type"]` | `"callloc_group"` |
| `is_native` | `False` |

---

## 7. 与 Container 的复用点

## 7.1 可以直接复用的点

### 1. `_build_attr_to_node_ids()`

`scripts/dag_normalizer.py:223-227` 现有这个索引构建函数。

用途：

- 对 A/B 本身不是核心分桶依据；
- 但在同一个 normalize 流程里，仍可继续复用它维护 registry 级别的 attr->node 映射；
- 尤其是如果后续需要统一处理 synthetic attr 的 registry 索引时，这个函数不需要重写。

### 2. `ModuleNode(...)` 直接构造模板

`scripts/dag_normalizer.py:139-155` 的 container node 构造方式可以直接当模板复用：

- 直接 new `ModuleNode`
- 直接挂 `inner_dag`
- 直接填 `metadata`
- 不回到构图期 `_create_module_node()`

### 3. member-member 边筛入 `inner_dag.edges` 的模式

`scripts/dag_normalizer.py:88-95`：

- 建 `member_id_set`
- 筛两端都在集合内的边

A/B 的 inner edge 提取可以直接复用这个模式。

### 4. 递归 normalize 框架

`scripts/dag_normalizer.py:24-48` 当前已经有：

- 当前层 normalize
- rebuild adjacency
- 递归进入子 `inner_dag`

A/B 只需要在这个框架上增加：

- `scope_frames`
- `owner_module_object`

不需要另起一套遍历器。

## 7.2 不能直接复用的点

### 1. Container 的成员识别逻辑不能直接复用

Container 用的是：

- `attr.parent`
- `container_attr.items`

A/B 用的是：

- `call_loc.frames`
- `(file, line)`

两者分组依据完全不同。

### 2. Container 的 `ContainerAttr` 不能复用给 A/B

serializer 对 container 有专门分支（`scripts/dag_serializer.py:123-154`）。

A/B 不是容器语义，而是 synthetic group 语义，必须走普通 `ModuleAttr` 分支（`156-191`）。

### 3. Container 的 `is_native=True` 语义不能照搬

当前 native container synthetic node 是为了表达框架原生容器结构。

A/B synthetic group 不是原生容器，也不是真实 module，不应复用这个语义。

---

## 8. 边处理设计

结论：**A/B 新 ModuleNode 的边处理原则与 Container 一致，不重写原始数据流边。**

具体规则：

1. 外层 `dag.edges` 保持不变；
2. A/B group 的 `inner_dag.edges` 收入“成员之间的原始边”；
3. 跨 group 边界的边仍留在外层；
4. 不生成新的 boundary edge；
5. 不生成 containment edge；
6. 归属关系只通过 `inner_dag.direct_nodes` 表达。

### 与 Container 的唯一区别

A/B 虽然边处理与 Container 一致，但 **`dag.direct_nodes` 的改写更关键**。

原因来自 serializer：

- `serialize_dag()` 只序列化 `dag.direct_nodes`（`scripts/dag_serializer.py:218-233`）；
- 如果 A/B 成员留在父层 `direct_nodes`，同时又挂进 synthetic group 的 `inner_dag.direct_nodes`，前端会重复看到同一批节点。

因此 A/B 必须做：

- 成员移出父层 `direct_nodes`；
- synthetic group 插回父层 `direct_nodes`。

这一步是 A/B 相比 container 设计时需要格外强调的点。

---

## 9. UT 设计（6 个测例，最终版）

UT 放置文件：

- `develop/ast/storage/testset/unit/test_dag_normalizer.py`

原因：

1. 当前 container normalize 的单测已经在这个文件里（`test_dag_normalizer.py:1-387`）；
2. A/B 也属于 `dag_normalizer.py` 的职责边界；
3. `dag_session_test_infra.py` 已经提供 `run_dag_session()` 等基础设施（`dag_session_test_infra.py:206-239`），符合 NaN 的 infra 复用要求。

## 9.1 测例 1：A 类基础

- **测例名**：`test_a1_forward_calls_helper_all_functional_builds_synthetic_group`
- **mock Module 结构**：
  - `forward(x)` 中调用 `self._helper(x)`；
  - `_helper(x)` 内只有多个 functional，例如 `torch.relu`、`torch.sigmoid`；
  - 无任何真实子 module 调用。
- **期望断言**：
  1. root `direct_nodes` 下生成 attr_name=`_helper` 的 synthetic `ModuleNode`；
  2. 原 helper 内 functional leaf 从 root `direct_nodes` 移走；
  3. `_helper.inner_dag.direct_nodes` 包含原 functional leaf；
  4. `_helper.metadata["is_synthetic"] is True`；
  5. `_helper.metadata["synthetic_type"] == "function_group"`；
  6. `_helper.call_loc.frames` 以 `forward -> _helper` 为前缀；
  7. `_helper.attr.def_loc` 指向 `_helper` 函数体内的代表执行帧位置。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

## 9.2 测例 2：A 类嵌套

- **测例名**：`test_a2_nested_helper_chain_builds_two_level_synthetic_groups`
- **mock Module 结构**：
  - `forward(x)` 调 `self._helper_a(x)`；
  - `_helper_a(x)` 内先做一部分 functional，再调 `self._helper_b(...)`；
  - `_helper_b(...)` 内也全是 functional；
  - 不包含真实 module call。
- **期望断言**：
  1. root `direct_nodes` 下有 `_helper_a` synthetic group；
  2. `_helper_a.inner_dag.direct_nodes` 下有 `_helper_b` synthetic group；
  3. `_helper_b` 不是 root 的 sibling；
  4. `_helper_a` 和 `_helper_b` 的 `synthetic_type` 都是 `function_group`；
  5. 两层 group 的 `inner_dag.direct_nodes` 分别只包含各自 scope 内的直属成员。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

## 9.3 测例 3：B 类基础

- **测例名**：`test_b1_same_line_multiple_functionals_inside_helper_builds_callloc_group`
- **mock Module 结构**：
  - `forward(x)` 调 `self._helper(x)`；
  - `_helper(x)` 内某一行产生至少 3 个 functional，例如嵌套调用链全写在一行；
  - `_helper` 整体仍满足 A 类“全 functional”前提。
- **期望断言**：
  1. root 下先生成 `_helper` A group；
  2. `_helper.inner_dag.direct_nodes` 下生成 `"_helper:<line>"` B group；
  3. B group `metadata["synthetic_type"] == "callloc_group"`；
  4. B group `inner_dag.direct_nodes` 恰好包含该行的 functional leaf；
  5. 这些 functional leaf 不再直接出现在 `_helper.inner_dag.direct_nodes` 顶层。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

## 9.4 测例 4：B 类不合并

- **测例名**：`test_b2_same_line_mixed_functional_and_module_call_skips_callloc_group`
- **mock Module 结构**：
  - `forward(x)` 调 `self._helper(x)`；
  - `_helper(x)` 整体已进入某个 helper scope；
  - 在 `_helper` 内某一行同时出现 functional 与真实 module call；
  - 需要确保这个“module call”位于待检查 B scope 内，而不是让 A 先整体失败。
- **期望断言**：
  1. 不为该行生成 `"_helper:<line>"` B group；
  2. 该行上的 functional leaf 继续散落在当前 helper scope 顶层；
  3. 同行混入的真实 module node 也保留在当前层；
  4. 证明 B 的门槛是“整桶全 functional”，而不是“抽出 functional 子集单独合并”。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

## 9.5 测例 5：A 类不合并

- **测例名**：`test_a3_forward_direct_functionals_do_not_trigger_a_group`
- **mock Module 结构**：
  - `forward(x)` 直接写多个 functional；
  - 不调用 helper；
  - 不经过任何额外 Python 函数封装；
  - 多个 functional 分布在不同行，避免 B 因同行规则介入。
- **期望断言**：
  1. root `direct_nodes` 下不存在 synthetic function group；
  2. 这些 functional leaf 继续直接保留在 root `direct_nodes`；
  3. 不生成 A group；
  4. 不生成 B group，因为它们不满足同行条件；
  5. 对应 leaf 的 `call_loc.frames` 长度应为 1，仅包含 `forward`。
- **覆盖目标**：只验证“forward 本体直属 functional 不触发 A”，不再承担“root 层 B 不生效”的旧语义。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

## 9.6 测例 6：B 类在 root 层生效

- **测例名**：`test_b3_same_line_forward_direct_functionals_builds_root_callloc_group`
- **mock Module 结构**：
  - `forward(x)` 直接在同一源码行写出多个 functional，例如嵌套调用或同一行表达式产生多个 functional leaf；
  - 不调用 helper；
  - 不经过任何额外 Python 函数封装。
- **期望断言**：
  1. root `direct_nodes` 下不存在 synthetic function group；
  2. root `direct_nodes` 下生成 B group，`metadata["synthetic_type"] == "callloc_group"`；
  3. B group 的 `attr_name` 使用 root scope 的函数名和行号，例如 `"forward:<line>"`；
  4. B group `inner_dag.direct_nodes` 恰好包含该行 forward 直属 functional leaf；
  5. 这些 functional leaf 不再直接出现在 root `direct_nodes` 顶层。
- **覆盖目标**：验证 B 的执行范围已扩展到 root dag，并与 A 的“不触发 helper group”语义解耦。
- **放置文件**：`testset/unit/test_dag_normalizer.py`

---

## 10. 最终设计结论

基于当前源码，最终设计结论如下：

1. **核心变化不是给 `CallLoc` 加一个函数名，而是把它升级成完整 user frame 调用链容器。**
2. `CallLoc` 保留现有 `file/line/col`，新增 `frames`，从而做到对象构造和序列化双向兼容。
3. `_extract_loc_from_frames()` 不改变现有 user frame 过滤规则，只把“第一个命中 frame”改成“收集全部 user frame”。
4. `frames` 顺序固定为**外层 → 内层**，root 通常是 `forward`。
5. A 类分组必须放在 `dag_normalizer.py::_normalize_single_dag()` 中，并通过 `scope_frames` 前缀递归来实现真正的嵌套 helper 分组。
6. B 类分组必须在 A 之后执行，只覆盖 root dag 和真实 ModuleNode 的 inner_dag，不进入 Container / A group / B group 等 synthetic inner_dag。
7. A/B synthetic group 都使用 `ModuleNode + ModuleAttr + inner_dag`，不回退到构图期 `_create_module_node()`。
8. A/B 边处理原则与 Container 一致：**不重写原始数据流边，只把组内 member-member 边收入 `inner_dag.edges`**。
9. A 类 `def_loc` 不引入 AST，也不做文本扫描；直接使用 `frames[-1]` 的 `file/line` 作为 helper 函数体内代表位置，不承诺精确到 `def` 首行。
10. 6 个 UT 都应落在 `testset/unit/test_dag_normalizer.py`，复用 `dag_session_test_infra.py` 现有 infra。

以上即本次“只出设计文档、不写实现代码”的最终版设计。