# DAG Normalizer Framework Pattern 设计文档

## 1. 问题描述

当前 DAG normalizer 中的 B-group 识别逻辑会按拓扑连通性和结构同构模式，把当前 scope 内游离的 `FunctionalAttr` 节点圈成 Pattern group。该逻辑本身适合识别模型代码中重复出现的拓扑连续链，但在框架函数被展开时会出现可读性问题。

以 lgtorch 框架函数（例如 `get_total_loss`）为例，其内部会展开大量框架层算子节点，例如 `add_` × 97、`sum` × 96 等。这些节点在 DAG 上通常与模型自身的 loss 计算节点（例如 `where`、`mul`、`bce` 等）拓扑连通。当前 B-group 候选池没有基于 `call_loc` / 调用栈区分框架层节点和模型层节点，因此这些框架函数展开节点会被 B-group 当作普通游离节点吸收。

根因可以拆成两点：

1. **拓扑连通性过强**：框架函数内部算子与模型 loss 计算算子之间存在真实数据依赖，按连通分量或拓扑连续性看，它们会自然落入同一候选区域。
2. **缺少 call_loc 过滤**：B-group 当前没有在候选节点池中排除框架路径来源节点，导致来自 `torch` / `lgtorch` 的框架展开节点与模型业务节点混在一起参与 Pattern group 识别。

在 5698781 模型中，典型表现是 `Pattern2` 被扩张为包含 **2065 个节点** 的巨型 Pattern。该 Pattern 同时包含大量 lgtorch 框架函数展开节点和模型自身 loss 计算节点，最终导致前端视图中出现超大 Pattern group，完全丧失可读性和定位价值。

## 2. 方案：在 B-group 前增加 Framework Pattern 识别步骤

在现有执行顺序中新增一个 B-group 前置步骤，用于先识别并圈出框架函数展开的游离节点。该步骤可称为 **C-group** 或 **Framework Pattern**。为避免引入额外概念混乱，设计文档中统一称为 **Framework Pattern**。

### 2.1 新执行顺序

调整后的 normalizer 执行顺序为：

```text
mode1 container expansion
→ Framework Pattern（新增）
→ B-group
→ A-group
```

Framework Pattern 必须位于 B-group 之前，原因是它的核心目标就是先从 B-group 候选节点池中剥离框架层节点，避免这些节点被 B-group 二次吸收。

### 2.2 框架节点识别逻辑

对于当前 scope 的每个候选游离节点（`FunctionalAttr`），检查其 `call_loc.frames`。`call_loc` 是 `CallLoc` 对象，包含：

- `file`：完整绝对路径
- `line`
- `col`
- `frames: list[Frame]`，每帧包含 `file` / `line` / `function_name`

若调用栈中**任意一帧**的 `file` 路径满足以下特征之一，则认为该节点是框架节点：

1. `file` 路径包含 `site-packages`，且同时包含 `torch` 或 `lgtorch`；
2. `file` 路径包含 `lgtorch`，且该路径不在 `modelcode_root` 目录下。

同时必须满足路径不在 `modelcode_root` 下，避免模型仓库中自带的同名目录或封装代码被误判为框架节点。

伪代码约束如下：

```text
is_framework_node(node, modelcode_root):
    for frame in node.call_loc.frames:
        file = normalized_absolute_path(frame.file)
        if file is under modelcode_root:
            continue
        if "site-packages" in file and ("torch" in file or "lgtorch" in file):
            return True
        if "lgtorch" in file:
            return True
    return False
```

该逻辑必须显式返回布尔值；如果节点没有 `call_loc` 或 `frames`，不能静默把它当作框架节点，应按非框架节点处理，保留给后续 B-group / A-group 正常处理。

### 2.3 Framework Pattern 成组逻辑

在当前 scope 内：

1. 收集所有满足 `_is_framework_node(node, modelcode_root) == True` 的游离 `FunctionalAttr` 节点；
2. 在这些框架节点之间按 DAG 边关系计算连通分量；
3. 每个连通分量生成一个 Framework Pattern group；
4. group 设置：
   - `metadata["synthetic_type"] = "framework_pattern"`
   - 命名为 `FrameworkPattern`、`FrameworkPattern#1`、`FrameworkPattern#2` ...
5. 已归入 Framework Pattern group 的节点必须从后续 B-group 候选节点池中排除。

Framework Pattern 不要求结构同构，不使用 `pattern_registry` 做模式归并；只要框架节点之间在当前 scope 内连通，就可以形成一个 Framework Pattern group。

### 2.4 B-group 候选池剔除约束

B-group 必须显式剔除已被 Framework Pattern 圈走的节点。

具体要求：`_apply_function_grouping_b` 构建候选节点池 `candidate_ids` 前，先排除已经归入 `metadata["synthetic_type"] == "framework_pattern"` group 的节点。这样框架层节点不会被 B-group 二次吸收，也不会再与模型自身 loss 计算节点合并成巨型 Pattern。

这里的剔除是强约束，不是展示层过滤。后续 B-group 的连通分量、同构识别、Pattern 命名都只能在剔除后的候选节点池上运行。

## 3. 关键决策

### 3.1 路径判断使用 `file` 字段，不使用 `function_name`

框架节点识别必须基于 `CallLoc.file` / `Frame.file` 路径，而不是 `function_name`。

原因：

- `function_name` 容易受装饰器、wrapper、内部实现重构影响；
- 框架函数名可能与用户模型函数名冲突；
- 路径能更稳定地区分 modelcode 与 site-packages / lgtorch 框架代码。

### 3.2 只要求任意一帧命中框架路径

框架节点判断只要求 `call_loc.frames` 中**任意一帧**命中框架路径，不要求所有帧都是框架帧。

原因是实际调用栈通常会同时包含：

```text
模型文件调用点
→ lgtorch wrapper / loss helper
→ torch op
```

调用点可能仍然是模型文件，但框架帧位于调用栈中间。因此如果要求所有帧都是框架帧，会漏掉框架函数展开节点。

### 3.3 Framework Pattern 不需要 `pattern_registry`

Framework Pattern 的目标是把框架层展开节点从模型业务节点中剥离出来，提升可读性，而不是发现重复结构模式。因此它不要求结构同构，不需要注册到 `pattern_registry`。

连通即可成组的好处是：

- 逻辑简单；
- 不依赖框架内部算子形态是否稳定；
- 能直接阻断框架节点继续进入 B-group 候选池。

### 3.4 不引入 fallback 或静默跳过路径

本设计不引入 fallback、兼容 shim 或软化错误路径。

- 路径判定不依赖 `function_name` 兜底；
- Framework Pattern 不依赖 pattern 同构失败后的 fallback；
- B-group 必须强制排除 Framework Pattern 节点，不能以“候选池后续自然不会选中”为理由省略该约束。

## 4. 需要修改的文件和函数

目标文件：`scripts/dag_normalizer.py`

### 4.1 新增 `_is_framework_node(node, modelcode_root: str) -> bool`

职责：判断单个 `FunctionalAttr` 节点是否来自 torch / lgtorch 框架路径。

设计要点：

- 输入 `node` 和 `modelcode_root`；
- 遍历 `node.call_loc.frames`；
- 使用 frame 的 `file` 字段做路径判断；
- 命中条件为：
  - 不在 `modelcode_root` 下；
  - 且包含 `site-packages` + `torch` / `lgtorch`，或包含非 modelcode 下的 `lgtorch`；
- 返回严格布尔值；
- 不使用 `function_name`；
- 不做正则 fallback。

### 4.2 新增 `_apply_framework_pattern_grouping(dag, dag_scope, registry, ...)`

职责：在当前 scope 内识别 Framework Pattern group。

设计要点：

- 输入与现有 grouping 函数保持一致，便于在 `normalize_containers_recursive` 中串接；
- 只处理当前 scope 内的游离 `FunctionalAttr` 候选节点；
- 调用 `_is_framework_node` 过滤框架节点；
- 基于框架节点子图计算连通分量；
- 为每个连通分量创建 group；
- 写入 `metadata["synthetic_type"] = "framework_pattern"`；
- group 命名为 `FrameworkPattern`、`FrameworkPattern#1`、`FrameworkPattern#2`；
- 返回或更新足够的信息，使 B-group 能排除这些节点。

### 4.3 修改 `_apply_function_grouping_b` 的候选节点池构建

职责：确保 B-group 不会二次吸收 Framework Pattern 节点。

设计要点：

- 在构建 `candidate_ids` 前，先识别已归入 `metadata["synthetic_type"] == "framework_pattern"` group 的节点；
- 从 B-group 的 `candidate_ids` 中排除这些节点；
- 后续连通分量、同构识别、Pattern group 创建都基于排除后的候选池；
- 该排除逻辑必须在后端 normalizer 内完成，不得放到前端展示层。

### 4.4 修改 `normalize_containers_recursive`

职责：调整 grouping 调用顺序。

设计要点：

- 当前顺序为：`mode1 container expansion → B-group → A-group`；
- 修改为：`mode1 container expansion → Framework Pattern → B-group → A-group`；
- 确保 Framework Pattern 的结果在进入 B-group 前已经写入 DAG / registry / group metadata；
- 保持 A-group 仍在 B-group 后执行。

## 5. UT 设计（测例方案，不是代码）

本节只描述测例方案，不直接实现代码。后续进入实现阶段前，需要按仓库测试规范在 `storage/testset/unit/` 中落地正式 UT，并复用 `dag_session_test_infra.py` 中已有测试基础设施。

### T_FP-1：框架节点判断——frame 含 `site-packages/torch` 路径

**目标**：验证 `_is_framework_node` 能识别 torch / lgtorch 框架路径。

**构造**：

- 构造一个 `FunctionalAttr` 节点；
- `call_loc.frames` 中至少一帧的 `file` 为类似：
  - `/usr/local/lib/python3.10/site-packages/torch/nn/functional.py`
  - 或 `/usr/local/lib/python3.10/site-packages/lgtorch/loss.py`
- `modelcode_root` 指向 `/workspace/modelcode`。

**断言**：

- `_is_framework_node(node, modelcode_root)` 返回 `True`。

### T_FP-2：框架节点判断——frame 全在 modelcode 路径下

**目标**：验证模型代码路径不会被误判为框架节点。

**构造**：

- 构造一个 `FunctionalAttr` 节点；
- `call_loc.frames` 全部位于 `/workspace/modelcode/...`；
- 即使路径中存在业务侧同名目录，例如 `/workspace/modelcode/lgtorch_adapter/loss.py`，只要它位于 `modelcode_root` 下，也不能判定为框架节点。

**断言**：

- `_is_framework_node(node, modelcode_root)` 返回 `False`。

### T_FP-3：Framework Pattern 识别——连通框架节点成组且不参与 B-group

**目标**：验证 2 个连通框架节点会被圈成一个 Framework Pattern group，并从 B-group 候选池中剔除。

**构造**：

- 构造当前 scope 下 2 个游离 `FunctionalAttr` 节点 `F1`、`F2`；
- 二者通过 DAG 边连通，例如 `F1 -> F2`；
- `F1`、`F2` 的 `call_loc.frames` 均包含非 modelcode 下的 `site-packages/torch` 或 `lgtorch` 路径；
- 同一 scope 下再放置若干普通模型节点，确保后续 B-group 仍有候选。

**断言**：

- 生成一个 Framework Pattern group；
- group 名称符合 `FrameworkPattern` / `FrameworkPattern#N` 命名规则；
- group 的 `metadata["synthetic_type"] == "framework_pattern"`；
- `F1`、`F2` 不出现在 `_apply_function_grouping_b` 的 `candidate_ids` 中；
- B-group 不会把 `F1`、`F2` 合并进普通 Pattern group。

### T_FP-4：Framework Pattern 不影响 B-group——非框架节点仍正常走 B-group

**目标**：验证新增 Framework Pattern 步骤不会破坏现有 B-group 对模型业务节点的识别。

**构造**：

- 构造当前 scope 下多个非框架 `FunctionalAttr` 节点；
- 这些节点的 `call_loc.frames` 全部位于 `modelcode_root` 下；
- 节点之间满足现有 B-group 的连通 / 同构识别条件；
- 同时可混入一组框架节点，验证两类节点分流。

**断言**：

- 非框架节点仍按现有 B-group 逻辑生成 Pattern group；
- 框架节点生成 Framework Pattern group；
- 两类 group 的成员不交叉；
- B-group 的行为只受 Framework Pattern 节点剔除影响，不改变非框架节点的正常 grouping 结果。
