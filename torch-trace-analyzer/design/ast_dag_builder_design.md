# AST 扫描填充新 DAG IR 设计文档

## 0. 文档目标

本文档只设计 **Phase 1~5**：从 `source_files` 出发，经 ASTFrontend 复用、Attr 扫描、Node 构建、Edge 构建，最终装配 `DagContext`。

**本文档不做的事情：**
- 不写完整实现代码
- 不引入新的正则兜底
- 不在前端做任何 trace 数据处理
- 不引入 annotation-only fallback；遇到只靠类型注解才能猜值的路径，一律返回 `None`
- 不删除旧接口 `build_static_module_tree()`；本阶段只增加桥接层

**本阶段目标：**
1. 复用已有 `ASTFrontend` / `build_static_module_tree()` / `_AST_ModuleTreeExtractor`
2. 把 `ModuleAttr / ContainerAttr / InputAttr / ResultAttr` 统一落到新 DAG IR
3. 让 `DagContext` 成为后续 lineage / 高亮 / edge evidence 的唯一后端输入

---

## 1. 关键设计结论

### 1.1 作用域划分

- **Phase 1**：只做 `ASTFrontend` cache 构建，不新增 AST 语义
- **Phase 2**：只做 Attr 发现与标准化，不直接产出 `DagContext`
- **Phase 3**：把每个 Attr 转成 `DagNode`，分配全局唯一 `node_id`
- **Phase 4**：把旧 `dep_edges` / containment / Input-Root / Root-Result 统一转成 `DataFlowEdge`
- **Phase 5**：把 registry + root DAG + loc index 装进 `DagContext`

### 1.2 `forward_use_loc` 删除后的新方案

`forward_use_loc` 已从 `attr_types.py` 删除，本设计**不恢复该字段**，原因是：

- `InputAttr` 的职责是“节点身份”，核心是输入源的**定义位置**
- forward 里的调用位置属于**边 evidence**，不是节点身份
- 一个 Input 可能在多个 helper / 多个分支中被使用，单个 `forward_use_loc` 无法完整表达

因此采用以下新方案：

- `InputAttr.def_loc`：保留输入源定义位置，作为 `InputNode.call_loc`
- forward 可达调用位置：**不存回 `InputAttr`**
- Phase 4 构建 `InputNode -> RootModuleNode` 边时，按 `InputAttr + ASTFrontend` 重新派生使用位置，写入 `VarEvidence.steps`

换句话说：

```text
节点位置 = def_loc
边使用位置 = evidence.steps[*].loc
```

这比旧的单字段方案更符合 IR 分层。

### 1.3 root 节点采用 synthetic root attr

`build_static_module_tree()` 返回的 `roots` 是**类名列表**，不是 Attr；而 Phase 3 规则要求“每个 Attr 对应一个 DagNode”。

因此设计上在进入 Phase 3 之前，先注入一个 synthetic root attr：

```python
ModuleAttr(
    attr_name="__root__",
    class_name=primary_root_class,
    def_loc=root_class_def_loc,
    source_expr=primary_root_class,
)
```

用途：
- 给 `Input -> Root -> Result` 提供稳定的 root endpoint
- 让 `root.inner_dag` 有明确归属节点
- 不修改 `dag_types.py`

---

## 2. 整体数据流（Pipeline 视图）

## 2.1 总览

```text
source_files: dict[str, list[str]]
  → [Phase 1] _build_ast_frontends(source_files)
  → [Phase 2] scan_all_attrs(source_files, ast_frontends)
  → [Phase 2.5] inject synthetic root attr
  → [Phase 3] build_nodes(attrs_by_class)
  → [Phase 4] build_edges(tree_dict, attrs_by_class, registry, roots, ast_frontends)
  → [Phase 5] assemble_dag_context(registry, roots, attrs_by_class, tree_dict)
  → DagContext
```

> 说明：`Phase 2.5` 不是新的 AST 解析阶段，只是 bridge 层对 `roots` 做一次 root attr 注入，目的是把 root 也纳入 “Attr → Node” 统一路径。

## 2.2 各阶段输入/输出

| Phase | 输入 | 输出 | 关键约束 |
|---|---|---|---|
| Phase 1 | `source_files` | `ast_frontends: dict[str, ASTFrontend]` | 只复用 `_build_ast_frontends`，不重复 `ast.parse` |
| Phase 2 | `source_files` + `ast_frontends` | `attrs_by_class: dict[str, list[Attr]]` | 每个 owner class 下 Attr 顺序稳定；不在 scan 层构 DAG |
| Phase 2.5 | `attrs_by_class` + `roots` + `class_map` | 注入 synthetic root attr 的 `attrs_by_class` | 只注入 `roots[0]` 对应 primary root |
| Phase 3 | `attrs_by_class` | `(registry: dict[int, DagNode], next_id: int)` | `node_id` 全局唯一递增；每个 Attr 恰好生成一个 DagNode；`attr.attr_id = node_id` |
| Phase 4 | `tree_dict` + `attrs_by_class` + `registry` + `ast_frontends` | `edges_by_scope` / 直接写入 DAG 所需边列表 | `dep_edges` 只转数据流边；containment 单独建；Input/Result 边先允许 evidence 为空 |
| Phase 5 | `registry` + `roots` + `attrs_by_class` + `tree_dict` | `DagContext` | `DagContext.__post_init__` 自动建 `loc_attr_index` |

## 2.3 推荐桥接主流程 pseudo-code

```python
def build_dag_context_from_source(source_files, conditional_mode="train"):
    tree_dict, roots, class_map = build_static_module_tree(
        source_files,
        conditional_mode=conditional_mode,
    )

    ast_frontends = _build_ast_frontends(source_files)
    attrs_by_class = scan_all_attrs(source_files, ast_frontends)
    attrs_by_class = inject_root_attr(attrs_by_class, roots, class_map)

    registry, _ = build_nodes(attrs_by_class)
    build_edges(tree_dict, attrs_by_class, registry, roots, ast_frontends)
    return assemble_dag_context(registry, roots, attrs_by_class, tree_dict)
```

---

## 3. Phase 2：Attr 扫描层设计

## 3.1 扫描入口函数签名

```python
def scan_all_attrs(
    source_files: dict[str, list[str]],
    ast_frontends: dict[str, ASTFrontend],
) -> dict[str, list[Attr]]:
    """
    Returns:
        {owner_class_name: [Attr, ...]}

    包含：
      - ModuleAttr
      - ContainerAttr
      - InputAttr
      - ResultAttr

    约束：
      - 不在 scan 层自行 ast.parse
      - owner_class_name 是“Attr 所属类”，不是 attr.class_name
      - 返回顺序稳定：按 (def_loc.file, def_loc.line, attr_name) 排序
    """
```

这里的 `dict[str, list[Attr]]` 语义是：

- key = **owner class**，即“这个 Attr 定义在哪个类的 `__init__` / `forward` 可达语义里”
- `Attr.class_name`：继续沿用现有类型自己的语义
  - `ModuleAttr.class_name` = child module class
  - `ContainerAttr.class_name` = container class / full class ref
  - `InputAttr.class_name` = owner class
  - `ResultAttr.class_name` = owner class

## 3.2 内部辅助数据结构

为了保持 `scan_all_attrs()` 的返回值简单，Phase 2 内部允许使用若干 side state，但**最终只返回 Attr 列表**。

```python
@dataclass
class PendingFeatureColumn:
    owner_class: str
    binding_kind: str        # local / dict_slot / self_attr
    binding_name: str        # e.g. fc / fc_dict / self.seq_fc
    dict_key_expr: str | None
    def_loc: CallLoc
    source_expr: str
    owner_expr: str | None
    slot_expr: str | None

@dataclass
class ResultSeed:
    owner_class: str
    var_name: str
    def_loc: CallLoc
```

这些结构只在扫描阶段临时使用：
- `PendingFeatureColumn`：解决 `feature_column` 先声明、后 `get_vector()` / `get_size_tensor()` 才真正出 tensor 的问题
- `ResultSeed`：保证 `result_alias.head(...)` 只匹配真正来自 `LG.result()` 的对象

## 3.3 LG alias 识别

### 3.3.1 目标

每个文件都需要建立 `lg_context_aliases: set[str]`，覆盖至少以下模式：

```python
from bytedance.lagrange_torch.context import CONTEXT as LG
from bytedance.lagrange_torch.context import CONTEXT as M
import bytedance.lagrange_torch.context as lg_ctx
```

识别结果：
- 前两种：alias 分别是 `LG` / `M`
- 第三种：真正可调用上下文是 `lg_ctx.CONTEXT`

因此 scanner 里应支持两类判断：
- **name alias**：`LG.dense_feature(...)`
- **module alias**：`lg_ctx.CONTEXT.dense_feature(...)`

### 3.3.2 伪码

```python
def collect_lg_aliases(file_tree: ast.Module) -> tuple[set[str], set[str]]:
    context_name_aliases = set()      # e.g. {"LG", "M"}
    context_module_aliases = set()    # e.g. {"lg_ctx"}

    for node in file_tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "bytedance.lagrange_torch.context":
                for alias in node.names:
                    if alias.name == "CONTEXT":
                        context_name_aliases.add(alias.asname or alias.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bytedance.lagrange_torch.context":
                    context_module_aliases.add(alias.asname or alias.name.split(".")[-1])

    return context_name_aliases, context_module_aliases
```

```python
def is_lg_context_expr(expr, name_aliases, module_aliases) -> bool:
    # LG
    if isinstance(expr, ast.Name) and expr.id in name_aliases:
        return True

    # lg_ctx.CONTEXT
    if (
        isinstance(expr, ast.Attribute)
        and expr.attr == "CONTEXT"
        and isinstance(expr.value, ast.Name)
        and expr.value.id in module_aliases
    ):
        return True

    return False
```

**禁止做法：**
- 不允许用字符串 contains / 正则匹配 import 文本
- 不允许假设 alias 一定叫 `LG`

## 3.4 ModuleAttr / ContainerAttr 与 `_AST_ModuleTreeExtractor` 的分工

### 3.4.1 分工原则

- `_AST_ModuleTreeExtractor` 继续负责扫描 `__init__` 中的 module / container 赋值
- `scan_all_attrs()` 只负责把 extractor 结果**标准化**为 `Attr` 对象
- 不在 scanner 中重复实现一套 module tree 提取逻辑

### 3.4.2 转换规则

输入来源直接来自 `_AST_ModuleTreeExtractor(source_files, ast_frontends=ast_frontends)`：
- `extractor.ast_class_attrs`
- `extractor.ast_container_elems`
- `extractor.ast_container_kinds`
- `extractor.class_assignments`（用于回拿定义行）

输出：
- 普通子模块 → `ModuleAttr`
- 容器本体（`ModuleList` / `ModuleDict` / `Sequential`）→ `ContainerAttr`
- 容器元素 → 子 `ModuleAttr`，并挂到 `ContainerAttr.items`

### 3.4.3 伪码

```python
def convert_extractor_attrs(source_files, ast_frontends):
    extractor = _AST_ModuleTreeExtractor(source_files, ast_frontends=ast_frontends)
    attrs_by_class = defaultdict(list)

    def build_def_loc(owner_class, attr_name):
        for item in extractor.class_assignments.get(owner_class, []):
            if item.get("attr") == attr_name and item.get("line"):
                return CallLoc(file=extractor.classes[owner_class]["file"], line=item["line"], col=0)
        return CallLoc(file=extractor.classes[owner_class]["file"], line=0, col=0)

    for owner_class in extractor.nn_module_classes:
        # 1. direct attrs
        for attr_name, child_class in extractor.ast_class_attrs.get(owner_class, {}).items():
            attrs_by_class[owner_class].append(
                ModuleAttr(
                    attr_name=attr_name,
                    class_name=child_class,
                    def_loc=build_def_loc(owner_class, attr_name),
                )
            )

        # 2. containers
        for container_attr, container_kind in extractor.ast_container_kinds.get(owner_class, {}).items():
            container = ContainerAttr(
                attr_name=container_attr,
                class_name=container_kind,
                def_loc=build_def_loc(owner_class, container_attr),
                container_kind=container_kind,
            )
            for idx_or_key, child_class in enumerate_or_keyed(extractor.ast_container_elems.get(owner_class, {}).get(container_attr, [])):
                child = ModuleAttr(...)
                container.add_child(idx_or_key, child)
                attrs_by_class[owner_class].append(child)
            attrs_by_class[owner_class].append(container)

    return attrs_by_class
```

> 注意：`ContainerAttr` 也要保留在 `attrs_by_class` 里，后续会进入 Phase 3。否则 containment 路径会断。
> 
> 这样做的好处是：Phase 2 直接复用 extractor，不需要为了扫描 Attr 再调用一遍旧的 DAG tree 组装逻辑。

## 3.5 InputAttr 扫描策略

## 3.5.1 扫描目标

本阶段最少覆盖以下 LG input source：

- `dense_feature`
- `feature_column`
- `get_sample_rate`
- `get_bias`
- `get_sample_bias`
- `label`
- `slot`
- `global_step`
- `group_candidate_size`

可通过白名单常量维护：

```python
LG_INPUT_FUNCS = {
    "dense_feature",
    "feature_column",
    "get_sample_rate",
    "get_bias",
    "get_sample_bias",
    "label",
    "slot",
    "global_step",
    "group_candidate_size",
}
```

## 3.5.2 两阶段策略

### 阶段一：定义阶段（`__init__` / init helper）

识别：

```python
self.attr = LG.dense_feature(...)
self.attr = LG.get_sample_rate()
local_fc = LG.feature_column(...)
fc_dict[key] = LG.feature_column(...)
self.fc_dict[key] = LG.feature_column(...)
```

处理策略：

1. `self.attr = LG.xxx(...)`
   - 若 `xxx != feature_column`，直接创建 `InputAttr`
   - 若 `xxx == feature_column`，先记为 pending，再等 forward 使用阶段确认

2. `local = LG.feature_column(...)`
   - 不立即产出 `InputAttr`
   - 记到 `PendingFeatureColumn(binding_kind="local")`

3. `dict[key] = LG.feature_column(...)`
   - 记到 `PendingFeatureColumn(binding_kind="dict_slot")`

4. `self.xxx = LG.feature_column(...)`
   - 也建议先走 pending
   - 因为真正产 tensor 的位置是后续 `self.xxx.get_vector(...)` / `self.xxx.get_size_tensor()`

### 阶段二：forward 可达使用阶段

方法集合：

```python
reachable_methods = {"forward"} | fe.get_reachable_helpers(owner_class, "forward")
```

识别以下使用模式：

- `self.attr()`
- `local_attr()`
- `fc.get_vector(slc)`
- `fc.get_size_tensor()`
- `dict_var[key].get_vector(slc)`
- `dict_var[key].get_size_tensor()`
- `self.fc_dict[key].get_vector(slc)`

确认逻辑：

- direct callable input（dense/sample_rate/label/bias）
  - 用于确认该输入是 forward 可达
  - 不写回 `InputAttr` 新字段
  - 后续 evidence 在 Phase 4 再派生

- feature_column
  - 一旦看到 `get_vector()` / `get_size_tensor()`，才把对应 pending binding **物化** 为 `InputAttr`
  - `kind = "feature_column"`
  - `owner_expr` / `slot_expr` / `source_expr` 保留下来，方便后续 evidence/debug

## 3.5.3 `forward_use_loc` 的替代存储方案

本设计不在 `InputAttr` 上加新字段，而是新增一个 **Phase 4 helper**：

```python
def collect_input_use_steps(
    input_attr: InputAttr,
    owner_class: str,
    ast_frontend: ASTFrontend,
) -> list[EvidenceStep]:
    """
    根据 InputAttr 反查 forward reachable 的调用点。
    结果直接进入 DataFlowEdge.evidence。
    """
```

也就是说：
- Phase 2 负责“Input 是谁”
- Phase 4 负责“Input 在 forward 哪里被消费”

这样不需要改 `attr_types.py`。

## 3.5.4 伪码

```python
def scan_input_attrs(owner_class, fe, lg_alias_ctx):
    direct_inputs = []
    pending_fc = {}

    # A. init bindings
    for stmt in iter_init_and_reachable_init_helpers(owner_class, fe):
        if matches_self_lg_assign(stmt, lg_alias_ctx):
            attr_name, func_name, call = parse_self_lg_assign(stmt)
            if func_name == "feature_column":
                pending_fc[("self_attr", attr_name)] = build_pending(...)
            else:
                direct_inputs.append(
                    InputAttr(
                        attr_name=attr_name,
                        class_name=owner_class,
                        def_loc=loc(stmt),
                        kind=func_name,
                        source_expr=fe._node_to_text(call),
                    )
                )

        elif matches_local_lg_assign(stmt, lg_alias_ctx) and func_name == "feature_column":
            pending_fc[("local", local_name)] = build_pending(...)

        elif matches_dict_slot_lg_assign(stmt, lg_alias_ctx) and func_name == "feature_column":
            pending_fc[("dict_slot", dict_var, key_expr)] = build_pending(...)

    # B. forward reachable use sites
    realized_fc = []
    for method_name in {"forward"} | fe.get_reachable_helpers(owner_class, "forward"):
        for call in iter_calls(owner_class, method_name, fe):
            if matches_feature_column_use(call):
                pending = resolve_pending_feature_column(call, pending_fc, fe, method_name)
                if pending:
                    realized_fc.append(materialize_feature_column_input(owner_class, pending))

    return dedup_inputs(direct_inputs + realized_fc)
```

## 3.6 ResultAttr 扫描策略

## 3.6.1 目标

支持以下模式：

```python
result = LG.result()
run_train = LG.result()
result.head(...)
LG.result().head(...)
LG.result().head(...).head(...)
```

## 3.6.2 识别规则

### 规则 A：先识别 result seed

```python
result = LG.result()
run_train = LG.result()
```

记为：

```python
ResultSeed(owner_class="MyModel", var_name="run_train", def_loc=...)
```

### 规则 B：只对真正的 result seed 匹配 `.head(...)`

合法 head receiver：
- `ast.Name` 且变量名在 `result_seeds` 中
- `LG.result()`
- 递归的 `*.head(...)`，其最内层 root receiver 满足以上两条之一

非法情况：
- 普通业务对象恰好也有 `.head(...)`
- 变量名叫 `result`，但不是由 `LG.result()` 得来

## 3.6.3 `ResultAttr` 字段填充

每个 `.head(...)` 生成一个 `ResultAttr`：

- `attr_name`：推荐用 head 名，如果静态不可解则回退为 `result_head@L{line}`
- `class_name`：owner class
- `def_loc`：`.head(...)` 调用行
- `head_name`：静态可解析时记录
- `prediction_expr` / `label_expr` / `loss_expr` / `sample_rate_expr`：统一保留 `fe._node_to_text(...)`
- `classifier_type`：如有则保留原文

## 3.6.4 链式 head 递归伪码

```python
def flatten_result_head_chain(call_node):
    chain = []
    cur = call_node
    while is_head_call(cur):
        chain.append(cur)
        receiver = cur.func.value
        if isinstance(receiver, ast.Call):
            cur = receiver
        else:
            break
    return list(reversed(chain))
```

```python
def is_result_head_root(receiver, result_seeds, lg_alias_ctx):
    if is_lg_result_call(receiver, lg_alias_ctx):
        return True
    if isinstance(receiver, ast.Name) and receiver.id in result_seeds:
        return True
    return False
```

```python
def scan_result_attrs(owner_class, fe, lg_alias_ctx):
    result_seeds = collect_result_seeds(owner_class, fe, lg_alias_ctx)
    out = []
    for method_name in {"forward"} | fe.get_reachable_helpers(owner_class, "forward"):
        for call in iter_calls(owner_class, method_name, fe):
            chain = flatten_result_head_chain(call)
            if not chain:
                continue
            root_receiver = resolve_chain_root(chain[0])
            if not is_result_head_root(root_receiver, result_seeds, lg_alias_ctx):
                continue
            for head_call in chain:
                out.append(build_result_attr(owner_class, fe, head_call))
    return dedup_results(out)
```

## 3.7 Phase 2 汇总伪码

```python
def scan_all_attrs(source_files, ast_frontends):
    attrs_by_class = convert_extractor_attrs(source_files, ast_frontends)
    class_map = _build_class_map(source_files, ast_frontends=ast_frontends)

    for (fname, owner_class), _info in class_map.items():
        fe = ast_frontends.get(fname)
        if fe is None:
            continue

        lg_alias_ctx = collect_lg_aliases(fe.tree)
        attrs_by_class[owner_class].extend(scan_input_attrs(owner_class, fe, lg_alias_ctx))
        attrs_by_class[owner_class].extend(scan_result_attrs(owner_class, fe, lg_alias_ctx))

    return sort_and_dedup(attrs_by_class)
```

> `scan_all_attrs()` 直接复用 `ASTFrontend + _AST_ModuleTreeExtractor + _build_class_map`；
> 它不负责 dep edge 计算，也不重新实现旧的静态树组装。

---

## 4. Phase 3：Node 构建层设计

## 4.1 入口函数签名

```python
def build_nodes(
    attrs_by_class: dict[str, list[Attr]],
) -> tuple[dict[int, DagNode], int]:
    """
    Returns: (registry: {node_id: DagNode}, next_id: int)
    """
```

## 4.2 节点映射规则

| Attr 类型 | DagNode 类型 | 说明 |
|---|---|---|
| `ModuleAttr` | `ModuleNode` | 普通业务子模块 |
| `ContainerAttr` | `ModuleNode` | 容器也是结构节点，需要参与 containment |
| `InputAttr` | `InputNode` | root / inner DAG 输入边界 |
| `ResultAttr` | `ResultNode` | root / inner DAG 输出边界 |
| synthetic root attr | `ModuleNode` | primary root endpoint |

**说明：** `ContainerAttr` 虽未在原始规则里单列，但实际必须建成 `ModuleNode`，否则容器层级无法表达。

## 4.3 `attr_id` 与 `node_id` 统一

建议在 Phase 3 直接采用：

```python
attr.attr_id = node_id
```

好处：
- 不再维护第二套 ID 空间
- `DagContext.loc_attr_index[(file, line, attr_id)]` 可直接映射到唯一节点
- 后续 timeline 定位更直观

## 4.4 伪码

```python
def build_nodes(attrs_by_class):
    registry = {}
    next_id = 1

    for owner_class in sorted(attrs_by_class.keys()):
        for attr in attrs_by_class[owner_class]:
            node_id = next_id
            next_id += 1
            attr.attr_id = node_id

            if isinstance(attr, (ModuleAttr, ContainerAttr)):
                node = ModuleNode(node_id=node_id, call_loc=attr.def_loc, attr=attr, inner_dag=None)
            elif isinstance(attr, InputAttr):
                node = InputNode(node_id=node_id, call_loc=attr.def_loc, attr=attr)
            elif isinstance(attr, ResultAttr):
                node = ResultNode(node_id=node_id, call_loc=attr.def_loc, attr=attr)
            else:
                raise TypeError(...)

            registry[node_id] = node

    return registry, next_id
```

## 4.5 root 层节点特殊处理

### 4.5.1 primary root 选取

桥接层沿用 `build_static_module_tree()` 的语义：
- `roots[0]` 视为 primary root
- 只对 primary root 注入 synthetic root attr
- 其余 roots 先不进入 `DagContext.root`

### 4.5.2 synthetic root attr 注入接口

```python
def inject_root_attr(
    attrs_by_class: dict[str, list[Attr]],
    roots: list[str],
    class_map: dict,
) -> dict[str, list[Attr]]:
    ...
```

伪码：

```python
def inject_root_attr(attrs_by_class, roots, class_map):
    if not roots:
        return attrs_by_class
    primary_root = roots[0]
    def_loc = lookup_class_def_loc(primary_root, class_map)
    attrs_by_class.setdefault("__graph__", []).append(
        ModuleAttr(
            attr_name="__root__",
            class_name=primary_root,
            def_loc=def_loc,
            source_expr=primary_root,
        )
    )
    return attrs_by_class
```

---

## 5. Phase 4：Edge 构建层设计

## 5.1 入口建议

用户只要求文档给 Phase 4 逻辑，本设计建议落一个统一入口：

```python
def build_edges(
    tree_dict: dict,
    attrs_by_class: dict[str, list[Attr]],
    registry: dict[int, DagNode],
    roots: list[str],
    ast_frontends: dict[str, ASTFrontend],
) -> None:
    """
    就地把 edge 信息组织成 DAG 装配所需结构。
    可返回 edges_by_class，也可暂存到局部变量供 Phase 5 使用。
    """
```

## 5.2 需要的索引

Phase 4 推荐先建立两个索引：

```python
attr_node_by_owner_and_name: dict[tuple[str, str], int]
root_node_id: int
```

构建方式：

```python
for owner_class, attrs in attrs_by_class.items():
    for attr in attrs:
        attr_node_by_owner_and_name[(owner_class, attr.attr_name)] = attr.attr_id
```

> 这里利用了 Phase 3 的 `attr.attr_id = node_id` 规则，因此不需要再反查 registry。

## 5.3 `dep_edges` → `DataFlowEdge`

### 5.3.1 输入格式

现有：

```python
tree_dict[class_name]["dep_edges"] == [(attr_a, attr_b), ...]
tree_dict[class_name]["dep_edge_locs"] == {
    (attr_a, attr_b): {"file": ..., "from_line": ..., "to_line": ...}
}
```

### 5.3.2 转换策略

- `src_id = attr_node_by_owner_and_name[(owner_class, attr_a)]`
- `dst_id = attr_node_by_owner_and_name[(owner_class, attr_b)]`
- `is_containment = False`
- `evidence = [VarEvidence(...)]`

### 5.3.3 Evidence 生成规则

最小桥接版只吃 `dep_edge_locs`：

```python
EvidenceStep(loc=from_loc, role="producer", var=attr_a)
EvidenceStep(loc=to_loc, role="consumer", var=attr_b)
```

封装为：

```python
VarEvidence(var=f"{attr_a}->{attr_b}", path_id=0, steps=[...])
```

### 5.3.4 伪码

```python
def build_dep_edge(owner_class, attr_a, attr_b, dep_edge_locs, attr_node_index):
    src_id = attr_node_index[(owner_class, attr_a)]
    dst_id = attr_node_index[(owner_class, attr_b)]
    loc_info = dep_edge_locs.get((attr_a, attr_b), {})

    steps = []
    if loc_info.get("file") and loc_info.get("from_line"):
        steps.append(EvidenceStep(loc=CallLoc(...from...), role="producer", var=attr_a))
    if loc_info.get("file") and loc_info.get("to_line"):
        steps.append(EvidenceStep(loc=CallLoc(...to...), role="consumer", var=attr_b))

    evidence = [VarEvidence(var=f"{attr_a}->{attr_b}", path_id=0, steps=steps)] if steps else []
    return DataFlowEdge(src_id=src_id, dst_id=dst_id, is_containment=False, evidence=evidence)
```

## 5.4 归属虚边（`is_containment=True`）

### 5.4.1 规则

每个非 root `ModuleNode` / `ContainerAttr` 节点都要有一条 parent → child 的 containment 边。

parent 的来源按优先级：
1. `attr.parent` 存在 → parent 是容器节点
2. owner class 是 primary root → parent 是 synthetic root node
3. 否则 parent 是“拥有该 owner class 的 ModuleNode”

### 5.4.2 第 3 条如何解

如果某节点属于 owner class `Tower`，则需要找到“哪个 `ModuleNode.attr.class_name == 'Tower'`”。

第一版建议：
- 允许 1:N，但 bridge 层先按**类名唯一**假设处理
- 若同一 child class 在多个位置实例化，后续 inner DAG / call-site disambiguation 再细化

即：

```python
module_instance_index: dict[str, list[int]]   # key = ModuleAttr.class_name
```

若 `module_instance_index[owner_class]`：
- 长度为 1 → 直接作为 containment parent
- 长度 > 1 → 当前桥接层选择全部挂到第一个实例，并记录 TODO/diagnostic

> 这是 bridge 阶段最需要后续收敛的地方，但不阻塞 root DAG 先落地。

### 5.4.3 伪码

```python
def build_containment_edge(parent_node_id, child_node_id):
    return DataFlowEdge(
        src_id=parent_node_id,
        dst_id=child_node_id,
        is_containment=True,
        evidence=[],
    )
```

## 5.5 Input / Result 边

### 5.5.1 目标

- 每个 `InputNode -> synthetic root ModuleNode`
- `synthetic root ModuleNode -> 每个 ResultNode`

### 5.5.2 evidence 策略

- **Input 边**：后续从 `collect_input_use_steps()` 派生
- **Result 边**：后续从 `prediction_expr` / `label_expr` / `loss_expr` / `sample_rate_expr` 派生
- 本阶段 bridge 落地允许 evidence 先为空列表

### 5.5.3 伪码

```python
def build_root_boundary_edges(root_node_id, attrs_by_class, attr_node_index, primary_root):
    edges = []
    for attr in attrs_by_class.get(primary_root, []):
        node_id = attr_node_index[(primary_root, attr.attr_name)]
        if isinstance(attr, InputAttr):
            edges.append(DataFlowEdge(src_id=node_id, dst_id=root_node_id, is_containment=False, evidence=[]))
        elif isinstance(attr, ResultAttr):
            edges.append(DataFlowEdge(src_id=root_node_id, dst_id=node_id, is_containment=False, evidence=[]))
    return edges
```

## 5.6 Phase 4 汇总伪码

```python
def build_edges(tree_dict, attrs_by_class, registry, roots, ast_frontends):
    primary_root = roots[0]
    attr_node_index = build_attr_node_index(attrs_by_class)
    root_node_id = attr_node_index[("__graph__", "__root__")]

    edges_by_class = defaultdict(list)

    # 1. dep edges per owner class
    for owner_class, info in tree_dict.items():
        for attr_a, attr_b in info.get("dep_edges", []):
            edge = build_dep_edge(owner_class, attr_a, attr_b, info.get("dep_edge_locs", {}), attr_node_index)
            if edge:
                edges_by_class[owner_class].append(edge)

    # 2. containment edges
    for owner_class, attrs in attrs_by_class.items():
        for attr in attrs:
            if not isinstance(attr, (ModuleAttr, ContainerAttr)):
                continue
            child_id = attr.attr_id
            parent_id = resolve_containment_parent(...)
            if parent_id is not None and parent_id != child_id:
                edges_by_class[owner_class].append(build_containment_edge(parent_id, child_id))

    # 3. root boundary edges
    edges_by_class[primary_root].extend(
        build_root_boundary_edges(root_node_id, attrs_by_class, attr_node_index, primary_root)
    )

    return edges_by_class
```

---

## 6. Phase 5：DagContext 装配设计

## 6.1 入口函数签名

```python
def assemble_dag_context(
    registry: dict[int, DagNode],
    roots: list[str],
    attrs_by_class: dict[str, list[Attr]],
    tree_dict: dict,
) -> DagContext:
```

## 6.2 root DAG 的定义

本阶段只保证 **primary root** 对应的 root DAG 装配正确：

- `inputs` = primary root owner class 下全部 `InputNode id`
- `outputs` = primary root owner class 下全部 `ResultNode id`
- `nodes` = primary root owner class 下全部 `ModuleNode id`（不含 synthetic root）
- `edges` = Phase 4 生成的 root 作用域边

synthetic root node 的处理：
- synthetic root `ModuleNode` 会出现在 `registry`
- 它的 `inner_dag = DagContext.root`
- `DagContext.root.nodes` **不包含** synthetic root node 本身

这是当前 `dag_types.py` 形状下最小侵入的方案。

## 6.3 装配步骤

### 步骤 1：拿到 primary root

```python
primary_root = roots[0]
```

### 步骤 2：按 owner class 拿 node ids

```python
root_input_ids = [...]   # attrs_by_class[primary_root] 中 InputAttr.attr_id
root_output_ids = [...]  # ResultAttr.attr_id
root_module_ids = [...]  # ModuleAttr/ContainerAttr.attr_id，不含 __graph__/__root__
```

### 步骤 3：构造 root DAG

```python
root_dag = DAG(
    inputs=root_input_ids,
    outputs=root_output_ids,
    nodes=root_module_ids,
    edges=root_edges,
)
```

### 步骤 4：回填 root node.inner_dag

```python
root_node = registry[root_node_id]
root_node.inner_dag = root_dag
```

### 步骤 5：构造 `DagContext`

```python
return DagContext(registry=registry, root=root_dag)
```

`loc_attr_index` 由 `DagContext.__post_init__` 自动建立，无需手工填。

## 6.4 伪码

```python
def assemble_dag_context(registry, roots, attrs_by_class, tree_dict):
    primary_root = roots[0]
    root_node = find_synthetic_root_node(registry)

    root_inputs = [a.attr_id for a in attrs_by_class.get(primary_root, []) if isinstance(a, InputAttr)]
    root_outputs = [a.attr_id for a in attrs_by_class.get(primary_root, []) if isinstance(a, ResultAttr)]
    root_modules = [
        a.attr_id
        for a in attrs_by_class.get(primary_root, [])
        if isinstance(a, (ModuleAttr, ContainerAttr))
    ]

    root_edges = collect_root_edges(...)
    root_dag = DAG(inputs=root_inputs, outputs=root_outputs, nodes=root_modules, edges=root_edges)
    root_node.inner_dag = root_dag
    return DagContext(registry=registry, root=root_dag)
```

---

## 7. 新旧桥接层设计

## 7.1 桥接函数签名

```python
def build_dag_context_from_source(
    source_files: dict[str, list[str]],
    conditional_mode: str = "train",
) -> DagContext:
```

## 7.2 设计要求

- 内部严格按 **Phase 1 → 5** 顺序串联
- **只复用现有 AST 能力**，不在桥接层写新的 AST walk 逻辑
- `build_static_module_tree()` 暂时保留；HTML 旧路径仍可继续依赖
- 最终目标是让调用方逐步从 `(tree_dict, roots, class_map)` 迁到 `DagContext`

## 7.3 伪码

```python
def build_dag_context_from_source(source_files, conditional_mode="train"):
    tree_dict, roots, class_map = build_static_module_tree(
        source_files,
        conditional_mode=conditional_mode,
    )
    ast_frontends = _build_ast_frontends(source_files)
    attrs_by_class = scan_all_attrs(source_files, ast_frontends)
    attrs_by_class = inject_root_attr(attrs_by_class, roots, class_map)
    registry, _ = build_nodes(attrs_by_class)
    edges_by_class = build_edges(tree_dict, attrs_by_class, registry, roots, ast_frontends)
    return assemble_dag_context(registry, roots, attrs_by_class, tree_dict)
```

## 7.4 迁移策略说明

### 当前阶段

- `build_static_module_tree()`：继续保留
- 新增 `build_dag_context_from_source()`：作为新 IR bridge

### 后续阶段

- 前端 HTML / 新 lineage / 新高亮逻辑逐步改读 `DagContext`
- 当调用方全部迁移完成后，再考虑下线旧 tree dict 接口

---

## 8. 分模块文件规划

```text
scripts/
  attr_scanner.py     # Phase 2：scan_all_attrs（重写现有原型）
  dag_builder.py      # Phase 2.5 + 3 + 4 + 5：inject_root_attr / build_nodes / build_edges / assemble_dag_context / build_dag_context_from_source
  dag_types.py        # 已有，不变
  attr_types.py       # 已有，不变
```

## 8.1 `attr_scanner.py` 规划

职责：
- LG alias 识别
- Module/Container Attr 标准化
- InputAttr 扫描
- ResultAttr 扫描
- 排序/去重

预计行数：300~400 行

**控制点：**
- 不负责 node/edge/DAG 装配
- 不直接依赖前端 HTML

## 8.2 `dag_builder.py` 规划

职责：
- root attr 注入
- `Attr -> DagNode`
- `dep_edges / containment / boundary edges`
- `DagContext` 装配
- 对外 bridge：`build_dag_context_from_source()`

预计行数：350~450 行

## 8.3 若 `dag_builder.py` 超 500 行的拆分点

优先拆出：

```text
scripts/
  dag_builder.py        # public bridge + orchestration
  dag_node_builder.py   # build_nodes / index helpers
  dag_edge_builder.py   # dep_edges / containment / boundary edges
```

但第一落地建议先放在一个 `dag_builder.py` 中，等 Phase 3-5 稳定后再拆。

---

## 9. UT 测例框架设计（只设计，不实现）

## 9.1 Phase 2 - Attr 扫描

### Phase 2 - InputAttr direct dense

- **mock**：单个 `MyModel`，`__init__` 里 `self.x = LG.dense_feature('f', 1, torch.int64)`，`forward` 里 `self.x()`
- **断言**：`attrs_by_class["MyModel"]` 包含 `InputAttr(attr_name="x", kind="dense_feature", def_loc.line=N)`

### Phase 2 - Sample rate direct callable

- **mock**：`self.sample_rate = LG.get_sample_rate()`，`forward` 里 `self.sample_rate()`
- **断言**：存在 `InputAttr(kind="get_sample_rate")`

### Phase 2 - Feature column local var

- **mock**：`fc = LG.feature_column(...)`；`forward` 里 `fc.get_vector(slc)`
- **断言**：存在 `InputAttr(kind="feature_column")`，且 `source_expr` 含 `feature_column`

### Phase 2 - Feature column dict slot

- **mock**：`fc_dict[2] = LG.feature_column(...)`；`forward` 里 `fc_dict[2].get_size_tensor()`
- **断言**：存在 `InputAttr(kind="feature_column")`，`slot_expr == "2"`（或等价表达）

### Phase 2 - LG alias import-from

- **mock**：`from bytedance.lagrange_torch.context import CONTEXT as M`；`self.x = M.dense_feature(...)`
- **断言**：仍能扫出 `InputAttr(kind="dense_feature")`

### Phase 2 - LG alias import module

- **mock**：`import bytedance.lagrange_torch.context as lg_ctx`；`self.x = lg_ctx.CONTEXT.dense_feature(...)`
- **断言**：仍能扫出 `InputAttr`

### Phase 2 - Result alias

- **mock**：`run_train = LG.result()`；`run_train.head(name='ctr', prediction=pred)`
- **断言**：存在 `ResultAttr(head_name="ctr", prediction_expr="pred")`

### Phase 2 - Chained head

- **mock**：`LG.result().head(name='a', prediction=x).head(name='b', prediction=y)`
- **断言**：返回 2 个 `ResultAttr`，head 分别为 `a` / `b`

### Phase 2 - Negative case non-LG head

- **mock**：业务对象 `obj.head(prediction=x)`，但 `obj` 不是 `LG.result()`
- **断言**：不生成 `ResultAttr`

### Phase 2 - Module / Container conversion

- **mock**：`self.blocks = nn.ModuleList(); self.blocks.append(Block())`
- **断言**：`attrs_by_class["Root"]` 同时包含 `ContainerAttr(attr_name="blocks")` 和 `ModuleAttr(attr_name="blocks[0]")`

## 9.2 Phase 3 - Node 构建

### Phase 3 - Attr to node type mapping

- **mock**：手工构造 `ModuleAttr / ContainerAttr / InputAttr / ResultAttr`
- **断言**：映射后类型分别是 `ModuleNode / ModuleNode / InputNode / ResultNode`

### Phase 3 - Global unique node ids

- **mock**：两个 class，分别含多个 Attr
- **断言**：`node_id` 严格递增且无重复

### Phase 3 - attr_id equals node_id

- **mock**：构造 3 个 Attr
- **断言**：`attr.attr_id == registry[attr.attr_id].node_id`

### Phase 3 - synthetic root attr

- **mock**：`roots = ["RootModel"]`
- **断言**：注入 `ModuleAttr(attr_name="__root__", class_name="RootModel")`，且 build_nodes 后得到一个 root `ModuleNode`

## 9.3 Phase 4 - Edge 构建

### Phase 4 - dep edge translation

- **mock**：`tree_dict["Root"]["dep_edges"] = [("tower", "head")]`，并提供 `dep_edge_locs`
- **断言**：生成 `DataFlowEdge(src=tower_id, dst=head_id, is_containment=False)`，且 evidence 非空

### Phase 4 - containment edge direct child

- **mock**：root class 下 `tower` / `head` 两个 ModuleAttr
- **断言**：`synthetic_root -> tower`、`synthetic_root -> head` 各有一条 `is_containment=True` 边

### Phase 4 - containment edge container child

- **mock**：`blocks` 是 `ContainerAttr`，`blocks[0]` 是子 ModuleAttr
- **断言**：存在 `blocks -> blocks[0]` 的 containment 边

### Phase 4 - input boundary edge

- **mock**：`Root` 中含一个 `InputAttr`
- **断言**：存在 `InputNode -> synthetic_root` 边

### Phase 4 - result boundary edge

- **mock**：`Root` 中含两个 `ResultAttr`
- **断言**：存在 `synthetic_root -> ResultNode` 两条边

## 9.4 Phase 5 - DagContext 装配

### Phase 5 - root dag membership

- **mock**：`Root` 类下有 1 个 Input、2 个 Module、1 个 Result
- **断言**：`DagContext.root.inputs == [input_id]`；`nodes == [module_ids...]`；`outputs == [result_id]`

### Phase 5 - root node inner_dag binding

- **mock**：已注入 synthetic root node
- **断言**：synthetic root `ModuleNode.inner_dag is DagContext.root`

### Phase 5 - loc_attr_index build

- **mock**：至少 3 个不同 `CallLoc`
- **断言**：`DagContext.loc_attr_index[(file, line, attr_id)] == node_id`

---

## 10. 风险与后续工作

## 10.1 当前 bridge 方案的已知限制

1. **多实例同类 owner 的 containment 归属仍可能歧义**
   - 例如同一 `Tower` class 被实例化多次
   - 当前 bridge 先按“类名唯一实例”或“第一个实例”处理

2. **Input / Result evidence 先允许为空**
   - 先把 DAG 结构接通
   - lineage_by_carrier 在后续任务补齐

3. **synthetic root node 不在 `DagContext.root.nodes` 中**
   - 这是当前 `dag_types.py` 形状下的桥接折中
   - 后续如需更严格 IR，可增 `root_node_id`

## 10.2 下一步实现顺序建议

1. 先重写 `attr_scanner.py`
2. 再落 `dag_builder.py` 的 root 注入 + node 构建
3. 再接 `dep_edges` / containment
4. 最后接 `Input/Result` boundary edges
5. 补 Phase 2~5 对应 UT，并按迭代流程跑 UT + 覆盖率

---

## 11. 最终建议

本设计的核心点不是“再造一套 AST 逻辑”，而是：

- **复用** `ASTFrontend`
- **复用** `_AST_ModuleTreeExtractor`
- **复用** `build_static_module_tree()` 的 `dep_edges / roots / attr_def_loc`
- 用一个最小 bridge，把旧树结构平滑映射到新 DAG IR

这样可以在不破坏现有 HTML 路径的前提下，先把后端 IR 站稳，再逐步把高亮、lineage、timing 等能力统一到 `DagContext` 上。