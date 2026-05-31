## AttrScanner 设计：ASTFrontend 三层工具集 + AttrScanner 语义层

本设计文档描述 AttrScanner 与 ASTFrontend 的分层架构，以及 ASTFrontend 三层工具集的接口设计、关键约束和 UT 方案。当前工作分支为 `feat/ast-timing-fix`，对应源码位于 `scripts/ast_frontend.py` 和 `scripts/attr_scanner.py` 等文件中。

---

### 1. 分层结构总览

整体分层如下：

- **Layer 0 — ASTFrontend：node 级原子提取**
- **Layer 1 — ASTFrontend：stmt 解构**
- **Layer 2 — ASTFrontend：pattern 匹配（所有 pattern 由调用者传入）**
- **Layer 3 — AttrScanner：语义决策层（持有 LG / Container 等业务语义常量）**

#### 1.1 Layer 0：ASTFrontend node 级原子提取

Layer 0 提供对单个 AST 节点的最小粒度原子信息提取能力，不涉及语句级结构，更不感知任何业务语义。典型接口如下：

- `_node_loc(node) -> CallLoc`
- `_node_to_text(node) -> str`
- `_extract_self_attr_name(target) -> Optional[str]`
- `_expr_leaf_name(node) -> Optional[str]`

这些函数只关心：
- 源码位置信息（行号、列号、文件路径等）
- AST 表达式结构到字符串的转写
- `self.x` 这类属性访问的属性名提取
- 任意表达式的“叶子名”（如 `pkg.mod.Class` → `Class`）

#### 1.2 Layer 1：ASTFrontend stmt 解构

Layer 1 提供对语句级结构的统一解构接口，仍不带任何业务语义，仅负责把「一条语句」拆解成更易消费的结构：

- `stmt_assign_target(stmt) -> Optional[ast.expr]`
- `stmt_assign_call_value(stmt) -> Optional[ast.Call]`
- `call_func_text(call) -> str`
- `call_kwarg_texts(call) -> dict[str, str]`
- `call_arg_texts(call) -> list[str]`
- `iter_init_stmts(class_name) -> Iterator[ast.stmt]`

其中 `iter_init_stmts` 为新增接口，语义为：

> 递归遍历给定 `class_name` 的 `__init__` 方法体内所有语句，包括 for/if/with/try 等所有控制流分支，调用方只需顺序遍历而不关心嵌套层级和具体控制流结构。

#### 1.3 Layer 2：ASTFrontend pattern 匹配

Layer 2 在 Layer 0/1 提供的原子信息和 stmt 解构之上，提供通用的 pattern 匹配帮助函数，用于匹配“函数调用形态”的语句/表达式。**所有 pattern 参数均由调用者显式传入**，ASTFrontend 本身不硬编码任何业务名。

接口列表：

- `match_self_attr_call_assign(stmt, *, func_prefixes=None, func_names=None) -> Optional[tuple[str, ast.Call]]`
  - 匹配形如 `self.attr = <func>(...)` 的赋值语句。
  - 仅在 `<func>` 满足调用方给定的 `func_prefixes` 或 `func_names` pattern 时返回匹配结果。
  - `func_prefixes: set[str]`，匹配 `func_text.startswith(any)`。
  - `func_names: set[str]`，匹配 `func_leaf in set`（其中 `func_leaf` 通常来自 `_expr_leaf_name(call.func)`）。
  - 当二者均为 `None` 时，只检查结构是否为 `self.attr = <func>(...)`，不做函数名过滤。

- `match_call_by_func(call, *, func_prefixes=None, func_names=None) -> Optional[str]`
  - 对单个 `ast.Call` 节点，根据 `func_prefixes` / `func_names` 进行匹配。
  - 返回完整的 `func_text`（如 `LG.Input`, `nn.ModuleList` 等），不匹配返回 `None`。

- `match_expr_or_stmt_call(node, *, func_attr=None) -> Optional[tuple[str, ast.Call]]`
  - 在表达式或语句粒度上寻找函数调用：支持 `Expr(Call)` / `Assign(Call)` / `Return(Call)` 三类节点。
  - 要求 `call.func.attr == func_attr` 时才视为匹配。
  - 返回 `(owner_text, call_node)`，其中 `owner_text` 通常是 `call.func.value` 的文本表示（例如 `LG`, `self` 等）。

#### 1.4 Layer 3：AttrScanner 语义决策层

AttrScanner 作为语义层，**唯一持有业务语义常量**，包括 LG 前缀、容器类名等。ASTFrontend 完全不知道这些常量的具体含义。

核心常量示例：

```python
_LG_PREFIXES          = {"LG.", "LG2."}
_CONTAINER_LEAF_NAMES = {"ModuleList", "ModuleDict", "Sequential"}
```

主要职责函数：

- `_scan_input_attrs(class_name, fe) -> list[InputAttr]`
- `_scan_module_attrs(class_name, fe) -> list[ModuleAttr | ContainerAttr]`
- `_scan_result_attrs(class_name, fe) -> list[ResultAttr]`
- `_register_attr(attr, owner_class)`：分配 `attr_id`，写入 `attrs_by_id` 等全局/上下文字典

AttrScanner 通过组合 Layer 0/1/2 提供的工具，完成具体的 LG.Input / 模块容器 / 结果节点等业务语义识别。

---

### 2. 关键约束与迁移要求

本节总结 ASTFrontend 与 AttrScanner 的协议约束，以及从旧实现迁移时必须遵守的规则。

#### 2.1 ASTFrontend 不感知业务语义

- ASTFrontend **禁止**硬编码以下任意业务名或前缀：
  - LG / LG2 / LocalGraph / LocalGraphV2 等所有 LG 系列名称
  - nn.ModuleList / nn.ModuleDict / nn.Sequential 等容器类名
  - 任何特定模型、特定项目中的自定义 Wrapper 名称
- 所有与业务相关的 pattern（例如 LG 前缀、容器类名、特定 API 名）必须由 AttrScanner 或 DagBuilder 等上层模块以 `func_prefixes` / `func_names` / `func_attr` 等参数传入。

#### 2.2 Pattern 参数由调用者传入

- Layer 2 中涉及 pattern 匹配的函数：
  - `match_self_attr_call_assign`
  - `match_call_by_func`
  - `match_expr_or_stmt_call`
- 这些函数只能根据调用方提供的参数执行匹配逻辑，不得内置任何业务常量。
- AttrScanner 负责维护：
  - LG 函数前缀集合 `_LG_PREFIXES`
  - 容器类叶子名集合 `_CONTAINER_LEAF_NAMES`
  - 其他可能出现的业务函数名表（例如 LG.Input / LG.Result 的枚举表等）

#### 2.3 parse_local_* 系列废弃

- 既有实现中以 `parse_local_*` 命名的一系列函数（例如 `parse_local_input`, `parse_local_result` 等）视为**废弃接口**。
- 所有原本依赖 `parse_local_*` 的能力，必须通过 Layer 1 + Layer 2 的组合能力实现：
  - 先用 `iter_init_stmts` / `stmt_assign_*` 等拿到语句结构
  - 再用 `match_*` 系列函数做 pattern 匹配
- `parse_local_*` 函数保留期仅用于过渡阶段，最终需要彻底删除，避免新逻辑继续依赖。

#### 2.4 _AST_ModuleTreeExtractor 遍历逻辑下沉

- 旧实现 `_AST_ModuleTreeExtractor._walk_init_body` 中负责遍历 `__init__` 方法体、穿透 for/if/with/try 等控制流的逻辑，统一沉入 `ASTFrontend.iter_init_stmts`。
- 迁移目标是：
  - `_AST_ModuleTreeExtractor` 不再持有遍历细节，只消费 `iter_init_stmts` 提供的统一语句流。
  - `AttrScanner` 在扫描 InputAttr/ModuleAttr/ContainerAttr/ResultAttr 时，同样使用 `iter_init_stmts`，避免重复实现遍历逻辑。

#### 2.5 Edge 相关字段暂留 ResultAttr

- ResultAttr 中现有的 Edge 相关字段：
  - `prediction_expr`
  - `label_expr`
  - `sample_rate_expr`
  - `loss_expr`
- 在当前阶段，这些字段仍暂留在 ResultAttr 上，作为从 AttrScanner 向后续阶段（dag_builder / edge evidence）传递信息的临时载体。
- 最终是否迁出到专门的 edge 结构体，由 dag_builder 阶段统一决策；在该决策落地前，不对字段语义做大的调整，只保证现有信息完整传递。

---

### 3. AttrScanner 调用示例

本节给出 AttrScanner 基于 ASTFrontend 的典型调用示例，帮助统一编码风格。

#### 3.1 扫描 InputAttr

```python
# 扫描 InputAttr
for stmt in fe.iter_init_stmts(class_name):
    r = fe.match_self_attr_call_assign(stmt, func_prefixes=self._LG_PREFIXES)
    if r is None:
        continue
    attr_name, call = r
    kind = LG_FUNC_KINDS.get(fe.call_func_text(call))
    loc = fe._node_loc(stmt)
    ...
```

要点说明：
- 遍历入口统一使用 `fe.iter_init_stmts(class_name)`，不再自己写 `_walk_init_body`。
- 使用 `match_self_attr_call_assign(..., func_prefixes=self._LG_PREFIXES)` 做 LG.Input / LG.Result 等的粗匹配。
- 使用 `LG_FUNC_KINDS` 这类业务表，从 `call_func_text` 映射到输入类型（如 input / label / prediction 等）。
- 位置信息统一从 `fe._node_loc(stmt)` 获取，后续写入 ResultAttr / InputAttr。

#### 3.2 扫描 ContainerAttr

```python
# 扫描 ContainerAttr
for stmt in fe.iter_init_stmts(class_name):
    r = fe.match_self_attr_call_assign(stmt, func_names=self._CONTAINER_LEAF_NAMES)
    if r is None:
        continue
    attr_name, call = r
    kind = fe._expr_leaf_name(call.func)
    ...
```

要点说明：
- 这里使用 `func_names=self._CONTAINER_LEAF_NAMES` 直接做叶子名匹配，如 `ModuleList` / `ModuleDict` / `Sequential`。
- 具体容器类型 `kind` 通过 `_expr_leaf_name(call.func)` 提取，并存入 `ContainerAttr` 的 `kind` 或类似字段。
- 若未来新增容器类型，只需在 `_CONTAINER_LEAF_NAMES` 中扩展集合，而无需修改 ASTFrontend。

---

### 4. UT 方案（仅设计，不写代码）

本节在文档末尾给出 UT 方案，覆盖 Layer 0 / Layer 1 / Layer 2 的关键函数。目标：

- line coverage ≥ 90%
- branch coverage ≥ 50%

UT 预计落地到 `storage` 仓库的 `testset/unit/` 目录，命名建议：`test_ast_frontend_attr_scanner.py` 或拆分为多个模块测试文件。

#### 4.1 UT 总体策略

- 对于简单结构、固定模式的 AST 节点，优先使用 `ast.parse` 从 code snippet 生成 AST，便于阅读和维护。
- 对于难以通过简单代码片段构造或需要精确控制字段的节点（如带特定 `lineno` / `col_offset` 的节点），可使用 synthetic AST 直接构造。
- 测试重点放在：
  - 输入 shape 覆盖（不同语句类型 / 控制流 / 函数名 pattern）
  - True/False 分支覆盖（匹配成功/失败，参数为空/非空）
  - 异常/边界情况（None / 空列表 / 非期望节点类型）

---

#### 4.2 Layer 0：`_node_loc`

##### 4.2.1 测例列表

<table header-row="true" header-col="false" col-widths="80,120,220,220,180">
  <tr>
    <td>测例 ID</td>
    <td>测试函数名</td>
    <td>输入</td>
    <td>预期输出</td>
    <td>覆盖目标</td>
  </tr>
  <tr>
    <td>L0-LOC-01</td>
    <td>_node_loc</td>
    <td>通过 `ast.parse("x = 1", filename="foo.py", mode="exec")` 取第一条语句节点 `stmt`，确保 `lineno`/`col_offset` 存在。</td>
    <td>返回 `CallLoc` 对象，`filename == "foo.py"`，`lineno == 1`，`col_offset` 等于 `x` 的起始列；不抛异常。</td>
    <td>基础路径：正常节点位置信息提取；验证文件名和行号映射逻辑。</td>
  </tr>
  <tr>
    <td>L0-LOC-02</td>
    <td>_node_loc</td>
    <td>synthetic AST：手动构造一个最小节点（如 `ast.Name(id="x")`），显式设置 `lineno`/`col_offset`/`end_lineno`/`end_col_offset`，并传入 `_node_loc`。</td>
    <td>返回的 `CallLoc` 中行列号与手动设置的值完全一致，特别是 `end_lineno`/`end_col_offset` 分支被覆盖。</td>
    <td>覆盖 `end_lineno`/`end_col_offset` 存在的分支逻辑，确保 branch 覆盖。</td>
  </tr>
  <tr>
    <td>L0-LOC-03</td>
    <td>_node_loc</td>
    <td>synthetic AST：构造不带位置信息的节点（不设置 `lineno`/`col_offset`），例如 `ast.Pass()` 默认状态。</td>
    <td>根据实现预期：要么返回带默认值的 `CallLoc`（如 `lineno == 0`），要么触发内部 fallback 分支；UT 只需断言不抛异常且行为稳定（例如返回某个约定默认值）。</td>
    <td>覆盖「缺失位置信息」分支，确保任何节点都能安全处理。</td>
  </tr>
</table>

##### 4.2.2 AST 构造方式说明

- L0-LOC-01：使用 `ast.parse` 生成。
- L0-LOC-02 / L0-LOC-03：使用 synthetic AST 直接构造节点，以精确控制字段是否存在。

---

#### 4.3 Layer 1：`iter_init_stmts`

##### 4.3.1 覆盖目标

- 递归遍历 `__init__` 方法体中的所有语句：
  - 顶层简单赋值
  - for 循环体
  - if 条件分支（`body` + `orelse`）
  - with 语句体
  - try/except/finally 块
- 确保所有控制流分支内的语句都会被 `iter_init_stmts` 产出一次且仅一次。

##### 4.3.2 测例列表

<table header-row="true" header-col="false" col-widths="80,120,220,220,180">
  <tr>
    <td>测例 ID</td>
    <td>测试函数名</td>
    <td>输入</td>
    <td>预期输出</td>
    <td>覆盖目标</td>
  </tr>
  <tr>
    <td>L1-INIT-01</td>
    <td>iter_init_stmts</td>
    <td>code snippet 方式：

```python
class A:
    def __init__(self):
        self.x = 1
```

通过 `ast.parse` 获取 ClassDef，并由 ASTFrontend 构造 frontend 对象。</td>
    <td>`list(iter_init_stmts("A"))` 只包含一条 `Assign` 语句，对应 `self.x = 1`。</td>
    <td>基础路径：无控制流，仅顶层语句遍历。</td>
  </tr>
  <tr>
    <td>L1-INIT-02</td>
    <td>iter_init_stmts</td>
    <td>code snippet：

```python
class A:
    def __init__(self):
        for i in range(3):
            self.x = i
```
</td>
    <td>迭代结果中包含一条来自 for-body 的 `Assign` 语句；可通过 `node.lineno` 或 `_node_to_text` 验证实际返回的语句内容。</td>
    <td>覆盖 for 分支遍历逻辑。</td>
  </tr>
  <tr>
    <td>L1-INIT-03</td>
    <td>iter_init_stmts</td>
    <td>code snippet：

```python
class A:
    def __init__(self):
        if flag:
            self.x = 1
        else:
            self.y = 2
```
</td>
    <td>产出的语句序列中同时包含 `self.x = 1` 和 `self.y = 2` 两条 Assign，顺序与源代码出现顺序一致（先 if-body 后 else-body）。</td>
    <td>覆盖 if `body` 和 `orelse` 两路分支。</td>
  </tr>
  <tr>
    <td>L1-INIT-04</td>
    <td>iter_init_stmts</td>
    <td>code snippet：

```python
class A:
    def __init__(self):
        with ctx:
            self.x = 1
```
</td>
    <td>产出 `with` 语句体内的 `self.x = 1` Assign。</td>
    <td>覆盖 with 语句遍历逻辑。</td>
  </tr>
  <tr>
    <td>L1-INIT-05</td>
    <td>iter_init_stmts</td>
    <td>code snippet：

```python
class A:
    def __init__(self):
        try:
            self.x = 1
        except Exception:
            self.y = 2
        finally:
            self.z = 3
```
</td>
    <td>遍历结果中包含 `self.x = 1`、`self.y = 2`、`self.z = 3` 三条 Assign。</td>
    <td>覆盖 try-body / except / finally 三路分支遍历。</td>
  </tr>
  <tr>
    <td>L1-INIT-06</td>
    <td>iter_init_stmts</td>
    <td>code snippet：`class A: pass`（无 `__init__`），或 `__init__` 为空 `pass`。</td>
    <td>`list(iter_init_stmts("A"))` 为空列表。</td>
    <td>覆盖「无初始化语句」路径，确保不会抛异常。</td>
  </tr>
</table>

##### 4.3.3 AST 构造方式说明

- 所有 L1-INIT-* 测例均可通过 `ast.parse` 生成完整模块 AST，再由 ASTFrontend 包装获取 class 级别视图，无需 synthetic AST。

---

#### 4.4 Layer 2：`match_self_attr_call_assign`

##### 4.4.1 覆盖目标

- 四类 pattern 组合路径：
  1. 仅 `func_prefixes` 非空
  2. 仅 `func_names` 非空
  3. `func_prefixes` 和 `func_names` 同时非空
  4. 二者均为 `None`
- 每类路径下还需要覆盖匹配成功和不匹配两路。

##### 4.4.2 测例列表

<table header-row="true" header-col="false" col-widths="80,150,260,240,200">
  <tr>
    <td>测例 ID</td>
    <td>测试函数名</td>
    <td>输入</td>
    <td>预期输出</td>
    <td>覆盖目标</td>
  </tr>
  <tr>
    <td>L2-SELF-01</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = LG.Input("a")` 解析为 Assign 语句；调用 `match_self_attr_call_assign(stmt, func_prefixes={"LG."})`。</td>
    <td>返回 `("x", call)`，其中 `call` 是 `LG.Input("a")` 对应的 `ast.Call`；不为 None。</td>
    <td>覆盖 prefix 匹配成功路径；验证 `func_text.startswith` 分支。</td>
  </tr>
  <tr>
    <td>L2-SELF-02</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = Other.Input("a")`；调用 `func_prefixes={"LG."}`。</td>
    <td>返回 `None`。</td>
    <td>覆盖 prefix 不匹配路径。</td>
  </tr>
  <tr>
    <td>L2-SELF-03</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = nn.ModuleList([])`；调用 `match_self_attr_call_assign(stmt, func_names={"ModuleList", "Sequential"})`。</td>
    <td>返回 `("x", call)`，匹配成功。</td>
    <td>覆盖 name 匹配成功路径；验证 `_expr_leaf_name` 与 `func_names` 交互。</td>
  </tr>
  <tr>
    <td>L2-SELF-04</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = nn.Other([])`；调用 `func_names={"ModuleList"}`。</td>
    <td>返回 `None`。</td>
    <td>覆盖 name 不匹配路径。</td>
  </tr>
  <tr>
    <td>L2-SELF-05</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = LG2.Input("a")`；调用 `func_prefixes={"LG."}`, `func_names={"Input"}`。</td>
    <td>根据实际实现：若逻辑为「prefix/leaf 任一命中即可」，则返回非 None；UT 中需要根据现有实现确认期望，并同时覆盖两路条件判断。</td>
    <td>覆盖 prefix + name 同时配置时的分支逻辑；确保两个集合都被访问到。</td>
  </tr>
  <tr>
    <td>L2-SELF-06</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`self.x = something()`；调用 `match_self_attr_call_assign(stmt, func_prefixes=None, func_names=None)`。</td>
    <td>只要语句结构为 `self.attr = func(...)`，即返回 `("x", call)`，不关心函数名。</td>
    <td>覆盖「仅结构检查」路径。</td>
  </tr>
  <tr>
    <td>L2-SELF-07</td>
    <td>match_self_attr_call_assign</td>
    <td>code snippet：`x = LG.Input("a")`（左侧非 self.attr），调用任意 pattern。</td>
    <td>返回 `None`。</td>
    <td>覆盖「非 self.attr 赋值」路径。</td>
  </tr>
</table>

##### 4.4.3 AST 构造方式说明

- 所有 L2-SELF-* 测例均可用 `ast.parse` 解析简单函数或方法体片段，提取对应的 Assign 语句。无需 synthetic AST。

---

#### 4.5 Layer 2：`match_call_by_func`

##### 4.5.1 覆盖目标

- 仅 prefix 模式、仅 name 模式、二者均配置、均为 None 四种路径。
- 匹配成功与失败两路。

##### 4.5.2 测例列表

<table header-row="true" header-col="false" col-widths="80,150,260,240,200">
  <tr>
    <td>测例 ID</td>
    <td>测试函数名</td>
    <td>输入</td>
    <td>预期输出</td>
    <td>覆盖目标</td>
  </tr>
  <tr>
    <td>L2-CALL-01</td>
    <td>match_call_by_func</td>
    <td>code snippet：`LG.Input("a")` 作为表达式，解析 `Expr` 后取其 `value` 作为 `ast.Call`。</td>
    <td>调用 `match_call_by_func(call, func_prefixes={"LG."})`，返回 `"LG.Input"`。</td>
    <td>覆盖 prefix 匹配成功路径。</td>
  </tr>
  <tr>
    <td>L2-CALL-02</td>
    <td>match_call_by_func</td>
    <td>同上 `LG.Input("a")`。</td>
    <td>调用 `match_call_by_func(call, func_names={"Input"})`，返回 `"LG.Input"`。</td>
    <td>覆盖 name 匹配成功路径。</td>
  </tr>
  <tr>
    <td>L2-CALL-03</td>
    <td>match_call_by_func</td>
    <td>code snippet：`Other.Input("a")`。</td>
    <td>调用 `func_prefixes={"LG."}`，返回 `None`。</td>
    <td>prefix 不匹配路径。</td>
  </tr>
  <tr>
    <td>L2-CALL-04</td>
    <td>match_call_by_func</td>
    <td>code snippet：`nn.ModuleList([])`。</td>
    <td>调用 `func_names={"Sequential"}`，返回 `None`。</td>
    <td>name 不匹配路径。</td>
  </tr>
  <tr>
    <td>L2-CALL-05</td>
    <td>match_call_by_func</td>
    <td>code snippet：`LG2.Input("a")`；调用 `func_prefixes={"LG."}, func_names={"Input"}`。</td>
    <td>与 L2-SELF-05 一致，根据现有实现确定组合逻辑（AND / OR），并断言返回值符合预期。</td>
    <td>覆盖 prefix + name 同时配置路径。</td>
  </tr>
  <tr>
    <td>L2-CALL-06</td>
    <td>match_call_by_func</td>
    <td>任意 `call`，调用 `match_call_by_func(call, func_prefixes=None, func_names=None)`。</td>
    <td>根据实现：若逻辑为「无过滤时全部通过」，则直接返回 `func_text`；UT 中按实际实现断言。</td>
    <td>覆盖「无过滤参数」路径。</td>
  </tr>
</table>

##### 4.5.3 AST 构造方式说明

- 所有 L2-CALL-* 测例均可通过 `ast.parse` 生成表达式，再取 `Expr.value` 作为 `ast.Call` 传入。

---

#### 4.6 Layer 2：`match_expr_or_stmt_call`

##### 4.6.1 覆盖目标

- 覆盖 Expr / Assign / Return 三种节点类型：
  - `Expr(LG.foo(...))`
  - `x = LG.foo(...)`
  - `return LG.foo(...)`
- 覆盖 `func_attr` 匹配成功与失败（例如 `func_attr="foo"` 时成功，`func_attr="bar"` 时失败）。

##### 4.6.2 测例列表

<table header-row="true" header-col="false" col-widths="80,180,260,240,200">
  <tr>
    <td>测例 ID</td>
    <td>测试函数名</td>
    <td>输入</td>
    <td>预期输出</td>
    <td>覆盖目标</td>
  </tr>
  <tr>
    <td>L2-EXPR-01</td>
    <td>match_expr_or_stmt_call</td>
    <td>code snippet：`LG.foo(x)` 作为独立表达式语句 `Expr`。</td>
    <td>调用 `match_expr_or_stmt_call(expr_node, func_attr="foo")`，返回 `("LG", call)`，其中 `call` 对应 `LG.foo(x)`。</td>
    <td>覆盖 Expr 节点匹配成功路径；验证 owner_text 和 call 返回值。</td>
  </tr>
  <tr>
    <td>L2-EXPR-02</td>
    <td>match_expr_or_stmt_call</td>
    <td>code snippet：`y = LG.foo(x)` 作为 Assign 语句。</td>
    <td>同样调用 `func_attr="foo"`，返回 `("LG", call)`。</td>
    <td>覆盖 Assign 节点匹配成功路径。</td>
  </tr>
  <tr>
    <td>L2-EXPR-03</td>
    <td>match_expr_or_stmt_call</td>
    <td>code snippet：`return LG.foo(x)` 作为 Return 语句。</td>
    <td>调用 `func_attr="foo"`，返回 `("LG", call)`。</td>
    <td>覆盖 Return 节点匹配成功路径。</td>
  </tr>
  <tr>
    <td>L2-EXPR-04</td>
    <td>match_expr_or_stmt_call</td>
    <td>任意上述三种形式之一，但调用 `func_attr="bar"`。</td>
    <td>返回 `None`。</td>
    <td>覆盖 attr 不匹配路径。</td>
  </tr>
  <tr>
    <td>L2-EXPR-05</td>
    <td>match_expr_or_stmt_call</td>
    <td>code snippet：`foo(x)`（无 owner，仅单名函数调用）。</td>
    <td>根据实现：可能直接返回 `None`（因为没有 `.attr`），或走特定 fallback。UT 断言与实际实现一致。</td>
    <td>覆盖「非 owner.attr 调用」路径。</td>
  </tr>
</table>

##### 4.6.3 AST 构造方式说明

- 所有 L2-EXPR-* 测例均可用 `ast.parse` 解析函数体片段，再提取对应的 `Expr` / `Assign` / `Return` 节点。

---

### 5. 覆盖率预期与后续工作

- 以上测例组合后，Layer 0/1/2 关键函数的主要分支（含正常路径、参数为空路径、不匹配路径、控制流分支遍历路径）均被覆盖，预期可达到：
  - line coverage ≥ 90%
  - branch coverage ≥ 50%
- UT 实现阶段需根据 `scripts/ast_frontend.py` 实际代码细节微调少量断言（例如 prefix+name 组合逻辑具体为 AND 还是 OR），但**不得改变本设计文档中描述的分层和职责边界**。
- 本设计文档仅提供 UT 方案，具体 UT 代码实现、文件命名与落盘路径由 storage 仓库 `feat/ast-step2-tests` 分支负责落地，在实现前需再次对照本方案确认未遗漏关键分支。