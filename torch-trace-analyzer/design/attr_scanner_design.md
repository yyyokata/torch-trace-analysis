## AttrScanner 设计：ASTFrontend 三层工具集 + AttrScanner 六类 Attr 扫描

本文档描述 `scripts/ast_frontend.py` 与 `scripts/attr_scanner.py` 在 `feat/ast-timing-fix` 分支上的分层方案、迁移边界与 UT 计划。目标是把 ASTFrontend 收敛为无业务语义的 Layer 0/1/2 工具层，由 AttrScanner 独占 LG / Container / Result 等语义，并统一产出 6 类 Attr：`InputAttr`、`ModuleAttr`、`ContainerAttr`、`ResultAttr`、`ForwardArgAttr`、`ReturnValAttr`。

---

### 1. 分层结构总览

整体分层如下：

- **Layer 0 — ASTFrontend：node 级原子提取**
- **Layer 1 — ASTFrontend：stmt 解构**
- **Layer 2 — ASTFrontend：pattern 匹配（所有 pattern 由调用者传入）**
- **Layer 3 — AttrScanner：语义决策层（持有 LG / Container 等业务常量）**

#### 1.1 Layer 0：ASTFrontend node 级原子提取

Layer 0 只提供最小粒度 AST 信息提取，不感知业务语义。

- `_node_loc(node) -> CallLoc`
- `_node_to_text(node) -> str`
- `_extract_self_attr_name(target) -> Optional[str]`
- `_expr_leaf_name(node) -> Optional[str]`
- `_get_method_node(class_name, method_name) -> Optional[ast.FunctionDef]`

#### 1.2 Layer 1：ASTFrontend stmt 解构

Layer 1 负责把语句拆成稳定可消费的结构。

- `stmt_assign_target(stmt) -> Optional[ast.expr]`
- `stmt_assign_call_value(stmt) -> Optional[ast.Call]`
- `call_func_text(call) -> str`
- `call_kwarg_texts(call) -> dict[str, str]`
- `call_arg_texts(call) -> list[str]`
- `iter_init_stmts(class_name) -> Iterator[ast.stmt]`

`iter_init_stmts` 递归遍历 `__init__` 方法体中的所有控制流分支，统一产出内部语句，调用方不再自行展开 `for / if / with / try`。

#### 1.3 Layer 2：ASTFrontend pattern 匹配

Layer 2 在 Layer 0/1 之上提供通用 pattern helper，本身不硬编码任何 LG / 容器 / 业务函数名。

- `match_self_attr_call_assign(stmt, *, func_prefixes=None, func_names=None) -> Optional[tuple[str, ast.Call]]`
- `match_call_by_func(call, *, func_prefixes=None, func_names=None) -> Optional[str]`
- `match_expr_or_stmt_call(node, *, func_attr=None) -> Optional[tuple[str, ast.Call]]`

组合语义约束：

- `func_prefixes is not None` 时，要求 `func_text.startswith(prefix)` 成立。
- `func_names is not None` 时，要求 `func_leaf in func_names` 成立。
- **二者同时配置时是 AND 语义，不是 OR 语义。**
- 二者都为 `None` 时，仅检查结构。

#### 1.4 Layer 3：AttrScanner 语义决策层

AttrScanner 是唯一持有业务语义常量的层。

```python
_LG_PREFIXES = {"LG."}
_CONTAINER_LEAF_NAMES = {"ModuleList", "ModuleDict", "Sequential"}
_LG_RESULT_FUNC_NAME = "result"
_HEAD_METHOD = "head"
```

AttrScanner 扫描来源如下：

| 类型 | 来源 | 扫描方式 |
|---|---|---|
| InputAttr | `__init__` 中 `self.attr = LG.xxx(...)` | `iter_init_stmts` + `match_self_attr_call_assign(..., func_prefixes={"LG."})` |
| ModuleAttr | `__init__` 中普通模块构造赋值 | `iter_init_stmts` + `match_self_attr_call_assign(...)`，排除 LG / Container |
| ContainerAttr | `__init__` 中容器构造赋值 | `iter_init_stmts` + `match_self_attr_call_assign(..., func_names=_CONTAINER_LEAF_NAMES)` |
| ResultAttr | `forward` 中 `result = LG.result()` 与后续 `.head(...)` | Layer 1/2 组合匹配 `LG.result` 与 `.head(...)` |
| ForwardArgAttr | `forward` 方法形参列表 | 读 `forward` 节点 `args.args`，跳过 `self` |
| ReturnValAttr | `forward` 方法返回语句 | 找 `ast.Return`，单值→`ret_index=0`，tuple→每个元素一个 `ReturnValAttr` |

---

### 2. 关键约束与迁移要求

#### 2.1 ASTFrontend 不感知业务语义

ASTFrontend 禁止硬编码：

- `LG` / `LG2` / `LocalGraph` 等 LG 系列名
- `ModuleList` / `ModuleDict` / `Sequential` 等容器业务集合
- 任何具体模型或 Wrapper 名称

所有业务 pattern 必须由上层以参数传入。

#### 2.2 去掉 LG2

本阶段所有设计与实现统一只保留：

```python
_LG_PREFIXES = {"LG."}
```

文档、代码、UT 中凡是 `LG2` 出现都必须删除或替换，不保留双前缀方案。

#### 2.3 `parse_local_*` 语义泄漏删除范围

Phase 2 删除以下 ASTFrontend 语义接口，不保留 shim / alias：

- `get_input_attrs`
- `get_result_attrs`
- `parse_result_head_call`
- `parse_local_lg_assign`
- `parse_local_container_ctor`

保留的 ASTFrontend 能力只应是 Layer 0/1/2 通用工具；语义聚合统一迁入 AttrScanner。

#### 2.4 `legacy_static_tree.py` 调用方同步替换

三个调用点必须切到 Layer 1/2 + AttrScanner：

1. 原 `parse_local_lg_assign` 调用点 → `match_self_attr_call_assign(..., func_prefixes={"LG."})`
2. 原 `parse_local_container_ctor` 调用点 → `match_self_attr_call_assign(..., func_names={"ModuleList", "ModuleDict", "Sequential"})`
3. 原 `get_result_attrs` 调用点 → `AttrScanner()._scan_result_attrs(cname, _fe)`，再转成按 `attr_name` 聚合的 dict

#### 2.5 ResultAttr 的边信息字段暂留

`ResultAttr` 中以下字段仍作为临时载体保留：

- `prediction_expr`
- `label_expr`
- `sample_rate_expr`
- `loss_expr`
- `classifier_type`

本阶段仅负责在 AttrScanner 内完整提取，不在 ASTFrontend 内保留结果语义。

---

### 3. AttrScanner 接口与职责

#### 3.1 对外结构

```python
class AttrScanner:
    _LG_PREFIXES = {"LG."}
    _CONTAINER_LEAF_NAMES = {"ModuleList", "ModuleDict", "Sequential"}
    _LG_RESULT_FUNC_NAME = "result"
    _HEAD_METHOD = "head"

    def __init__(self):
        self._attr_id_counter = 0
        self.attrs_by_id: dict[int, AttrType] = {}
```

#### 3.2 attr_id 分配

AttrScanner 负责为所有扫描结果分配唯一 `attr_id`，并写入 `attrs_by_id`。

```python
def _next_id(self) -> int: ...
def _register(self, attr: AttrType) -> AttrType: ...
```

要求：

- 同一个 `AttrScanner` 实例内，所有 attr 的 `attr_id` 唯一且递增。
- `scan_class()` 返回的 6 个列表中的对象全部经过 `_register()`。

#### 3.3 `scan_class` 返回结构

```python
{
    "inputs":       list[InputAttr],
    "modules":      list[ModuleAttr],
    "containers":   list[ContainerAttr],
    "results":      list[ResultAttr],
    "forward_args": list[ForwardArgAttr],
    "return_vals":  list[ReturnValAttr],
}
```

---

### 4. 六类扫描方法设计

#### 4.1 `_scan_input_attrs`

实现要求：

- 遍历 `fe.iter_init_stmts(class_name)`
- 使用 `fe.match_self_attr_call_assign(stmt, func_prefixes=self._LG_PREFIXES)` 做粗匹配
- 使用 `fe.match_call_by_func(call, func_prefixes=self._LG_PREFIXES)` 拿到完整函数名
- 去掉 `LG.` 前缀后得到 `kind`
- 生成 `InputAttr(attr_name, class_name, def_loc, kind, source_expr)`

#### 4.2 `_scan_module_attrs`

实现要求：

- 遍历 `fe.iter_init_stmts(class_name)`
- 使用 `fe.match_self_attr_call_assign(stmt)` 捕获所有 `self.attr = call(...)`
- 排除 `LG.` 前缀调用
- 排除 `_CONTAINER_LEAF_NAMES` 容器构造
- 剩余项生成为 `ModuleAttr`

#### 4.3 `_scan_container_attrs`

实现要求：

- 遍历 `fe.iter_init_stmts(class_name)`
- 使用 `fe.match_self_attr_call_assign(stmt, func_names=self._CONTAINER_LEAF_NAMES)`
- 容器种类通过 `fe._expr_leaf_name(call.func)` 的 leaf name 获得
- 生成 `ContainerAttr(container_kind=kind, ...)`

#### 4.4 `_scan_result_attrs`

实现要求：

- 只扫描 `forward` 方法体
- 对每条语句先识别 `result_var = LG.result()`
  - 使用 Layer 1/2 组合：`stmt_assign_target` + `stmt_assign_call_value` + `match_call_by_func(..., func_prefixes={"LG."}, func_names={"result"})`
- 再识别 `result_var.head(...)`
  - 使用 `match_expr_or_stmt_call(stmt, func_attr="head")`
  - `owner_text` 必须落在已记录的 `result_var` 集合里
- 从 `call_arg_texts` / `call_kwarg_texts` 提取：
  - `head_name`
  - `prediction_expr`
  - `label_expr`
  - `sample_rate_expr`
  - `loss_expr`
  - `classifier_type`
- 返回 `list[ResultAttr]`

#### 4.5 `_scan_forward_arg_attrs`

```python
def _scan_forward_arg_attrs(self, class_name: str, fe: ASTFrontend) -> list[ForwardArgAttr]:
    """读 forward 形参列表（跳过 self），每个形参 → ForwardArgAttr(arg_index=i)"""
```

实现要求：

- `method_node = fe._get_method_node(class_name, "forward")`
- 读取 `method_node.args.args`
- 若首参为 `self` 则跳过
- 其余形参依次生成 `ForwardArgAttr(attr_name=arg.arg, arg_index=i)`

#### 4.6 `_scan_return_val_attrs`

```python
def _scan_return_val_attrs(self, class_name: str, fe: ASTFrontend) -> list[ReturnValAttr]:
    """找 forward 里所有 ast.Return，单值→[ReturnValAttr(ret_index=0)]
    tuple→[ReturnValAttr(ret_index=0), ReturnValAttr(ret_index=1), ...]"""
```

实现要求：

- `method_node = fe._get_method_node(class_name, "forward")`
- `ast.walk(method_node)` 找所有 `ast.Return`
- `return x` → 一个 `ReturnValAttr(ret_index=0)`
- `return (a, b, c)` → 三个 `ReturnValAttr(ret_index=0/1/2)`

---

### 5. Layer 2 组合语义 UT 修正

#### 5.1 `match_self_attr_call_assign` 组合语义修正

- **L2-SELF-05 反例**
  - 代码：`self.x = LG.label("l")`
  - 参数：`func_prefixes={"LG."}` + `func_names={"dense_feature"}`
  - 期望：返回 `None`
  - 原因：prefix 满足，但 leaf name `label` 不在 `{"dense_feature"}` 中，AND 不通过

- **L2-SELF-05b 正例**
  - 代码：`self.x = LG.dense_feature("f")`
  - 参数：`func_prefixes={"LG."}` + `func_names={"dense_feature"}`
  - 期望：返回非 `None`

#### 5.2 `match_call_by_func` 组合语义修正

- **L2-CALL-05 反例**
  - 代码：`LG.label("l")`
  - 参数：`func_prefixes={"LG."}` + `func_names={"dense_feature"}`
  - 期望：返回 `None`

- **L2-CALL-05b 正例**
  - 代码：`LG.dense_feature("f")`
  - 参数：`func_prefixes={"LG."}` + `func_names={"dense_feature"}`
  - 期望：返回非 `None`

---

### 6. UT 落地方案

UT 文件落地到：`develop/ast/storage/testset/unit/test_attr_scanner.py`

约束：

- 使用 `ast.parse` 生成合成 AST
- `ASTFrontend(source=...)` 构造 frontend
- 每个测例一个 `test_` 函数
- 覆盖率目标：`attr_scanner.py` line ≥ 90%，branch ≥ 50%

#### 6.1 AttrScanner 测例列表

| 测例 ID | 场景 | 期望 |
|---|---|---|
| T1 | `self.feat = LG.dense_feature("x")` | 扫出 1 个 `InputAttr`，`kind == "dense_feature"` |
| T2 | `self.linear = nn.Linear(4, 8)` | 扫出 1 个 `ModuleAttr` |
| T3 | `self.layers = nn.ModuleList([])` | 扫出 1 个 `ContainerAttr`，`container_kind == "ModuleList"` |
| T4 | `result = LG.result(); return result.head(...)` | 扫出 1 个 `ResultAttr`，`prediction/label/...` 字段正确 |
| T5 | `__init__` 中混合 Input / Module / Container | 三类扫描互不串扰 |
| T6 | 无 `__init__` | Input / Module / Container 返回空列表 |
| T7 | `match_call_by_func` AND 反例 | `LG.label` + `dense_feature` 组合返回 `None` |
| T8 | `match_call_by_func` AND 正例 | `LG.dense_feature` + `dense_feature` 组合匹配成功 |
| T9 | `match_self_attr_call_assign` AND 反例 | `self.x = LG.label(...)` 返回 `None` |
| T10 | `match_self_attr_call_assign` AND 正例 | `self.x = LG.dense_feature(...)` 匹配成功 |
| T11 | 整合场景：同一类中同时包含 6 类 Attr | `scan_class` 全部返回，且 `attr_id` 全局唯一不重叠 |
| T12 | `def forward(self, x, y): ...` | 两个 `ForwardArgAttr`，`arg_index == 0/1` |
| T13a | `return (a, b, c)` | 三个 `ReturnValAttr`，`ret_index == 0/1/2` |
| T13b | `return x` | 一个 `ReturnValAttr`，`ret_index == 0` |

#### 6.2 重点断言

- `scan_class()` 返回结构完整，6 个 key 均存在
- `attrs_by_id` 数量与全部 attr 数量一致
- `attr_id` 无重复
- ResultAttr 的 `def_loc` 锚定 `LG.result()` 定义位置
- `ForwardArgAttr` / `ReturnValAttr` 的索引严格按源码顺序递增

---

### 7. 实施完成后的代码检查项

- `scripts/attr_scanner.py`：完全移除旧 `NodeVisitor` 原型，不出现 `LG2`
- `scripts/ast_frontend.py`：删除 `LG_FUNC_KINDS`、`get_input_attrs`、`get_result_attrs`、`parse_result_head_call`、`parse_local_lg_assign`、`parse_local_container_ctor`
- `scripts/legacy_static_tree.py`：三个调用点全部完成替换
- `scripts/attr_scanner.py` 文件行数 `< 300`
- `scripts/ast_frontend.py` 文件行数 `< 2000`

---

### 8. 本次阶段性边界

本次仅完成 AttrScanner 六类扫描、ASTFrontend 语义泄漏清理与对应 UT。更大范围的 DAG builder / edge evidence 迁移不在本文档范围内。
