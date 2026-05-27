# AST 求值器重新架构设计

> 状态：架构设计文档（不修改任何代码，不跑回归）
> 作者：林楠（Constructor 整理）
> 适用主脚本：`ast_refactor_workdir/scripts/analyze_trace.py`（7921 行）
> 设计目标：用一套**纯 AST 驱动的常量求值系统**彻底替代当前 regex+AST 混合的评估器
> 关联背景：`scenario5_design.md`（场景 5 跨 call site 常量传播） · `ast_parse_unification_evaluation.md`（parse 统一化评估）

---

## 0. TL;DR

| 项 | 当前实现 | 新架构 |
|---|---|---|
| 求值入口 | `_eval_int_atom` / `_eval_list_len` / `_resolve_range_n` / `_resolve_iter_len`（4 个独立函数） | 统一为单一类 `ConstantResolver`，对外暴露 `eval_int(expr, scope)` / `eval_list_len(expr, scope)` |
| 求值依赖的"事实表" | `file_int_consts` / `global_int_const_values` / `ctor_kw_int_args` / `ctor_kw_list_lens` / `instance_kw_list_lens`（5 张分散的 dict，类级聚合） | **统一进 `ConstantTable`**，按 `Scope` 分桶，加上 `dataclass_field_defaults` / `local_dataclass_instances` / `instance_const_chain` 三张新表 |
| 属性链深度 | 一级 `self.attr` | **任意深度** `self.a.b.c.d…` |
| 跨 call site 参数传播 | 无（`SeqTrans(ins_config)` 拿不到 `ins_config` 的字段） | 有（`_BindResolver` 形参→实参绑定 + `dataclass` materialization） |
| 局部变量别名链 | 单步 `var = INT` / `var = CONST` | 任意步的 `var1 = var2 = ... = literal` 链 |
| Regex fallback | 散落在 `_extract_kw_int_args` / `_extract_kw_list_lens` / `_eval_list_len`（NAME 识别）/ 主扫描循环（local_vars）等多处 | **完全删除**，AST 解析失败 → `None`（记 metric，不兜底） |
| memo / DP | 无（同一表达式可被多次求值） | 有（按 `(scope_key, expr_text)` cache） |
| 失败语义 | 部分函数返回 None，部分返回空 dict，部分写 stderr warning | 统一 `Optional[T]` + `EvalTrace`（debug 模式下记录失败原因） |

**核心设计原则**：
1. **一切皆 AST 节点**：求值器输入是 `ast.expr` 节点（不是字符串），杜绝二次 `ast.parse` 字符串片段。
2. **Scope 是一等公民**：每次求值必须显式带上 `(file, class, method, parent_class, parent_attr)` 上下文；无 scope 时禁止求值。
3. **事实表先行，求值后行**：`ConstantTable.build_all(source_files, ast_frontends)` 是一个**纯**的预扫描阶段，把所有静态字面量、dataclass、局部变量、ctor kwargs、形参绑定一次性扫完；`ConstantResolver` 只读不写。
4. **失败保守**：求值失败一律 `None`，由调用方决定是否走 wildcard / dynamic_attrs 兜底；不允许在求值器内部偷偷返回"猜测值"。

---

## 1. 当前实现速查 + 痛点定位

### 1.1 当前 4 个求值函数

| 函数 | 行号 | 输入 | 求值范围 |
|---|---|---|---|
| `_eval_int_atom(expr, fname)` | 5321 | 字符串表达式 | int 字面量 / 文件内 `[A-Z_]+ = N` 常量 / 跨文件唯一常量 |
| `_eval_list_len(expr, fname)` | 5405 | 字符串表达式 | `[a,b]` / `NAME`（指 str-list global） / `[x]+[y]` / `[x]*N` |
| `_resolve_range_n(expr, fname, cname, self_to_param, local_vars)` | 6653 | 字符串表达式 | 上述 + 形参名 + `self.X`→形参名 |
| `_resolve_iter_len(expr, fname, cname, self_to_param, local_vars)` | 6628 | 字符串表达式 | 上述（list 版本） |

### 1.2 当前 5 张事实表（数据流分散）

| 表 | 行号 | key | value | scope |
|---|---|---|---|---|
| `file_int_consts` | 5276 | `(fname, NAME)` | `int` | 文件 |
| `global_int_const_values` | 5288 | `NAME` | `Set[int]` | 全局 |
| `file_str_list_globals_raw` | 5298 | `(fname, NAME)` | `str`（list literal raw text） | 文件 |
| `ctor_kw_int_args` | 5306 | `(ClassName, kw)` | `Set[int]` | **类级**（多 instance union） |
| `ctor_kw_list_lens` | 5313 | `(ClassName, kw)` | `Set[int]` | **类级** |
| `instance_kw_list_lens` | 5319 | `(parent_cname, attr)` | `Dict[kw, int]` | **per-instance** |

### 1.3 痛点（为何当前实现不够）

| 痛点 | 触发场景 | 当前结果 |
|---|---|---|
| **P1: 一级属性硬编码** | `range(self.config.num_hidden_layers)` | `_resolve_range_n` line 6667 `expr.split('.', 1)[1]` 拿到 `"config.num_hidden_layers"`，再用 attr 整体去 `self_to_param` 查找 → miss → `None`。 |
| **P2: 跨 call site 不传播** | `self.ins_trans = SeqTrans(ins_config)` + `ins_config = TransformerConfig(num_hidden_layers=6)` | 位置实参 `ins_config` 完全不进 `ctor_kw_int_args`（因为它不是 kw=）。子类内部 `self.config` 解析时拿不到值。 |
| **P3: regex 提 kw 有边界 bug** | `Foo(a=b, msg="kw=2")` | `_extract_kw_int_args` 的 `re.sub(r"([\"\']).*?\1", "''", text)` 有时把内层引号 scrub 不干净；又因为只识别 ATOM=`[A-Z_]+\d+`，`a=foo()` 这种调用被忽略。 |
| **P4: regex 提 list literal 不严谨** | `self.x = [a, b, [c, d]]` | `_split_top_level` 是手写状态机，对 f-string / 三引号字符串 / 类型注解 `Dict[str, int]` 等可能误切。 |
| **P5: 字符串片段二次 parse** | `_parse_local_stmt` line 5945 | `ast.parse(_text)` 把单行重新 parse → lineno=1，无法用 ASTFrontend 的位置 helper（已被 ast_parse_unification_evaluation.md 指出）。 |
| **P6: 局部变量别名链断** | `n = 6; m = n; range(m)` | `local_vars` 在 line 6722 只识别 `var = ATOM`（ATOM = `[A-Z_]+|\d+`），不支持 `var = OTHER_VAR`，所以多跳 alias 直接断。 |
| **P7: memo 缺失** | 同一类被多次扫到 | `_eval_list_len("[a,b,c]")` 在每个调用点都从头解析。 |
| **P8: 失败信号噪声** | regex 路径触发 | `_warn_regex_fallback` 写 stderr，但**不会让结果失败**——AST 路径返回空就回退到 regex，regex 再返回空就静默 `None`，根本看不出哪个 case 失败了。 |

---

## 2. 新架构总览

### 2.1 模块层次

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 0：ASTFrontend（已存在，复用）                                    │
│  - 全文件 AST 树 / class_registry / get_init_assignments_ast            │
│  - get_self_param_aliases / get_loop_expansion_records ...              │
│  - 提供查询，不做求值                                                    │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 1：ConstantTable（新建）                                          │
│  事实表集合，预扫描阶段 build_all() 一次填满，build 后只读。              │
│  ─────────────────────────────────────────────────────────────────────  │
│  · file_int_consts       (file scope)                                    │
│  · file_str_list_consts  (file scope，注意：直接存 List[str]，不再存 raw)│
│  · file_dict_literals    (file scope，dict literal)                      │
│  · global_int_consts     (cross-file unique)                             │
│  · dataclass_defaults    (class scope，@dataclass field defaults)        │
│  · class_init_params     (class scope，__init__ 形参列表)                │
│  · ctor_kw_args          (class scope，kw → {literal_int, ...})          │
│  · ctor_kw_list_args     (class scope，kw → {list_len, ...})             │
│  · instance_kw_int       (per-instance，覆盖类级)                        │
│  · instance_kw_list_len  (per-instance，已有)                            │
│  · instance_const_chain  (per-instance，dataclass 字段表)                │
│  · local_aliases         (method scope，var → AST 节点)                  │
│  · local_dataclass_inst  (method scope，var → {field: value})            │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 2：ConstantResolver（新建）                                       │
│  纯求值器，**唯一**对外接口：                                             │
│    eval_int(node, scope)        -> Optional[int]                         │
│    eval_list_len(node, scope)   -> Optional[int]                         │
│    eval_dataclass_fields(node, scope) -> Optional[Dict[str, int]]        │
│  内部 dispatch（visit_Constant / visit_Name / visit_Attribute /          │
│   visit_Call / visit_BinOp / visit_Subscript / visit_Starred）。         │
│  全程 AST 节点输入，不接受字符串。                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 3：调用方（build_static_module_tree / _AST_ModuleTreeExtractor）  │
│  - 构造 Scope 对象后调用 resolver.eval_int(node, scope)                  │
│  - 拿到 None 时走原有 wildcard / dynamic_attrs / 不展开等保守路径        │
│  - 不再持有任何 regex / 字符串 split 逻辑                                │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 与现有代码的集成点

1. **`ASTFrontend` 实例池化**（参考 `ast_parse_unification_evaluation.md`）：
   - 在 `build_static_module_tree` 顶层一次性建好 `ast_frontends: Dict[fname, ASTFrontend]`，传入 `ConstantTable.build_all`。
   - 替代当前 `_get_ast_frontend(fname)` 闭包的 lazy 缓存（避免 ASTFrontend 在不同闭包中重复建）。

2. **`ConstantTable.build_all` 在 `build_static_module_tree` 顶部调用一次**：
   - 输入：`ast_frontends`、`source_files`（仅用于辅助）、`nn_module_classes`。
   - 输出：一个 `ConstantTable` 实例，挂在外层闭包变量 `const_table` 上，所有下层函数共享。
   - 调用时机：`class_map`、`module_call_order` 之后，第一次进入"扫描各方法"循环之前。

3. **`ConstantResolver` 替代 4 个旧求值函数**：
   - 删除 `_eval_int_atom` / `_eval_list_len` / `_extract_kw_int_args` / `_extract_kw_list_lens` / `_resolve_range_n` / `_resolve_iter_len` 共 6 个函数。
   - 新增 `resolver = ConstantResolver(const_table)`，在外层闭包中构造一次。
   - 调用方改为：
     ```python
     # 以前
     n = _resolve_range_n(range_expr_str, fname, cname, self_to_param, local_vars)
     # 现在
     n = resolver.eval_int(loop_record.iter_expr_node,
                           Scope(file=fname, cls=cname, method=mname,
                                 parent_cls=parent_cname, parent_attr=parent_attr))
     ```

4. **`_parse_local_*` 系列函数**（行 5945~6105 及以后）：见 §4 详细讨论。简单说：
   - `_parse_local_stmt`（字符串再 parse）**完全删除**；
   - `_parse_local_ctor_assign` / `_parse_local_lg_assign` / `_parse_local_*_ctor` 等节点级 helper **保留并改造**：把它们的输入从"字符串 → AST 节点"二重路径改为只接受"已经从 ASTFrontend 拿到的 AST 节点"；
   - `_node_to_text` 这个本地 helper（依赖 `ast.unparse`）**改为统一调用 `ASTFrontend._node_to_text`**（O(1) 切片）。

---

## 3. ConstantTable 详细设计

### 3.1 数据结构

```python
# === Scope 定义 ===
@dataclass(frozen=True)
class Scope:
    file: str                       # fname
    cls: Optional[str] = None       # 当前类
    method: Optional[str] = None    # 当前方法
    parent_cls: Optional[str] = None      # 父类（实例化我的类）
    parent_attr: Optional[str] = None     # 父类中保存我的 attr 名

    @property
    def class_key(self) -> Tuple[str, str]:
        return (self.file, self.cls)

    @property
    def method_key(self) -> Tuple[str, str, str]:
        return (self.file, self.cls, self.method)

    @property
    def instance_key(self) -> Tuple[str, str]:
        return (self.parent_cls, self.parent_attr)

# === Value-bearing entries ===
# 所有 value 都附带 origin 信息（用于 trace / debug，不影响求值）
@dataclass(frozen=True)
class IntValue:
    value: int
    origin: str  # "literal" | "file_const" | "global_const" | "kw_arg" | "alias" | "dataclass_field" | ...

@dataclass(frozen=True)
class ListValue:
    length: int
    items: Optional[Tuple[ast.expr, ...]] = None   # 可选：保留原 AST 节点用于深度追踪
    origin: str = "list_literal"

# === 表 ===
class ConstantTable:
    # ── 文件级 ──
    file_int_consts: Dict[str, Dict[str, IntValue]]
    file_str_list_consts: Dict[str, Dict[str, ListValue]]
    file_dict_literals: Dict[str, Dict[str, Dict[str, ast.expr]]]
        # {fname: {NAME: {key: ast.expr_value}}}  仅当 dict literal 全字符串 key 时

    # ── 全局唯一 ──
    global_int_consts: Dict[str, IntValue]   # NAME → IntValue（仅当跨文件取值唯一）

    # ── 类级 ──
    class_init_params: Dict[Tuple[str, str], List[str]]
        # (file, cls) → ['self', 'config', 'is_last_layer', ...]（不含 *args/**kwargs）

    dataclass_defaults: Dict[Tuple[str, str], Dict[str, IntValue]]
        # (file, dataclass_name) → {field: IntValue}

    ctor_kw_args: Dict[str, Dict[str, Set[IntValue]]]
        # ClassName → {kw: {IntValue, ...}}  类级 union（保留多 instance 的全部值）

    ctor_kw_list_args: Dict[str, Dict[str, Set[int]]]
        # ClassName → {kw: {list_len, ...}}

    # ── 实例级 ──
    instance_kw_int: Dict[Tuple[str, str], Dict[str, IntValue]]
        # (parent_cls, parent_attr) → {kw: IntValue}（**覆盖**类级，用于 per-instance 精度）

    instance_kw_list_len: Dict[Tuple[str, str], Dict[str, int]]
        # 已有，等价于现 instance_kw_list_lens

    instance_const_chain: Dict[Tuple[str, str], Dict[str, Dict[str, IntValue]]]
        # (parent_cls, parent_attr) → {param_name → {field → IntValue}}
        # 例：("InsTrans","ins_trans") → {"config": {"num_hidden_layers": 6, ...}}
        # 对应 scenario5 跨 call site 传播

    # ── 方法级 ──
    local_aliases: Dict[Tuple[str, str, str], Dict[str, ast.expr]]
        # (file, cls, method) → {var_name: AST 节点（赋值的右值）}
        # 不直接存值，存 AST 节点，求值器按需展开

    local_dataclass_inst: Dict[Tuple[str, str, str], Dict[str, Dict[str, IntValue]]]
        # (file, cls, method) → {var_name: {field: IntValue}}

    # ── self.attr → 形参映射（已有） ──
    self_to_param: Dict[Tuple[str, str], Dict[str, str]]
        # (file, cls) → {self.attr: param_name}

    # ── memo ──
    _eval_cache: Dict[Tuple[str, str, str, str, int], Optional[IntValue]]
        # key = (file, cls, method, parent_cls or '', id(node))
        # 注意：用 id(node) 而不是 ast.unparse(node)，因为 unparse 又会 O(n) 再走一遍
```

### 3.2 构建时机与流程

```python
class ConstantTable:
    @classmethod
    def build_all(cls, source_files: Dict[str, List[str]],
                  ast_frontends: Dict[str, ASTFrontend],
                  nn_module_classes: Set[str]) -> "ConstantTable":
        table = cls()

        # Pass 1：文件级 scan（按文件顺序，互不依赖）
        for fname, fe in ast_frontends.items():
            table._scan_file_level_consts(fname, fe)
            table._scan_dataclass_defaults(fname, fe)

        # Pass 2：跨文件唯一常量收敛（依赖 Pass 1）
        table._reduce_global_int_consts()

        # Pass 3：类级 scan（依赖 Pass 1+2）
        for fname, fe in ast_frontends.items():
            for cname in fe.class_registry:
                if cname not in nn_module_classes and not table._is_dataclass(fname, cname):
                    continue
                table._scan_class_init_params(fname, cname, fe)
                table._scan_self_to_param(fname, cname, fe)
                table._scan_local_aliases(fname, cname, fe)
                table._scan_local_dataclass_inst(fname, cname, fe)
                table._scan_ctor_call_sites(fname, cname, fe)   # 收集 self.x = SubClass(...) 的 kwargs

        # Pass 4：跨 call site 形参绑定（依赖 Pass 3 的所有数据）
        # 这是 scenario5 的核心：把父类位置实参映射到子类形参
        for fname, fe in ast_frontends.items():
            for cname in fe.class_registry:
                if cname not in nn_module_classes:
                    continue
                table._bind_param_value_chains(fname, cname, fe)

        return table
```

**Pass 1 细节**（文件级 scan，举例 `_scan_file_level_consts`）：

```python
def _scan_file_level_consts(self, fname, fe: ASTFrontend):
    """扫顶层 ast.Assign，识别：
       - INT_NAME = <literal_int>             → file_int_consts
       - LIST_NAME = [<str>, <str>, ...]      → file_str_list_consts
       - DICT_NAME = {'k': <int>, ...}        → file_dict_literals
    """
    for node in fe.tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        name = tgt.id
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, int):
            self.file_int_consts.setdefault(fname, {})[name] = IntValue(v.value, "file_const")
        elif isinstance(v, ast.List) and all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in v.elts
        ):
            self.file_str_list_consts.setdefault(fname, {})[name] = ListValue(
                length=len(v.elts), items=tuple(v.elts), origin="file_str_list"
            )
        elif isinstance(v, ast.Dict):
            d = {}
            ok = True
            for k, val in zip(v.keys, v.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    ok = False
                    break
                d[k.value] = val   # 保留 AST 节点，求值时再展开
            if ok:
                self.file_dict_literals.setdefault(fname, {})[name] = d
```

**Pass 2** 等价于当前 `global_int_const_values` 收敛逻辑，但只接受 `len(set_of_values) == 1` 的常量。

**Pass 3** 中 `_scan_local_aliases` 是关键改进：

```python
def _scan_local_aliases(self, fname, cname, fe: ASTFrontend):
    """扫 __init__（以及其他可能含常量的 method），识别：
       - var = <ast.Constant>                 → IntValue (literal)
       - var1 = var2                          → 别名链（保留为 AST 节点）
       - var = SOME_NAME                      → 同上
       - var = self.X                         → 同上（不立即求值）
       - var = SomeDataclass(...)             → 标记为 dataclass instance（见 _scan_local_dataclass_inst）
    在多步赋值（var1 = var2 = literal）下，把所有 LHS 都登记。
    """
    for mname in fe.class_registry[cname]['method_map']:
        node = fe._get_method_node(cname, mname)
        if node is None:
            continue
        scope_key = (fname, cname, mname)
        for stmt in self._iter_assign_stmts(node):
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        self.local_aliases.setdefault(scope_key, {})[tgt.id] = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                self.local_aliases.setdefault(scope_key, {})[stmt.target.id] = stmt.value
```

**Pass 4** 形参绑定：

```python
def _bind_param_value_chains(self, fname, cname, fe):
    """对每个 self.x = SubClass(arg0, arg1, kw=...) 调用：
       1. 拿 SubClass 的 __init__ 形参列表（class_init_params）
       2. positional bind：args[i] → params[i]
       3. kwargs bind：kw → param_name
       4. 若 bound value 指向已知 dataclass 实例（local_dataclass_inst），
          写入 instance_const_chain[(cname, attr)][param] = field_dict
       5. 若 bound value 是字面量 int，写入 instance_kw_int
       6. 若 bound value 是 list literal，写入 instance_kw_list_len
    """
```

### 3.3 memo 策略

- **Cache key 用 `id(node)`**：AST 节点在 `ast_frontends` 中是稳定的（同一个 frontend 实例下），id 唯一。把字符串 cache 改成 id cache，避免 unparse 开销。
- **Cache 作用域**：跟 `ConstantTable` 实例同生命周期；`ConstantTable` 跟 `build_static_module_tree` 调用同生命周期。
- **Cache 维度**：`(file, cls, method, parent_cls + parent_attr, id(node))`。其中 `parent_cls + parent_attr` 是因为同一个 AST 节点在不同 instance scope 下可能解析出不同值（典型如 `self.config.num_hidden_layers`）。
- **递归保护**：`_eval_cache` 在递归过程中先放占位 `RECURSING`，命中即视为环 → 返回 `None`，并清掉占位。

---

## 4. 求值器（ConstantResolver）详细设计

### 4.1 对外 API

```python
class ConstantResolver:
    def __init__(self, table: ConstantTable):
        self.table = table

    # ─────────── 三个公开入口 ───────────
    def eval_int(self, node: ast.expr, scope: Scope) -> Optional[IntValue]: ...
    def eval_list_len(self, node: ast.expr, scope: Scope) -> Optional[int]: ...
    def eval_dataclass_fields(self, node: ast.expr, scope: Scope) -> Optional[Dict[str, IntValue]]: ...

    # ─────────── 内部 dispatcher ───────────
    def _visit(self, node, scope, kind):  # kind ∈ {"int", "list_len", "fields"}
        method = getattr(self, f"_visit_{type(node).__name__}_{kind}", None)
        if method is None:
            return None
        return method(node, scope)
```

### 4.2 支持的表达式形式

下表覆盖**所有**新求值器必须能跑通的场景，按 AST 节点类型组织：

| AST 节点 | 形式举例 | int 求值路径 | list_len 求值路径 | fields 求值路径 |
|---|---|---|---|---|
| `Constant(int)` | `6` | 直接返回 | – | – |
| `Constant(str)` | `"abc"` | – | – | – |
| `Name` | `LAYERS` / `n` / `cfg` | ① local_aliases 链解一层（递归求值 RHS）；② file_int_consts；③ global_int_consts；④ class_init_params 中的形参 → ctor_kw_args[cls][param] union（仅唯一时返回） | ① local_aliases；② file_str_list_consts；③ ctor_kw_list_args | ① local_dataclass_inst |
| `Attribute` | `self.X` / `self.X.Y` / `self.X.Y.Z` | **递归剥层**（见 §4.3） | 同左 | 同左 |
| `Subscript` | `LIST_NAME[2]` / `self.x[0]` | 仅当 base 是 `ast.List` 字面量、`file_str_list_consts`、或 `local_aliases` 解到 list literal，且 index 是 `Constant(int)`：取出第 index 元素后递归求值 | 不支持（list 元素本身是 list 的场景留给 fallback） | – |
| `BinOp(+, *, -, //)` | `n + 1` / `n * 2` / `n // 2` | 双侧递归 eval_int → 算术；除零返回 None；非 4 种运算符返回 None | `[a]+[b]` / `[x]*N` 见 §4.4 | – |
| `UnaryOp(-)` | `-1` | 子式 eval_int → 取负 | – | – |
| `Call` | `len([...])` / `int("6")` / `TransformerConfig(...)` | ① `len(<list>)` → eval_list_len；② `int(<literal>)` 退化；③ 已知 dataclass 构造 → eval_dataclass_fields → 不展开为单 int（返回 None） | – | dataclass 构造（见 §4.5） |
| `List` | `[a, b, c]` | – | 长度 = `len(elts)`；元素带 `Starred` 时递归求 list_len 累加 | – |
| `Starred` | `*self.x` | – | 子式 eval_list_len | – |
| 其他（IfExp, Lambda, ListComp, ...） | – | 返回 None（保守失败） | 同左 | – |

### 4.3 任意深度属性链算法（取代 P1 痛点）

```python
def _visit_Attribute_int(self, node: ast.Attribute, scope: Scope) -> Optional[IntValue]:
    # 拆出整条 chain：[base_node, attr_1, attr_2, ...]
    chain = self._flatten_attr_chain(node)   # → ('self', ['config', 'num_hidden_layers'])
    if chain is None:
        return None
    base, attrs = chain
    if isinstance(base, ast.Name) and base.id == 'self':
        return self._resolve_self_chain(attrs, scope)
    if isinstance(base, ast.Name):
        return self._resolve_local_chain(base.id, attrs, scope)
    return None

def _resolve_self_chain(self, attrs: List[str], scope: Scope) -> Optional[IntValue]:
    """对 self.<attrs[0]>.<attrs[1]>...<attrs[n]> 求 int 值。
    算法：
      1. 若 len(attrs)==1 → 走旧路径：self_to_param[attrs[0]] → ctor_kw_args[cls][param]
      2. 若 len(attrs)>=2：
         a. attrs[0] 必须是 self.X = formal_param X 的别名
            （从 self_to_param[attrs[0]] 拿到 param_name）
         b. 在 instance_const_chain[(parent_cls, parent_attr)][param_name] 里
            按 attrs[1:] 一路下钻
         c. 若任意层不存在 → 返回 None
    """
    cls_key = (scope.file, scope.cls)
    if len(attrs) == 1:
        return self._kw_lookup(scope.cls, attrs[0], scope)

    head = attrs[0]
    param_name = self.table.self_to_param.get(cls_key, {}).get(head)
    if param_name is None:
        return None

    # 取 instance_const_chain 中的 field 表
    inst_key = (scope.parent_cls, scope.parent_attr)
    chain = self.table.instance_const_chain.get(inst_key, {}).get(param_name)
    if chain is None:
        # fallback：跨 call site union（仅当全局唯一时返回）
        return self._param_field_union(scope.cls, param_name, attrs[1:])

    # 任意深度下钻
    cur: Any = chain
    for level, key in enumerate(attrs[1:]):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    if isinstance(cur, IntValue):
        return cur
    if isinstance(cur, int):
        return IntValue(cur, "dataclass_field")
    return None
```

**深度限制**：保守地把链长度上限设为 8（远超目前 5476790 中实际出现的 2 级；防御性兜底，不限制就有递归 hang 风险）。

### 4.4 list_len 求值路径

```python
def _visit_List_list_len(self, node, scope):
    total = 0
    for elt in node.elts:
        if isinstance(elt, ast.Starred):
            n = self.eval_list_len(elt.value, scope)
            if n is None: return None
            total += n
        else:
            total += 1
    return total

def _visit_BinOp_list_len(self, node, scope):
    if isinstance(node.op, ast.Add):
        l = self.eval_list_len(node.left, scope)
        r = self.eval_list_len(node.right, scope)
        if l is None or r is None: return None
        return l + r
    if isinstance(node.op, ast.Mult):
        # [x] * N or N * [x]
        for a, b in [(node.left, node.right), (node.right, node.left)]:
            la = self.eval_list_len(a, scope)
            ib = self.eval_int(b, scope)
            if la is not None and ib is not None and ib.value >= 0:
                return la * ib.value
    return None
```

### 4.5 dataclass 构造求值（取代 P2 痛点）

```python
def _visit_Call_fields(self, node: ast.Call, scope: Scope) -> Optional[Dict[str, IntValue]]:
    """识别 SomeDataclass(...) 调用，返回字段 dict。
    支持三类 args：
      - 位置实参：按 dataclass 字段定义顺序绑定
      - kw=<literal_int>：覆盖默认值
      - **<dict_literal_var>：从 file_dict_literals / local_aliases 拿 dict 字面量
    """
    cls_name = self._call_class_name(node.func)
    if cls_name is None:
        return None
    # 找到 dataclass 默认值
    defaults = None
    for (fname, cname), d in self.table.dataclass_defaults.items():
        if cname == cls_name:
            defaults = dict(d)
            break
    if defaults is None:
        return None

    fields = dict(defaults)  # copy

    # 1. 位置实参：需要拿 dataclass 字段顺序
    field_order = list(defaults.keys())
    for i, arg in enumerate(node.args):
        if isinstance(arg, ast.Starred):
            # *args 解包：未来可扩展，目前 return None 保守
            return None
        if i >= len(field_order):
            return None
        v = self.eval_int(arg, scope)
        if v is not None:
            fields[field_order[i]] = v

    # 2. kwargs
    for kw in node.keywords:
        if kw.arg is None:
            # **dict_literal 解包
            d = self._resolve_kwargs_unpack(kw.value, scope)
            if d is None:
                return None  # **kwargs 来源不明 → 保守失败
            for k, val in d.items():
                fields[k] = val
        else:
            v = self.eval_int(kw.value, scope)
            if v is not None:
                fields[kw.arg] = v
    return fields
```

### 4.6 跨 call site 参数传播完整链路

举例 `5476790/modelcode/`:

```python
# main_model.py
class InsTrans(torch.nn.Module):
    def __init__(self):
        ...
        self.ins_trans_base_params = {
            'num_hidden_layers': 6,
            'hidden_size': 512,
        }
        trans_conf = TransformerConfig(**self.ins_trans_base_params)
        self.ins_trans = SeqTrans(trans_conf)

# trans_refactor.py
@dataclass
class TransformerConfig:
    hidden_size: int = 512
    num_hidden_layers: int = 6
    ...

class SeqTrans(nn.Module):
    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.trans_blocks = nn.ModuleList([
            TransBlock(config, ...)
            for i in range(self.config.num_hidden_layers)
        ])
```

**新求值器执行流程**：

| 步骤 | 表/求值器 | 动作 | 结果 |
|---|---|---|---|
| Pass 1 | `dataclass_defaults` | 扫 `trans_refactor.py`，识别 `@dataclass class TransformerConfig` | `{("trans_refactor.py","TransformerConfig"): {"hidden_size": 512, "num_hidden_layers": 6, ...}}` |
| Pass 1 | `file_dict_literals` | 不直接收（这是 `self.X` 而非顶层 NAME） | – |
| Pass 3 | `local_aliases` | 扫 `InsTrans.__init__`：识别 `self.ins_trans_base_params = {...}`、`trans_conf = TransformerConfig(**self.ins_trans_base_params)` | `{("main_model.py","InsTrans","__init__"): {"trans_conf": <ast.Call>}}` |
| Pass 3 | `local_dataclass_inst` | 触发 `eval_dataclass_fields(<ast.Call(TransformerConfig, **self.ins_trans_base_params)>, scope)` | resolver 在求值过程中先解 `**self.ins_trans_base_params`：`_resolve_kwargs_unpack` 把 `self.ins_trans_base_params` 当作 `local_aliases` 不到的形式，转而看 `__init__` 中是否赋值过 dict literal —— 这一信息**必须额外预扫一遍**，写入 `local_self_dict_literals[(file,cls,method)]["ins_trans_base_params"] = {"num_hidden_layers": 6, ...}` |
| Pass 3 | `class_init_params` | 扫 `SeqTrans.__init__` 形参 | `[("trans_refactor.py","SeqTrans"): ["config"]]` |
| Pass 4 | `instance_const_chain` | 看到 `self.ins_trans = SeqTrans(trans_conf)`：`SeqTrans` 的形参 0 = `config`，trans_conf 是已知 dataclass instance → 写入 | `{("InsTrans","ins_trans"): {"config": {"num_hidden_layers": 6, ...}}}` |
| 主扫描 | `eval_int(<ast Attribute self.config.num_hidden_layers>, Scope(parent_cls="InsTrans", parent_attr="ins_trans"))` | `_resolve_self_chain` → attrs=["config","num_hidden_layers"] → self_to_param["config"]="config" → instance_const_chain["InsTrans","ins_trans"]["config"]["num_hidden_layers"] | **`6`** ✓ |

关键点：
- 这个求值过程中，`self.ins_trans_base_params` 这种 `self.X` 在 `__init__` 内部被赋值的 dict literal，需要**新增一张表** `local_self_dict_literals: Dict[(file,cls,method), Dict[attr, Dict[str, ast.expr]]]`，否则 `_resolve_kwargs_unpack` 拿不到。这张表与 `file_dict_literals` 平行，但 scope 不同。
- 这是当前实现完全没有的；新设计明确把它加入 ConstantTable。

---

## 5. 与 `_parse_local_*` 系列的关系

### 5.1 现状梳理

`build_static_module_tree` 中（行 5945~6150）有 12 个 `_parse_local_*` 局部函数：

| 函数 | 功能 | AST/Regex 性质 |
|---|---|---|
| `_parse_local_stmt(line)` | 单行字符串重新 parse 成 AST | **regex 风险点**（不是 regex，但本质是字符串 → 二次 parse，引入 lineno=1 bug） |
| `_parse_local_ctor_assign(stmt)` | 识别 `self.x = Cls(...)` | 纯 AST，输入是 stmt |
| `_parse_local_lg_assign(stmt)` | 识别 `self.x = LG.feature_column(...)` | 纯 AST |
| `_parse_local_fc_get_vector_assign(stmt)` | 识别 `self.x = fc_dict[...].get_vector(...)` | 纯 AST |
| `_parse_local_container_ctor(stmt)` | 识别 `self.x = nn.ModuleList(...)` | 纯 AST |
| `_parse_local_append_ctor(stmt)` | 识别 `self.x.append(Cls(...))` | 纯 AST |
| `_parse_local_subscript_ctor(stmt)` | 识别 `self.x[k] = Cls(...)` | 纯 AST |
| `_parse_local_var_ctor(stmt)` | 识别 `var = Cls(...)` | 纯 AST |
| `_parse_local_setattr_ctor(stmt)` | 识别 `setattr(self, name, Cls(...))` | 纯 AST |

### 5.2 迁移方案

| 函数 | 新去向 |
|---|---|
| `_parse_local_stmt` | **删除**。所有调用方改为：从 `ASTFrontend._get_method_node(cls, method).body` 直接拿 stmt 节点，不再走 line→parse 路径。 |
| `_parse_local_ctor_assign` 等 8 个 stmt 级 helper | **保留并搬迁到 `ASTFrontend` 的实例方法**（更名为 `extract_*_from_stmt(stmt)`），变成纯查询 helper。这样可以被 ConstantTable 在 Pass 3/4 复用，也保留对 build_dag_recursive 的支持。 |
| 函数中的 `_node_to_text` 局部 helper | **删除**。改为统一调用 `ASTFrontend._node_to_text`（已支持位置感知 fast path） |

### 5.3 迁移后调用样例

```python
# 现在
for raw in method_lines:
    stmt = _parse_local_stmt(raw)          # 字符串再 parse（坏）
    info = _parse_local_ctor_assign(stmt)
    if info: ...

# 新方案
method_node = fe._get_method_node(cname, mname)
for stmt in method_node.body:
    info = fe.extract_ctor_assign_from_stmt(stmt)   # 纯节点级
    if info: ...
```

---

## 6. 删除 regex 后的能力对照与补回方案

| 当前 regex 路径 | 行号 | 删除后是否退化 | 新求值器补救方案 |
|---|---|---|---|
| `_extract_kw_int_args` 主体（regex 提 `kw=ATOM`） | 5336-5361 | **会退化**（当前是兜底 + 双重保险） | Pass 3 `_scan_ctor_call_sites` 直接用 AST 取 `ast.Call.keywords`，对每个 `kw.value` 调 `eval_int` 求值 → 写入 `ctor_kw_args`。**完全等价**。 |
| `_extract_kw_list_lens` 主体（_split_top_level + regex 提 list） | 5522-5539 | **会退化** | 同上：`ast.Call.keywords` + `eval_list_len(kw.value, scope)`。**完全等价**。 |
| `_eval_list_len` 中的 `re.match(r'^[A-Za-z_]\w*$', expr)` 识别 NAME | 5507 | 退化 | 走 `_visit_Name_list_len`：① local_aliases；② file_str_list_consts；③ 跨文件 fallback。 |
| `_eval_list_len` 中的 `re.match(r'^(.+?)\s*\*\s*(.+)$', expr)` 识别乘法 | 5485 | 退化 | 走 `_visit_BinOp_list_len`，AST 直接给 `BinOp(*)`。 |
| `_resolve_range_n` / `_resolve_iter_len` 中 local_vars 的 regex 扫描 | 6722, 6777 (`r'^\s*(\w+)\s*=\s*([A-Z_]+|\d+)'`) | **会退化**（当前只支持 var=ATOM） | Pass 3 `_scan_local_aliases` 用 AST 扫所有 `ast.Assign(target=ast.Name)`，覆盖更广（var=expr 任意形式 → 求值时再展开）。**比当前更强**。 |
| `_DYN_IDX_RE` 扫 `self.x[idx]` 动态索引 | 6702 | 已有 ASTFrontend 等价方法 `get_dynamic_indexed_self_attrs` | 直接用，删 regex 路径。 |
| `_FOR_RANGE_RE` / `_FOR_ENUM_RE` / `_FOR_ITER_RE` 扫 for 循环 | 6783-6790 | 已有 `get_loop_expansion_records` | 让 AST 路径成为唯一路径，删 regex fallback。 |
| `_SELF_SET_RE.match(ln)` 提 self.attr=param | 6689 | 已有 `get_self_param_aliases` | 同上。 |
| 顶层 `re.match(r'^class (\w+)\(([^)]+)\)', line)` 提继承 | 5550 | 已有 `class_registry[*].bases` | 同上。 |
| `_build_class_map_regex` 等其它 regex fallback 函数 | 1641, 1704 | 已有 ASTFrontend 路径 | 如 ast_parse_unification_evaluation.md 推荐：直接删除整段 regex 函数体。 |

**删除原则**：所有标 `_warn_regex_fallback` 的代码块全删。任何"AST 优先 + regex 保底"模式改为"AST only，失败即 None"。

### 6.1 失败信号制度（替代 `_warn_regex_fallback`）

新增 `EvalTrace` 数据结构（debug 模式下启用）：

```python
@dataclass
class EvalTrace:
    failures: List[Tuple[Scope, str, str]] = field(default_factory=list)
    # (scope, expr_text, reason)

class ConstantResolver:
    def __init__(self, table, trace: Optional[EvalTrace] = None):
        ...
        self.trace = trace

    def _fail(self, node, scope, reason):
        if self.trace:
            self.trace.failures.append((scope, ast.unparse(node), reason))
        return None
```

调用方在 `--debug-eval` 启用时打印 trace；正常运行**不**打印（避免 stderr 噪声）。

---

## 7. 测例设计

### 7.1 目录结构

```
testset/synthetic_cases/ast_evaluator_redesign/
├── README.md
├── case01_basic_int_literal/
├── case02_file_const/
├── case03_global_const/
├── case04_local_alias_chain/
├── case05_self_attr_one_level/
├── case06_self_attr_two_level/
├── case07_self_attr_three_level/
├── case08_self_attr_arbitrary_depth/
├── case09_dataclass_default_only/
├── case10_dataclass_kw_override/
├── case11_dataclass_dict_unpack/
├── case12_cross_callsite_positional/
├── case13_cross_callsite_kwargs_mixed/
├── case14_multi_instance_isolation/
├── case15_arithmetic/
├── case16_list_literal_concat/
├── case17_list_mul/
├── case18_list_str_global/
├── case19_subscript_index/
├── case20_failure_runtime_load/
├── case21_failure_dynamic_setattr/
├── case22_circular_alias/
├── case23_deep_chain_8_levels/
└── case24_chain_through_helper_method/
```

每个 case 包含 `modelcode/model.py` + `expected.json`（断言新求值器在该 scope 下应返回的值）。

### 7.2 关键 case 详解

#### case04 — 局部别名链（补 P6）

```python
class M(nn.Module):
    def __init__(self):
        n = 6
        m = n
        k = m
        self.layers = nn.ModuleList([Block() for _ in range(k)])
```
**断言**：`eval_int(ast<k>, scope=("M","__init__")) == 6`，最终展开 6 个 Block。

#### case07 — 三级属性链

```python
@dataclass
class InnerCfg:
    depth: int = 4
@dataclass
class OuterCfg:
    inner: InnerCfg = field(default_factory=InnerCfg)
    width: int = 128

class Sub(nn.Module):
    def __init__(self, cfg: OuterCfg):
        self.cfg = cfg
        self.blocks = nn.ModuleList([B() for _ in range(self.cfg.inner.depth)])

class Parent(nn.Module):
    def __init__(self):
        cfg = OuterCfg(inner=InnerCfg(depth=4), width=128)
        self.sub = Sub(cfg)
```

**断言**：`eval_int(ast<self.cfg.inner.depth>, scope=parent="Parent",attr="sub") == 4`。
**当前实现失败**（一级硬编码），新求值器通过 `instance_const_chain` 嵌套 dataclass 解决。

#### case08 — 任意深度

```python
@dataclass
class L1: x: int = 7
@dataclass
class L2: a: L1 = field(default_factory=L1)
@dataclass
class L3: b: L2 = field(default_factory=L2)
@dataclass
class L4: c: L3 = field(default_factory=L3)
@dataclass
class L5: d: L4 = field(default_factory=L4)

class Net(nn.Module):
    def __init__(self, top: L5):
        self.top = top
        self.layers = nn.ModuleList([B() for _ in range(self.top.d.c.b.a.x)])

class Root(nn.Module):
    def __init__(self):
        self.net = Net(L5(d=L4(c=L3(b=L2(a=L1(x=7))))))
```

**断言**：5 级属性链解出 7。验证深度上限不在常规场景下意外阻断（depth=5 < limit=8）。

#### case12 — 跨 call site 位置实参（scenario5 主用例）

```python
@dataclass
class Cfg:
    n_layers: int = 6

class Sub(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.layers = nn.ModuleList([B() for _ in range(self.cfg.n_layers)])

class Parent(nn.Module):
    def __init__(self):
        c = Cfg(n_layers=6)
        self.sub = Sub(c)
```

**断言**：6 个 B；`instance_const_chain[("Parent","sub")] == {"cfg": {"n_layers": 6}}`。

#### case13 — kwargs 混合 + dict literal **解包**

```python
class Parent(nn.Module):
    def __init__(self):
        self.params = {'n_layers': 6, 'hidden': 256}
        self.sub = Sub(Cfg(**self.params, dropout=0.1))
```

**断言**：`local_self_dict_literals[("module.py","Parent","__init__")]["params"] = {"n_layers": <Constant 6>, "hidden": <Constant 256>}`，最终 `instance_const_chain[("Parent","sub")]["cfg"]["n_layers"] == 6`。

#### case14 — 多实例隔离

```python
class Parent(nn.Module):
    def __init__(self):
        self.a = Sub(Cfg(n_layers=6))   # 6 层
        self.b = Sub(Cfg(n_layers=2))   # 2 层
```

**断言**：
- `instance_const_chain[("Parent","a")]["cfg"]["n_layers"] == 6`
- `instance_const_chain[("Parent","b")]["cfg"]["n_layers"] == 2`
- DAG 中 `a` 展开 6 个 B，`b` 展开 2 个 B（**当前会被 union 成单一未知数**，新设计精确分开）。

#### case20 — 运行时配置加载（必须保守失败）

```python
def load_cfg() -> Cfg: ...

class Parent(nn.Module):
    def __init__(self):
        self.sub = Sub(load_cfg())
```

**断言**：`eval_dataclass_fields` 返回 None；`instance_const_chain` 不写入；DAG 走 wildcard 路径，**不报错、不静默错算**。

#### case22 — 循环别名（必须不 hang）

```python
class M(nn.Module):
    def __init__(self):
        a = b   # NameError 在运行时，但 AST 解析不报
        b = a
        self.x = nn.ModuleList([B() for _ in range(a)])
```

**断言**：递归保护命中，返回 `None`，不 hang。

#### case24 — 通过 helper 方法的链

```python
class Net(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_layers()

    def _build_layers(self):
        self.layers = nn.ModuleList([B() for _ in range(self.cfg.n_layers)])
```

**断言**：求值发生在 `_build_layers` scope，但 `self_to_param` 是从 `__init__` scope 收集的；新设计在 Pass 3 把 `self_to_param` 按 `(file, cls)` 而非 `(file, cls, method)` 存储，实现方法间共享。**当前实现也是这样做的，case24 主要是回归保护。**

### 7.3 测例运行框架

```python
# testset/synthetic_cases/ast_evaluator_redesign/run.py
def run_case(case_dir: Path):
    src = (case_dir / "modelcode" / "model.py").read_text()
    expected = json.loads((case_dir / "expected.json").read_text())

    ast_frontends = {"model.py": ASTFrontend(source=src, path="model.py")}
    table = ConstantTable.build_all(
        source_files={"model.py": src.splitlines()},
        ast_frontends=ast_frontends,
        nn_module_classes=expected["nn_module_classes"],
    )
    resolver = ConstantResolver(table)

    for assertion in expected["assertions"]:
        scope = Scope(**assertion["scope"])
        node = locate_ast_node(ast_frontends["model.py"], assertion["node_path"])
        if assertion["kind"] == "int":
            actual = resolver.eval_int(node, scope)
            assert (actual.value if actual else None) == assertion["expected"], \
                f"{case_dir.name} :: {assertion['name']}"
        elif assertion["kind"] == "list_len":
            ...
        elif assertion["kind"] == "fields":
            ...
```

**断言文件 `expected.json` 样例（case07）**：

```json
{
  "nn_module_classes": ["Sub", "Parent"],
  "assertions": [
    {
      "name": "self.cfg.inner.depth resolves to 4 in Sub.__init__",
      "kind": "int",
      "node_path": "Sub.__init__.body[1].iter.func.args[0]",
      "scope": {"file": "model.py", "cls": "Sub", "method": "__init__",
                "parent_cls": "Parent", "parent_attr": "sub"},
      "expected": 4
    }
  ]
}
```

### 7.4 集成回归

24 个 case 全部通过后，把它们挂到 `testset/test_dag_rules.py` 的 13 条规则之外，作为**第 13 条规则之后的独立测试集**：`test_ast_evaluator.py`，并在 CI 中和 7 模型 + 12 规则一起跑，作为防回退基准。

---

## 8. 删除 regex 后的影响分析

### 8.1 短期可能退化点（实现 PR 中必须处理）

| 退化点 | 原因 | 缓解 |
|---|---|---|
| 多行复合 ctor 参数（如 `Cls(\n  x=1,\n  y=[a,\n    b],\n)`）的 kw 提取 | 当前 regex 通过 `_join_logical_lines` 拼接处理；新方案直接用 AST，**完全无影响**。 | 无需缓解。 |
| 形如 `Cls(*args, **kwargs)` 的复杂调用 | regex 跳过；新方案：`*args` 时返回 None（保守），不写 ctor_kw_args | 等价于当前行为，无回归。 |
| f-string 内的 kw 形参（`Cls(**{f"k{i}": v})`） | regex 也无能力，新方案 `_resolve_kwargs_unpack` 同样返回 None | 等价。 |
| 类属性 `self.X = self.Y`（不是 ctor，但是 alias） | 当前完全不处理；新方案 `local_aliases` 会捕获 → 后续可被链式求值串起来 | **能力增强**，无回退。 |

### 8.2 性能影响

- **AST 节点缓存命中率提升**：cache key 从字符串改 `id(node)`，省 unparse 开销。
- **Pass 3 一次性扫**：对每个 (file,cls,method) 只走一遍 method 体；当前是每次进 for-range 循环现扫 local_vars，多次重复扫。
- **regex 删除**：直接砍掉所有 `re.compile` / `re.match` / `re.sub` 调用；ASTFrontend 已经把 AST 树缓存住，零增量解析成本。

预估全量跑（7 模型）：当前约 90s，新设计预估 60~75s（10~30% 加速；regex 部分占耗时的 15~20%）。

### 8.3 与 Rule 集兼容性

| Rule | 影响 |
|---|---|
| Rule1 / Rule1b / Rule1c | **零影响**（边的源码 evidence 由独立路径生成，不依赖求值器） |
| Rule2 / Rule2-strict | **轻度改善**（容器精确剪枝，少掉的伪节点也少了伪边） |
| Rule3 | **零影响**（无新边） |
| Rule5 / Rule6 / Rule6_out | **轻度改善**（boundary 边数量更精确） |
| NoInputPierce / NoDangling | **轻度改善** |
| ProdConsCompleteness | **零影响** |
| TrainInferGroupConsistency | **轻度改善**（多实例隔离消除单边塌陷） |

### 8.4 防回退基准

- 7 模型 + 1 trace 的 12 条规则**全部继续 PASS**，关键节点数防回退基准维持：
  - 5476790 nodes=229；5547919 nodes=246；5698781/5820572 nodes=243/internal_edges=215；5800920 nodes=166；5758056 nodes=40。
- 24 个新 case **必须 100% PASS**。
- 旧合成测例 (`bug1_dense_tower_expand`, `bug2_parallel_layernorm`) **必须继续 PASS**。
- 任何"PASS → FAIL" 视为阻塞性回退。

---

## 9. 实施次序（建议拆分 PR）

| PR | 范围 | 行数 | 风险 | 备注 |
|---|---|---|---|---|
| PR1 | `ConstantTable` + `ConstantResolver` 骨架（不接入主逻辑），单元测试 24 个 case 通过 | +800 | 低（无现网调用方） | 通过后才动主逻辑 |
| PR2 | 接入 PR1，**保留** 旧求值器作为 fallback；新求值器先以 `--use-new-eval` flag 启用，对 7 模型做 A/B 比对 | +200 | 中 | A/B byte-for-byte 等价后再上 |
| PR3 | 删除 `_eval_int_atom`/`_eval_list_len`/`_extract_kw_int_args`/`_extract_kw_list_lens`/`_resolve_range_n`/`_resolve_iter_len` 共 6 个旧函数，删除所有 `_warn_regex_fallback` 调用与对应 regex 路径 | -600 | 中 | 必须 PR2 A/B 全绿才能合 |
| PR4 | `_parse_local_stmt` 删除，`_parse_local_*` 系列搬到 `ASTFrontend`；`_node_to_text` 局部 helper 全部删除 | ±300 | 低 | ast_parse_unification_evaluation 推荐项 |
| PR5 | `local_self_dict_literals` 注入；scenario5 的端到端 case12+case13 拉通 5476790 真实 commit 的 transformer 块精确化 | +100 | 中 | 验收：silent over-expansion 修复 |

---

## 10. 总结

| 设计目标 | 是否达成 | 实现机制 |
|---|---|---|
| ① 删除所有 regex 后正确性不回退 | **是** | Pass 3 用 AST 收集 ctor kwargs / local_vars / self_to_param，覆盖 + 超过当前 regex 能力；防回退靠 24 case + 12 规则 + A/B 比对 |
| ② 任意深度属性链求值 | **是** | `_resolve_self_chain` 用 instance_const_chain 嵌套 dict 一路下钻，深度上限保守设 8 |
| ③ 跨 call site 参数值传播 | **是** | Pass 4 `_bind_param_value_chains` + `class_init_params` 形参绑定 + `local_dataclass_inst` 局部 dataclass 实例 |
| ④ memo / DP | **是** | `_eval_cache` 按 `id(node)` + scope 维度做 key；递归保护用占位符 |
| ⑤ 失败明确不兜底 | **是** | 求值器失败一律 `Optional[T] = None`，调用方走原有 wildcard / 不展开路径；`EvalTrace` 在 debug 模式记录失败原因 |

**核心架构创新点**：
1. **ConstantTable + ConstantResolver 双层**：表与求值器分离，事实表先 build 后 read，求值器纯函数。
2. **Scope 一等公民**：每次求值带上完整 scope（file/cls/method/parent_cls/parent_attr），消除"漏传上下文导致 silent fallback"的 bug。
3. **AST 节点入参**：所有求值入口接 `ast.expr` 而非 `str`，杜绝二次 parse 与 lineno=1 bug。
4. **三类常量统一**：字面量、ctor kwargs、dataclass 字段在求值器内部走同一套 dispatcher，没有"int 走一条路、list 走另一条路、dataclass 第三条路"的分裂。

**风险与缓解**：
- R1：跨 call site 形参绑定误判（位置实参与可变实参混用）→ 任何 `*args` / `**unknown` 一律保守 None。
- R2：AST 节点 `id()` 复用风险（理论上不会在同 build 周期内 GC）→ 显式持有 `ast_frontends` 强引用。
- R3：dataclass 字段嵌套求值 hang → 深度上限 8 + 占位符递归保护。
- R4：现网类对 regex 的 corner case 依赖（如 `Cls(msg="kw=2")` 这种字符串中带 = 的）→ AST 路径天然免疫，反而比 regex 更稳。

**预期收益**：
- 修复 5476790 等 commit 的 silent over-expansion bug（多实例时不同 `num_hidden_layers` 被错误 union）。
- 删除 ~800 行 regex 代码；新增 ~800 行 AST 求值代码；净行数持平，但**可维护性显著提升**。
- 测试覆盖从"集中在 7 个 commit baseline"扩展到"24 个最小化合成 case + 7 baseline"，可定位精度提升一个数量级。
