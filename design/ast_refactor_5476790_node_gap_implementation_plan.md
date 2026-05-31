# 5476790 节点缺口实现方案

> 范围：本文件只给出实现方案，不修改代码。等 NaN 确认后，再进入编码实现。
>
> 输入依据：`design/ast_refactor_5476790_node_gap_test_design.md` 已确认的 A1/A2/A3/B1/C1/C2 测例设计。
>
> 总原则：AST-only；禁止正则兜底；严禁 annotation fallback；dataclass default 属于真实 `Config()` 构造语义，允许在真实 caller 链中参与解析。

---

## 1. 实施目标

当前 5476790 节点数 `221 -> 229`，差 8 个节点。实现目标分两层：

1. **测试层先落地**：把 A1/A2/A3/B1/C1/C2 变成可执行防回退测例。
2. **代码层再修复**：针对三类根因改 `analyze_trace.py`：
   - 根因 A：`range(self.config.num_hidden_layers)` 的真实 caller 链解析。
   - 根因 B：`self.attr.helper(...)` 作为 submodule use，避免 dead-child 误裁。
   - 根因 C：DomainEmbedding 直接注册和内部 `nn.Embedding` 子模块在真实 5476790 中稳定保留。

验收标准：

1. 新增 synthetic 测例全部 PASS。
2. `5476790` 的 DomainEmbedding coverage PASS。
3. 全量 `python3 testset/test_dag_rules.py` 保持 7/7 PASS。
4. 修复后 `5476790 nodes >= 229`。
5. 无 annotation fallback；A2 无 runtime 配置时必须 warning/error；补 runtime 配置后必须能解析。

---

## 2. 改动文件清单

<table header-row="true" header-col="false" col-widths="260,180,420">
  <tr>
    <td>文件</td>
    <td>改动类型</td>
    <td>改动内容</td>
  </tr>
  <tr>
    <td>`develop/ast/storage/testset/test_dag_rules.py`</td>
    <td>新增测例</td>
    <td>落地 A1/A2/A3/B1/C1/C2；新增 runtime_overrides 二阶段断言；新增 5476790 DomainEmbedding coverage。</td>
  </tr>
  <tr>
    <td>`develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py`</td>
    <td>修复实现</td>
    <td>ConstantTable / ConstantResolver / build_static_module_tree / ASTFrontend helper call 识别 / reachability 过滤修复。</td>
  </tr>
  <tr>
    <td>`develop/ast/torch-trace-analyzer/design/ast_refactor_5476790_node_gap_implementation_plan.md`</td>
    <td>设计文档</td>
    <td>本文档。</td>
  </tr>
</table>

编码前需要按安全规则复制基线：

```bash
cd develop/ast/torch-trace-analyzer
cp torch-trace-analyzer/scripts/analyze_trace.py torch-trace-analyzer/scripts/analyze_trace.py_5476790_node_gap_BASELINE.bak

cd ../storage
cp testset/test_dag_rules.py testset/test_dag_rules.py_5476790_node_gap_BASELINE.bak
```

验证通过后删除 `.bak` 临时文件，不提交备份。

---

## 3. 测试实现方案

### 3.1 A1/A2/A3：ConstantResolver / ConstantTable synthetic

**落地位置**：`develop/ast/storage/testset/test_dag_rules.py` 的 `AST synthetic regression tests (P8)` 区域。

当前文件已有：

- `_SYNTH_MODULELIST_LISTCOMP_STRLIST`
- `_SYNTH_CONFIG_CHAIN_DICT_SPREAD`
- `_SYNTH_HELPER_METHOD_ATTR_USE`
- `run_ast_synthetic_tests()`

建议新增三组 inline source 字符串：

1. `_SYNTH_CONFIG_CHAIN_DICT_SPREAD_EXPLICIT`
   - 对应 A1。
   - `Config.num_hidden_layers` default 故意设为 `999`。
   - `Root.params = {"num_hidden_layers": 6}`。
   - `self.stack = Stack(Config(**self.params))`。

2. `_SYNTH_CONFIG_ANNOTATION_RUNTIME_ONLY`
   - 对应 A2。
   - `Stack.__init__(config: Config)`。
   - `Root.__init__(runtime_config)` 中 `self.stack = Stack(runtime_config)`。
   - 无 runtime overrides 阶段必须返回 `None` + warning/error。
   - 有 runtime overrides 阶段必须返回 `6`。

3. `_SYNTH_CONFIG_DATACLASS_DEFAULT_CALL`
   - 对应 A3。
   - `Root.__init__` 中 `self.stack = Stack(Config())`。
   - `Config.num_hidden_layers: int = 6`。
   - 必须解析为 6 并展开 `blocks[0..5]`。

**辅助函数设计**：

新增或复用现有 `_load_analyze_module()` 逻辑，直接 import 当前 `ANALYZE_SCRIPT_OVERRIDE` 对应脚本，调用：

```python
frontends = analyze._build_ast_frontends(source_files)
class_map = analyze._build_class_map(source_files, ast_frontends=frontends)
nn_classes = analyze._collect_nn_module_classes(source_files, class_map, ast_frontends=frontends)  # 若已有同类函数则复用
constant_table = analyze.ConstantTable.build_all(source_files, frontends, nn_classes)
resolver = analyze.ConstantResolver(constant_table)
```

如果现有脚本没有公开 `_collect_nn_module_classes`，测试侧不新增生产逻辑，改为调用 `build_static_module_tree()` 后从内部结构验证；或在测试侧用 `class_map`/继承信息做最小 AST-only 收集。

**断言方案**：

- A1：
  - `resolver.eval_int(self.config.num_hidden_layers, Scope(... parent_cls="Root", parent_attr="stack")) == 6`。
  - `build_static_module_tree()` 后 `Stack.attrs` 包含 `blocks[0]..blocks[5]`，不含 `blocks[6]`。
  - 不能返回 default `999`。

- A2 两阶段：
  - Phase 1：无 `runtime_overrides`。
    - resolver 返回 `None`。
    - resolver 或 diagnostics 收集器出现 warning/error，内容包含 `num_hidden_layers`、`runtime`、`explicit` 或中文等价提示。
    - tree 不展开 `blocks[0..5]`。
  - Phase 2：传 `runtime_overrides`。
    - resolver 返回 `6`。
    - tree 展开 `blocks[0]..blocks[5]`。

- A3：
  - `Stack(Config())` 解析 default 得到 `6`。
  - tree 展开 `blocks[0]..blocks[5]`，不含 `blocks[6]`。

### 3.2 B1：helper-method attr use synthetic

**落地位置**：继续复用 `_SYNTH_HELPER_METHOD_ATTR_USE`，但补强断言。

当前实现已经有 `_extract_called_self_attr()` 的 helper 链逻辑雏形，需要测试把行为固定下来。

新增断言：

1. `tree['Wrapper']['first_call_loc']['bb']` 存在，line 指向 `self.bb.custom_query(x)`。
2. DAG JSON 中 `Backbone` 不被裁剪。
3. 若产生边，edge evidence 不允许退回 `__init__` 构造行。

### 3.3 C1：DomainEmbedding 多文件 synthetic

**落地方式**：建议在 `test_dag_rules.py` 中新增临时多文件 source dict，而不是先落目录文件。理由是 A1/A2/A3/B1 已经是 inline synthetic，集中维护更易读。

source_files：

```python
{
  "model_common.py": [...DomainEmbedding...],
  "main_model.py": [...ApgStar/SequenceModule/SeqBias/LogitAdaptive/Root...],
}
```

断言：

1. `tree['DomainEmbedding']['attrs']['domain_embedding']` 存在，值为 `Embedding` 或 `nn.Embedding` 等价表示。
2. 四个父类 attr 存在：
   - `ApgStar.gen_domain_embedding`
   - `SequenceModule.seq_ads_domain_embeddings`
   - `SeqBias.bias_domain`
   - `LogitAdaptive.la_domain`
3. `first_call_loc` 指向各自 forward 中 `self.xxx(x)`。
4. DAG JSON 生成后四个 attr 不能被 reachability 裁掉。

### 3.4 C2：5476790 真实防回退断言

**落地位置**：`test_dag_rules.py` 中新增 `check_5476790_domain_embedding_coverage(data)`。

触发条件：`model_name == "5476790"`。

硬断言列表：

```python
required = {
    "ApgStar": ["gen_domain_embedding"],
    "SequenceModule": ["seq_ads_domain_embeddings"],
    "SeqBias": ["bias_domain"],
    "LogitAdaptive": ["la_domain"],
    "DomainEmbedding": ["domain_embedding"],
}
```

实现注意：

- 不依赖 HTML 文案，直接查 DATA JSON 中 nodes/groups 的 class/name/attr/attr_id_map。
- 对结构字段做兼容读取，例如 `class_name`、`label`、`attrs`、`attr_id_map`、`children`。
- 失败时输出缺失列表，方便定位。

---

## 4. 生产代码实现方案

### 4.1 移除 annotation fallback，只保留真实 caller 链

**当前风险位置**：`analyze_trace.py` 中 `ConstantTable.__init__` 附近已经存在：

```python
self.class_init_param_anno: Dict[Tuple[str, str], Dict[str, str]] = {}
```

且注释写明“用于 fallback 读取 dataclass default”。这与 NaN 明确要求冲突。

**存储来源说明**：`class_init_param_anno` 存的是 `__init__(config: Config)` 这种**参数类型注解**，它本身不是 runtime 值，也不是构造调用结果。通过它去找 `Config` 并读取字段默认值，属于 annotation fallback，要删。

**改动方案**：

1. 保留字段可以作为注释/调试信息，但不得参与 resolver 解析。
2. 如果当前 `_resolve_self_chain()` 或其他路径引用 `class_init_param_anno` 走 annotation fallback，必须删除该分支。
3. 若字段完全无使用，可同步删除 `class_init_param_anno` 的填充逻辑，避免后续误用。

**函数影响**：

- `ConstantTable._scan_class_init_params()`
- `ConstantResolver._resolve_self_chain()`
- 可能的 `_resolve_bound_dataclass_fields()` / `_resolve_dataclass_call_fields()`

**验收**：A2 Phase 1 不会从 `config: Config` 解析出 6。

### 4.2 支持 dataclass default 作为真实 `Config()` 构造语义

当前 `ConstantResolver._eval_fields_uncached()` 已经：

```python
fields = dict(defaults)
```

这符合 A3 positive：`Config()` 应得到 dataclass defaults。

**存储来源说明**：这里的 `defaults` 来自 `ConstantTable.dataclass_defaults`，由 `_scan_dataclass_defaults()` 扫描 `@dataclass class Config: num_hidden_layers: int = 6` 的**字段真实默认赋值**得到。只有当源码里存在真实 caller 调用 `Config()` / `Config(**kwargs)` 时，`dict(defaults)` 才作为 Python 构造语义的一部分参与解析；这和 4.1 的参数注解存储不是同一个逻辑。

需要补强的是调用链传播：

1. `_resolve_dataclass_call_fields(Config())` 必须返回 defaults。
2. `_resolve_bound_dataclass_fields()` 在 parent call-site `Stack(Config())` 时必须把 `{num_hidden_layers: 6}` 写入：

```python
instance_const_chain[("Root", "stack")]["config"]["num_hidden_layers"] = IntValue(6, "dataclass_field")
```

3. `_resolve_self_chain(["config", "num_hidden_layers"], Scope(cls="Stack", parent_cls="Root", parent_attr="stack"))` 读取上述 chain。

**涉及函数**：

- `ConstantTable._pass4_callsite_bind()`
- `ConstantTable._resolve_bound_dataclass_fields()`
- `ConstantTable._resolve_dataclass_call_fields()`
- `ConstantResolver._eval_fields_uncached()`
- `ConstantResolver._resolve_self_chain()`

**验收**：A3 PASS。

### 4.3 支持 `**self.dict_attr` 的真实 caller 链

A1 和 5476790 都依赖：

```python
self.params = {"num_hidden_layers": 6}
self.stack = Stack(Config(**self.params))
```

当前 `ConstantTable._pass4_callsite_bind()` 已有 `local_self_dict_literals` 和 `**self.<attr>` 展开逻辑；实现要确保两层都通：

1. Parent `Root.__init__` 中 `self.params = {...}` 被 `_scan_self_dict_literals()` 捕获。
2. `Config(**self.params)` 被 `_resolve_dataclass_call_fields()` 的 `_resolve_to_dict_items()` 解析。
3. `Stack(Config(**self.params))` 绑定到 `Stack.__init__(config)`。
4. 写入 `instance_const_chain[("Root", "stack")]["config"]`。

**修复策略**：

- 如果 `Config(**self.params)` 解析失败，优先补 `_resolve_to_dict_items()` 对 `self.<dict_attr>` 的作用域读取。
- 确认 `scope.method_key` 使用的是 parent class 的 `__init__`，不是 child class。
- 不增加正则 fallback。

**验收**：A1 返回 6，不返回 999。

### 4.4 A2 runtime_overrides 接口与 diagnostics

A2 需要两阶段行为：无配置失败并提示；补配置后完成解析。

#### 4.4.1 接口设计

给 `ConstantResolver` 新增可选参数：

```python
class ConstantResolver:
    def __init__(self, table: ConstantTable, runtime_overrides: Optional[Dict[str, int]] = None, diagnostics: Optional[list] = None):
        self.table = table
        self.runtime_overrides = runtime_overrides or {}
        self.diagnostics = diagnostics if diagnostics is not None else []
```

`build_static_module_tree()` 增加可选参数：

```python
def build_static_module_tree(source_files, preferred_root=None, conditional_mode="infer", runtime_overrides=None):
```

内部创建 resolver 时传入：

```python
_new_eval_resolver = ConstantResolver(_new_const_table, runtime_overrides=runtime_overrides, diagnostics=_static_diagnostics)
```

同时 tree 中可记录：

```python
tree[cname]["static_diagnostics"] = _static_diagnostics
```

或者统一返回/输出到已有 summary。测试阶段至少能从 resolver 实例读取 diagnostics。

#### 4.4.2 override key 规范

建议支持三种 key，按从精确到宽松匹配：

1. `"Stack.config.num_hidden_layers"`
2. `"Stack.stack.config.num_hidden_layers"` 或 `"Root.stack.config.num_hidden_layers"`
3. `"num_hidden_layers"`

内部标准化为候选列表：

```python
candidates = [
    f"{scope.cls}.{head}.{tail}",
    f"{scope.parent_cls}.{scope.parent_attr}.{head}.{tail}",
    tail,
]
```

对于 A2，`head=config`，`tail=num_hidden_layers`。

#### 4.4.3 resolver 查找顺序

在 `_resolve_self_chain()` 中：

1. 先走真实 caller 链 `instance_const_chain`。
2. 若无 chain，查 `runtime_overrides`。
3. 若仍无，记录 warning/error 并返回 `None`。
4. 不走 annotation fallback。

伪代码：

```python
if chain is None:
    override = self._lookup_runtime_override(attrs, scope)
    if override is not None:
        return IntValue(override, "runtime_override")
    self._warn_unresolved_runtime_config(attrs, scope)
    return None
```

`_warn_unresolved_runtime_config()` 去重，避免同一表达式刷屏。

warning 内容建议：

```text
Cannot statically resolve Stack.config.num_hidden_layers for Root.stack. Please provide runtime_overrides or explicit constructor arguments.
```

**验收**：A2 Phase 1 有 warning/error；Phase 2 返回 6 并展开。

### 4.5 ModuleList range 展开接入 runtime_overrides

当前 `build_static_module_tree()` 内部 `_eval_int_node()` 调用：

```python
iv = _new_eval_resolver.eval_int(expr_node, _resolver_scope(...))
```

修改后 `_new_eval_resolver` 带 `runtime_overrides` 即可。需要检查所有 `range(...)` 展开入口是否都走 `_eval_int_node()`：

- listcomp `ModuleList([Block() for _ in range(...)])`
- for-loop append / setattr 展开
- container prune 相关 per-instance 展开

若存在独立 evaluator，需要统一接入 `ConstantResolver`，不要单独写解析逻辑。

### 4.6 B1 helper-method attr use 修复

当前 `ASTFrontend._extract_called_self_attr()` 已有 helper 链支持：

```python
self.<attr>.<helper>(...) -> <attr>
self.<attr>[idx].<helper>(...) -> <attr>[idx]
```

但仍需确认以下调用点都使用它：

1. `get_module_calls()` / `_get_stmt_infos()` 识别调用点。
2. `_method_has_attr_call()` helper reachability 判断。
3. call-count prepass。
4. dataflow `_rhs_last_called_attr()`。
5. `get_first_call_loc()`。

**风险点**：`_method_has_attr_call()` 目前显式判断 `self.<attr>(...)` 和 `self.<attr>[...]`，可能没有调用 `_extract_called_self_attr()`，导致 `self.bb.custom_query(x)` 不算 attr use。

**改动方案**：

- 把 `_method_has_attr_call()` 中手写判断替换/补充为：

```python
called_attr = _ast_fe_dd._extract_called_self_attr(sub.func)
if called_attr:
    base = called_attr.split("[", 1)[0]
    if called_attr in attrs or base in attrs:
        return True
```

- call-count prepass 同样复用 `_extract_called_self_attr()`，避免漏掉 helper method chain。

**验收**：B1 PASS；5476790 中 `seq_trans` / `ins_trans` 不因 helper 调用被裁。

### 4.7 C1/C2 DomainEmbedding 保留修复

DomainEmbedding 真实结构：

```python
class DomainEmbedding(nn.Module):
    def __init__(...):
        self.domain_embedding = nn.Embedding(...)
```

真实父类直接注册：

- `ApgStar.gen_domain_embedding = DomainEmbedding(...)`
- `SequenceModule.seq_ads_domain_embeddings = DomainEmbedding(...)`
- `SeqBias.bias_domain = DomainEmbedding(...)`
- `LogitAdaptive.la_domain = DomainEmbedding(...)`

**需要检查的实现点**：

1. `_AST_ModuleTreeExtractor` 是否跨文件识别 `from model_common import DomainEmbedding`。
2. `class_map` 是否包含 `DomainEmbedding`。
3. `nn_module_classes` 是否把 `DomainEmbedding` 识别为 nn.Module 子类。
4. tree 构建是否记录四个父类 attr。
5. reachable root 过滤是否误删 `DomainEmbedding` 或四个父类 attr。
6. `first_call_loc` 是否找到 forward 中 `self.xxx(...)`。

**修复策略**：

- 优先修 class_map / nn_module_classes 识别，不在 DAG 后处理硬加节点。
- 若 parent attr 存在但 child tree 被裁，修 reachability 收集逻辑，让通过 parent attrs 可达的 child class 保留。
- 若 `DomainEmbedding.domain_embedding` 缺失，修 `nn.Embedding` 内置模块注册识别路径。

**禁止做法**：

- 不允许针对 `DomainEmbedding` 名字做硬编码补节点。
- 不允许用正则从源码文本扫 `DomainEmbedding(`。
- 不允许 HTML 层补假节点。

---

## 5. 函数签名变更汇总

<table header-row="true" header-col="false" col-widths="260,300,300">
  <tr>
    <td>函数/类</td>
    <td>当前签名</td>
    <td>目标签名/新增能力</td>
  </tr>
  <tr>
    <td>`ConstantResolver.__init__`</td>
    <td>`__init__(self, table)`</td>
    <td>`__init__(self, table, runtime_overrides=None, diagnostics=None)`</td>
  </tr>
  <tr>
    <td>`ConstantResolver.eval_int`</td>
    <td>`eval_int(self, node, scope)`</td>
    <td>签名可不变；通过 resolver 实例携带 overrides。若更清晰，也可加 keyword-only `runtime_overrides=None`，但优先避免大范围改调用点。</td>
  </tr>
  <tr>
    <td>`build_static_module_tree`</td>
    <td>`build_static_module_tree(source_files, preferred_root=None, conditional_mode="infer")`</td>
    <td>`build_static_module_tree(source_files, preferred_root=None, conditional_mode="infer", runtime_overrides=None)`</td>
  </tr>
  <tr>
    <td>`_eval_int_node` 内部闭包</td>
    <td>无 overrides</td>
    <td>使用携带 overrides 的 `_new_eval_resolver`，无需单独加参数。</td>
  </tr>
  <tr>
    <td>`ASTFrontend._extract_called_self_attr`</td>
    <td>已支持部分 helper chain</td>
    <td>作为唯一 self-attr call 识别入口，供 helper reachability / call-count / dataflow / first_call_loc 复用。</td>
  </tr>
</table>

---

## 6. 实施顺序

1. **备份基线**
   - `analyze_trace.py_5476790_node_gap_BASELINE.bak`
   - `test_dag_rules.py_5476790_node_gap_BASELINE.bak`

2. **先写测例，不修代码**
   - 落 A1/A2/A3/B1/C1/C2。
   - 预期部分失败，用失败结果确认测例有效。

3. **修根因 A**
   - annotation fallback 移除。
   - dataclass default 保留。
   - `**self.dict` caller chain 修正。
   - runtime_overrides + diagnostics。

4. **修根因 B**
   - `_extract_called_self_attr()` 统一接入 helper reachability / call-count / dataflow。

5. **修根因 C**
   - 跨文件 DomainEmbedding 注册、内置 `nn.Embedding` 子模块、reachable 保留。

6. **验证**
   - 单独跑新增 synthetic。
   - 跑 5476790。
   - 跑全量 7/7。

7. **清理**
   - 删除 `.bak` 临时备份。
   - 如生成 HTML，按基线验证规范打包返回。

---

## 7. 验证命令

单测入口：

```bash
cd develop/ast/storage
ANALYZE_SCRIPT_OVERRIDE=/workspace/iris_a493ea8b-53dc-4055-b5a7-cca66d850ce9/develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py \
  python3 testset/test_dag_rules.py testset/extracted/5476790
```

全量回归：

```bash
cd develop/ast/storage
ANALYZE_SCRIPT_OVERRIDE=/workspace/iris_a493ea8b-53dc-4055-b5a7-cca66d850ce9/develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py \
  python3 testset/test_dag_rules.py
```

如果 synthetic runner 也接入：

```bash
cd develop/ast/storage
ANALYZE_SCRIPT_OVERRIDE=/workspace/iris_a493ea8b-53dc-4055-b5a7-cca66d850ce9/develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py \
  python3 testset/synthetic_cases/run_synthetic_tests.py
```

---

## 8. 风险与回退

<table header-row="true" header-col="false" col-widths="240,360,280">
  <tr>
    <td>风险</td>
    <td>表现</td>
    <td>回退/防护</td>
  </tr>
  <tr>
    <td>runtime_overrides key 过宽</td>
    <td>多个类都有 `num_hidden_layers` 时误用同一个 override</td>
    <td>优先匹配精确 key；宽松 key 只在唯一候选时生效，并输出 diagnostics。</td>
  </tr>
  <tr>
    <td>diagnostics 刷屏</td>
    <td>同一 unresolved 表达式在多个 range 调用中反复 warning</td>
    <td>resolver 内用 set 去重。</td>
  </tr>
  <tr>
    <td>helper call 识别过宽</td>
    <td>`self.not_module.helper()` 被误认为模块 use</td>
    <td>必须以 `known_attrs` / `attrs` 为白名单。</td>
  </tr>
  <tr>
    <td>DomainEmbedding 被硬编码</td>
    <td>只修 5476790，其他模型可能退化</td>
    <td>必须通过通用 class_map / nn_module_classes / reachable 逻辑修，不写类名特判。</td>
  </tr>
  <tr>
    <td>全量回归失败</td>
    <td>7/7 失败或已有规则回退</td>
    <td>立即从 `.bak` 恢复，不在破损文件上叠加修改。</td>
  </tr>
</table>

---

## 9. 预期提交内容

等 NaN 确认后，编码阶段预期产生：

1. `analyze_trace.py`：实现根因 A/B/C 修复。
2. `test_dag_rules.py`：新增 A1/A2/A3/B1/C1/C2 防回退测例。
3. 测试报告：
   - 新增 synthetic PASS 列表。
   - 5476790 nodes 对比。
   - 全量 7/7 PASS。
4. 生成 HTML 打包件：按基线验证规范，验证通过后自动打包返回。
