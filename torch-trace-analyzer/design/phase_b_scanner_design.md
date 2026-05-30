# Phase B Scanner 设计分析

## 0. 结论摘要

scanner 这次确实是先编码后补设计，流程需要纠正。基于当前模型源码与已有实现，scanner 需要解决的不是简单地“扫几个 LG API 调用”，而是要把 lagrange_torch 的输入源与结果源提升为 DAG 构图里的**一等 Attr**：

1. **InputAttr**：识别 `LG.feature_column` / `LG.dense_feature` / `LG.get_sample_rate` / `LG.get_bias` / `LG.label` 等由 lagrange_torch CONTEXT 创建的输入源，并记录“定义位置”和“forward 使用位置”。
2. **ResultAttr**：识别 `LG.result()` 返回对象上的 `.head(...)` 调用，把模型输出头作为 Result 节点来源，而不是靠前端或旧逻辑后补。
3. **ASTFrontend 模块化**：当前 `ASTFrontend` 已经包含大量 AST 元数据、调用点、变量数据流和局部语句解析能力，适合从 `analyze_trace.py` 拆到 `scripts/ast_frontend.py`，让 scanner、静态树构图、ConstantTable 都复用同一套 AST 前端能力。

当前 `scripts/attr_scanner.py` 的实现可以作为 Phase A 试验原型，但还不足以作为正式设计：它只支持直接赋值和简单 `result.head`，没有处理 alias、链式 `.head`、dict/container、import alias、跨 helper 方法、LG source 使用位置的语义边界等问题。

---

## 1. 阅读范围与事实来源

### 1.1 已确认的代码位置

- 工作分支：`develop/ast/torch-trace-analyzer/torch-trace-analyzer`，当前分支为 `feat/ast-timing-fix`。
- storage 测试源码：`develop/ast/storage/testset/extracted/`，当前分支为 `feat/ast-step2-tests`。
- 当前 scanner 原型：`scripts/attr_scanner.py`。
- Attr 数据结构：`scripts/attr_types.py`。
- 主流程 AST 前端：`scripts/analyze_trace.py` 中 `ASTFrontend`，范围为 **135-1239 行**。

### 1.2 lagrange_torch 包定位结果

在 `testset/extracted/` 下没有找到完整的 `bytedance/lagrange_torch` 包源码目录。能直接阅读到的是模型源码里的使用方式，以及 mock/unit test 里对 lagrange_torch CONTEXT 的轻量 fake：

- `5820572/modelcode/cvr_grouped_task_tower_test.py` 中 `_FakeContext.dense_feature()` 返回一个 `nn.Module`，其 `forward()` 返回 tensor；`_FakeContext.get_sample_rate()` 同样返回一个 feature-like module。
- `5820572/modelcode/cvr.py`、`5547919/modelcode/main_model.py`、`5820572/modelcode/models/sparse_module.py` 展示了真实模型里 `LG.feature_column`、`LG.dense_feature`、`LG.get_sample_rate`、`LG.result().head(...)` 的典型用法。

因此，本报告对 API 语义的总结以“模型调用语义 + mock 中可执行语义”为准，不假设未读到的框架内部实现细节。

---

## 2. LG input source API 语义总结

### 2.1 `LG.dense_feature(name, dim, dtype)`

**作用**：声明一个 dense 特征输入模块。模型中通常在 `__init__` 中赋给 `self.xxx`，在 `forward()` 或 helper 中通过 `self.xxx()` 取出输入 tensor。

**典型代码**：

```python
self.v3l3tag_module = LG.dense_feature('fc_line_id_live_slice_v3l3tag', 1, torch.int64)
self.is_diffusion_module = LG.dense_feature('fc_line_id_is_diffusion_instance', 1, torch.int64)

fc_line_id_live_slice_v3l3tag = self.v3l3tag_module().view(-1)
fc_line_id_is_diffusion_instance = self.is_diffusion_module().view(-1)
```

**返回语义**：从 mock 代码看，它返回一个 `nn.Module` 子类实例，调用 `forward()` 后返回 tensor。真实框架里应可视为“输入源 Module”。

**scanner 关注点**：

- 定义位置是 `__init__` 中 `self.attr = LG.dense_feature(...)` 的调用行。
- 真实数据进入 DAG 的位置是 `forward()` 中 `self.attr()` 的调用行。
- 需要同时记录这两个位置：`call_loc` 用于节点展示，`forward_use_loc` 用于 edge evidence 的起点。

### 2.2 `LG.feature_column(slot_id, fc_name=None, seqlen=-1, version=...)`

**作用**：声明 sparse/embedding feature column。它不一定直接赋给 `self.xxx`，大量模型会先写入 dict，再通过 `.get_vector(slice)` 或 `.get_size_tensor()` 产生 tensor。

**典型代码**：

```python
fc_dict[2] = LG.feature_column(slot_id=2, seqlen=-1, version=1)
seq_fc_dict[f_slot] = LG.feature_column(slot_id=fs_dict[f_slot].slot_id, fc_name=f_name, seqlen=seq_len, version=2)
seq_tensor = fc_dict[slot_id].get_vector(slc)
mask4att = fc_dict[slot_id].get_size_tensor()
```

**返回语义**：返回 FeatureColumn-like 对象，不一定是 `nn.Module`，但它是输入特征的声明对象。它通过 `get_vector()` / `get_size_tensor()` 等方法把 sparse 特征转成 tensor。

**scanner 关注点**：

- 不能只识别 `self.attr = LG.feature_column(...)`，还要识别 `dict[key] = LG.feature_column(...)`、局部变量赋值、helper 函数返回等路径。
- 对于 `feature_column`，forward 使用位置可能不是 `fc()`，而是 `fc.get_vector(...)`、`fc.get_size_tensor()` 或经 dict 读取后的方法调用。
- 如果只做 direct-call scanner，会漏掉大部分 sparse 输入源。

### 2.3 `LG.get_sample_rate()`

**作用**：声明 sample rate 输入模块。通常在 `__init__` 赋给 `self.sample_rate` / `self.sample_rate_module`，在 `forward()` 中调用并传入 loss/head 逻辑。

**典型代码**：

```python
self.sample_rate_module = LG.get_sample_rate()
sample_rate = self.sample_rate_module()
_ = result.head(..., sample_rate=sample_rate, ...)
```

或：

```python
self.sample_rate = LG.get_sample_rate()
sample_rate = self.sample_rate()
```

**返回语义**：从 mock 代码看，它和 dense feature 一样返回一个 callable module，`forward()` 返回 float tensor。

**scanner 关注点**：

- 属于 InputAttr，但 `lg_source_kind` 应区分为 `get_sample_rate`，因为它通常参与 loss/reweight/head，而不是普通 feature embedding。
- `forward_use_loc` 仍然是 `self.sample_rate()` 调用位置。

### 2.4 `LG.get_bias()` / `LG.get_sample_bias()` / `LG.label(...)`

虽然本次问题重点提到 feature_column、dense_feature、get_sample_rate、result，但现有模型里还存在：

```python
self.sample_bias = LG.get_sample_bias()
self.label = LG.label("label", 0)
```

这些 API 也属于输入源：

- `get_sample_bias()`：训练样本 bias 输入，常和 sample rate 一起参与 logit 校正。
- `label(...)`：label 输入源，在 loss module 中通过 `self.label()` 读取。
- `get_bias()`：bias 类输入，和 feature/bias slot 有关。

scanner 的正式设计里应该将这些纳入同一套 InputAttr 分类，而不是只覆盖当前 case 出现的几个函数。

---

## 3. Result API 语义总结

### 3.1 `LG.result()`

**作用**：创建结果收集对象。模型 `forward()` 通常返回这个对象，或者返回包含 result 的 tuple。它是 DAG 顶层 `Result` 节点的语义来源。

**典型代码**：

```python
result = LG.result()
_ = result.head(name=..., prediction=pred, label=..., loss=..., sample_rate=...)
return result
```

或链式写法：

```python
result = (
    LG.result()
    .head(name="ctr_head", prediction=y_pred_style, label=label, loss=total_loss_style.squeeze())
    .head(name="ctr_head_debias", prediction=y_pred_style_ori, label=label, loss=None)
)
return result
```

### 3.2 `result.head(...)`

**作用**：向 result 对象注册一个输出 head。每个 head 是一个模型结果出口，关键参数包括：

- `name`：head 名称。
- `prediction`：预测 tensor，是从模型主干或 tower 输出流入 Result 的主要数据依赖。
- `label`：训练 label，可来自 `LG.label` 或其他 label/mask module。
- `loss`：训练 loss。
- `sample_rate`：样本率输入，常来自 `LG.get_sample_rate()`。
- `classifier_type`：分类/回归类型，例如 `LGResult.CLASSIFIER_BINARY`、`LGResult.CLASSIFIER_REGRESSION`。

**scanner 关注点**：

- ResultAttr 的 `call_loc` 应该对应每个 `.head(...)` 调用行，而不是只对应 `LG.result()` 行。
- 需要支持 `result = LG.result(); result.head(...)` 和 `LG.result().head(...).head(...)` 两种风格。
- 后续 edge 构图应从 `prediction` / `loss` / `label` / `sample_rate` 参数向上追踪 carrier，尤其 `prediction` 是模型主干到 Result 的关键边。

---

## 4. scanner 要解决的核心问题定义

### 4.1 为什么需要 scanner

当前静态 DAG 的目标结构是：

```text
Input -> Root -> Result
```

用户模型中的 `torch.nn.Module` 都应该挂在 Root 内，并通过源码数据流连接。过去的实现更多关注 `self.xxx = SomeModule()` 与 `forward()` 中的 `self.xxx(...)` 调用，但 lagrange_torch 模型里还有一类重要对象：

- 它们不是普通业务 `nn.Module` 子模块；
- 但它们是模型输入或输出边界；
- 它们的定义和使用决定了 DAG 的 Input/Result 边界是否正确。

scanner 的核心任务就是把这些边界对象用 AST 统一抽取出来，交给静态 DAG 构图使用，而不是让前端 HTML 或散落的规则补边。

### 4.2 scanner 的正式输出对象

建议 scanner 输出四类 Attr 的统一列表或 registry：

1. **ModuleAttr**：普通 `torch.nn.Module` 子模块。
2. **ContainerAttr**：`ModuleList`、`ModuleDict`、list/dict 容器中的模块或输入源。
3. **InputAttr**：LG input source，包括 `dense_feature`、`feature_column`、`get_sample_rate`、`get_bias`、`get_sample_bias`、`label`、`slot` 等。
4. **ResultAttr**：`result.head(...)` 注册出的输出 head。

现有 `attr_types.py` 已经有这些类型的雏形，但 `ModuleAttr` / `ResultAttr` 还过薄，后续可能需要补字段，例如：

- `ResultAttr.head_name`：静态可解析时记录 head name。
- `ResultAttr.kwarg_exprs`：记录 `prediction` / `label` / `loss` / `sample_rate` 的 AST 文本与 node。
- `InputAttr.source_args` / `source_kwargs`：保留 LG API 参数，便于展示和 debug。
- `InputAttr.forward_use_loc`：已有字段，应保留为 edge evidence 起点。

### 4.3 scanner 必须支持的模式

#### A. 直接 self 属性输入源

```python
self.x = LG.dense_feature(...)
y = self.x()
```

输出：`InputAttr(attr_name="x", call_loc=定义行, forward_use_loc=使用行, lg_source_kind="dense_feature")`。

#### B. sample rate / label / bias 类输入源

```python
self.sample_rate = LG.get_sample_rate()
sample_rate = self.sample_rate()
self.label = LG.label("label", 0)
label = self.label()
```

输出同样是 InputAttr，但要保留 kind，避免和普通 dense feature 混淆。

#### C. feature_column dict / local var / method call 输入源

```python
fc_dict[slot_id] = LG.feature_column(...)
seq = fc_dict[slot_id].get_vector(slc)
mask = fc_dict[slot_id].get_size_tensor()
```

这是当前原型最明显的缺口。scanner 需要能把 `feature_column` 的定义和 tensor 产生点关联起来，否则 sparse 输入不会进入 Input 边界。

#### D. result alias + head 调用

```python
result = LG.result()
result.head(name="x", prediction=pred, ...)
```

输出：每个 `.head(...)` 生成一个 ResultAttr。

#### E. 链式 result.head

```python
result = LG.result().head(...).head(...)
```

输出：链上每个 `.head(...)` 都应该生成 ResultAttr。

#### F. helper 方法与跨方法使用

```python
class M(nn.Module):
    def __init__(self):
        self.sample_rate = LG.get_sample_rate()

    def _loss(self):
        return self.sample_rate()

    def forward(self):
        return self._loss()
```

scanner 不应该只扫描 `forward()`，还需要结合 `ASTFrontend.get_reachable_helpers()` 识别从 forward 可达的 helper。

### 4.4 scanner 不应该做的事情

- 不在前端 JS 中计算 trace 分类、聚合或边；scanner 输出应在 Python 后端完成。
- 不使用正则兜底；源码解析必须 AST-only。
- 不把 `LG.feature_column` 误当普通 `ModuleAttr`；它是输入源声明，不是普通业务子模块。
- 不用“节点数变多”作为回退理由；Input/Result 边界补全后节点数增加是预期行为。

---

## 5. 当前 `attr_scanner.py` 原型评估

### 5.1 已覆盖能力

当前实现已经具备以下原型能力：

- 通过 AST 识别 `LG.<func>(...)`。
- 识别 `self.attr = LG.func(...)` 和 `VAR = LG.func(...)`。
- 在后续 `self.attr()` 或 `attr()` 调用时补 `forward_use_loc`。
- 识别 `LG.result().head(...)` 和简单 `result.head(...)`。
- 输出 `InputAttr` / `ResultAttr`。

### 5.2 主要缺口

1. **LG import alias 固定为 `LG`**：没有识别 `from ... import CONTEXT as X` 或其它 alias。
2. **class/name key 不够稳定**：`inputs` 只用 `attr_name` 去重，不含 class/file，跨类同名属性会冲突。
3. **Result alias 过宽**：只要变量名叫 `result` 且 `.head` 就会记录，没有确认它来自 `LG.result()`。
4. **链式 head 支持不完整**：当前 `_is_lg_result_head()` 只能稳定识别 `LG.result().head(...)` 或 `result.head(...)`，对 `.head(...).head(...)` 的链式嵌套需要明确递归解析。
5. **feature_column dict 路径缺失**：无法处理 `fc_dict[k] = LG.feature_column(...)` 和 `fc_dict[k].get_vector(...)`。
6. **helper 可达性缺失**：没有和 `ASTFrontend.get_reachable_helpers()` 打通，无法区分 forward 可达与不可达代码。
7. **没有参数语义保留**：Input/Result 只记录位置，未保留 API 参数，后续 evidence 和 debug 信息不足。
8. **扫描项目使用 `os.walk`**：需要和主流程 source file 管理、ASTFrontend cache、忽略规则合并，避免重复 parse 与路径不一致。

---

## 6. `ASTFrontend` 拆出为 `ast_frontend.py` 的可行性分析

### 6.1 当前 `ASTFrontend` 范围

`ASTFrontend` 当前位于 `scripts/analyze_trace.py` 的 **135-1239 行**，方法列表如下：

- 初始化与类注册：`__init__`、`_build_class_registry`、`_node_to_text`、`_build_line_starts`。
- nn.Module 判定：`_resolve_is_nn_module`、`is_nn_module`。
- 方法/类元数据：`get_method_lines`、`_get_method_node`。
- `__init__` 和 module 调用扫描：`get_init_assignments_ast`、`get_module_calls`、`get_first_call_loc`。
- forward/helper 可达性：`get_reachable_helpers`。
- 变量与数据流环境：`build_var_env`、`build_alias_env`、`build_dict_slot_env`、`extract_return_vars`。
- 动态属性与容器辅助：`get_self_param_aliases`、`get_dynamic_indexed_self_attrs`、`get_loop_expansion_records`。
- AST statement info：`_get_stmt_infos`、`_stmt_info_from_assign`、`_stmt_info_from_augassign`、`_stmt_info_from_expr`。
- 通用 AST helper：`_extract_target_names`、`_extract_name_list`、`_literal_key`、`_line_excerpt`、`_extract_self_attr_name`、`_normalize_index_expr`、`_extract_self_method_name`、`_extract_module_call_info`、`_extract_called_self_attr`。
- ConstantTable 接入：`get_constant_table`。
- local statement parser：`parse_local_stmt`、`parse_local_ctor_assign`、`parse_local_lg_assign`、`parse_local_fc_get_vector_assign`、`parse_local_container_ctor`、`parse_local_append_ctor`、`parse_local_subscript_ctor`、`parse_local_var_ctor`、`parse_local_setattr_ctor`。

### 6.2 为什么适合拆出

`ASTFrontend` 已经不是 `analyze_trace.py` 的局部工具，而是多条 AST 能力的公共入口：

- static DAG 构图需要它获取类、方法、模块调用和变量流。
- ConstantTable / ConstantResolver 需要它提供 class registry 与 source file 视图。
- scanner 需要它识别 LG source、forward 可达 helper、call loc、result head 参数表达式。
- 后续 edge evidence 也需要复用它的 var env / dict slot env 能力。

继续放在 1.1 万行的 `analyze_trace.py` 中，会让 scanner 和后续模块化都被主脚本牵制，且文件行数已经明显超过 2000 行约束。拆出是合理的。

### 6.3 建议迁移边界

建议把 **完整 `ASTFrontend` 类**迁移到 `scripts/ast_frontend.py`，不要只迁一部分方法。原因：

- 类内部方法互相调用较多，局部迁移会增加双向依赖。
- scanner 需要的不只是简单 `node_to_text`，还可能需要 reachable helpers、var env、dict slot env。
- 一次迁移完整类，`analyze_trace.py` 只保留 import，可降低后续功能重复实现的风险。

建议目标文件：

```python
# scripts/ast_frontend.py
class ASTFrontend:
    ...
```

`analyze_trace.py` 改为：

```python
try:
    from .ast_frontend import ASTFrontend
except ImportError:
    from ast_frontend import ASTFrontend
```

### 6.4 需要处理的依赖

拆出时最大的依赖问题是 `ASTFrontend.get_constant_table()` 引用了后文定义的 `ConstantTable`，而 `get_loop_expansion_records()` 引用了 `ConstantResolver` 与 `Scope`。

有两种可行路线：

#### 路线 A：先做轻量抽出，保留延迟 import

在 `ast_frontend.py` 中保留 ASTFrontend，并在需要 ConstantTable 的方法内做延迟 import，例如：

```python
def get_constant_table(...):
    try:
        from .ast_constants import ConstantTable
    except ImportError:
        from ast_constants import ConstantTable
```

但这要求同时把 ConstantTable/ConstantResolver/Scope 拆到 `ast_constants.py`。如果不拆 constants，会形成 `ast_frontend -> analyze_trace -> ast_frontend` 的循环依赖，不建议。

#### 路线 B：ASTFrontend 与 ConstantTable 同步模块化

- `scripts/ast_frontend.py`：ASTFrontend。
- `scripts/ast_constants.py`：Scope、ConstantTable、ConstantResolver。
- `scripts/analyze_trace.py`：只作为 orchestration 和 CLI 主流程。
- `scripts/attr_scanner.py`：依赖 `ast_frontend.py`，不再自行重复 parse 所有能力。

路线 B 更干净，符合模块化方向，但改动面更大，需要单独测例设计和回归验证。

### 6.5 推荐执行顺序

因为这是设计阶段，暂不直接编码。后续建议按以下顺序推进：

1. **先补 scanner 测例设计**：覆盖 direct dense、sample_rate、label/bias、feature_column dict、result alias、chain head、helper 可达。
2. **再拆 ASTFrontend**：先迁完整类和 constants，保持行为不变，跑 `test_dag_rules.py`。
3. **再重写 scanner 接入 ASTFrontend**：让 scanner 复用 `ASTFrontend` 的 class/method/statement 能力。
4. **最后接入 DAG 构图**：把 InputAttr/ResultAttr 作为 Root 的边界节点，由后端生成 edge evidence，前端只展示。

---

## 7. scanner 正式设计建议

### 7.1 模块结构

建议拆成以下模块：

```text
scripts/
  ast_frontend.py        # ASTFrontend，统一 AST 元数据前端
  ast_constants.py       # Scope / ConstantTable / ConstantResolver
  attr_types.py          # Attr / ModuleAttr / ContainerAttr / InputAttr / ResultAttr
  attr_scanner.py        # 基于 ASTFrontend 的 Attr 扫描器
  analyze_trace.py       # 主流程 orchestration / CLI
```

### 7.2 scanner 输入

scanner 不应该自己 `os.walk + ast.parse` 成孤岛，而应接收主流程已经准备好的 source files 与 ASTFrontend cache：

```python
scan_project_attrs(source_files: dict[str, list[str]], ast_frontends: dict[str, ASTFrontend]) -> AttrScanResult
```

建议输出：

```python
@dataclass
class AttrScanResult:
    inputs: list[InputAttr]
    results: list[ResultAttr]
    by_class: dict[tuple[str, str], list[Attr]]
    diagnostics: list[AttrScanDiagnostic]
```

### 7.3 LG alias 识别

扫描每个文件的 import：

```python
from bytedance.lagrange_torch.context import CONTEXT as LG
from bytedance.lagrange_torch.context import CONTEXT as M
```

建立 `lg_context_aliases = {"LG", "M"}`。后续判断 `LG.dense_feature(...)` 时，不写死 `LG`。

### 7.4 InputAttr 扫描策略

分两阶段：

1. **定义阶段**：扫描 `__init__` 和初始化 helper，记录 input source binding。
   - `self.attr = LG.dense_feature(...)`
   - `self.attr = LG.get_sample_rate()`
   - `local = LG.feature_column(...)`
   - `dict[key] = LG.feature_column(...)`
2. **使用阶段**：扫描 `forward` 可达方法，记录 tensor 产生点。
   - `self.attr()`
   - `local()`
   - `fc.get_vector(...)`
   - `fc.get_size_tensor()`
   - `dict[key].get_vector(...)`

其中 `feature_column` 可能需要从 dict/local binding 映射到 synthetic InputAttr，例如：

```text
InputAttr attr_name="fc_dict[2]" lg_source_kind="feature_column"
```

### 7.5 ResultAttr 扫描策略

也分两阶段：

1. 识别 result 对象来源：
   - `result = LG.result()`
   - `run_train = LG.result()`
   - `return LG.result().head(...)`
2. 递归识别 `.head(...)` 调用链：
   - `result.head(...)`
   - `LG.result().head(...)`
   - `LG.result().head(...).head(...)`

ResultAttr 应保存每个 head 的参数表达式，尤其是 `prediction`，用于后续从 Result 反向追踪 carrier。

### 7.6 与 edge evidence 的关系

scanner 只负责“发现边界对象和关键 call loc”，不负责最终边计算。建议职责切分：

- scanner：输出 InputAttr / ResultAttr 以及 call loc、参数 AST。
- edge builder：基于 `ASTFrontend.build_var_env()`、`build_dict_slot_env()`、`extract_return_vars()`、已有 `var_history`，生成 Input/Result 的 edge evidence。
- frontend_html：只展示后端生成的节点、边、evidence，不做 trace 事件计算或源码推断。

---

## 8. 后续测例设计方向

正式编码前应先给出测例设计并确认。建议至少覆盖：

1. **dense direct**：`self.x = LG.dense_feature(...)`，`forward` 调用 `self.x()`，断言 InputAttr kind、定义行、使用行。
2. **sample_rate + head**：`self.sample_rate = LG.get_sample_rate()`，`sample_rate = self.sample_rate()`，`result.head(sample_rate=sample_rate)`。
3. **label/sample_bias**：`self.label = LG.label(...)`、`self.sample_bias = LG.get_sample_bias()`。
4. **feature_column dict**：`fc_dict[k] = LG.feature_column(...)`，`fc_dict[k].get_vector(...)`。
5. **result alias**：变量名不是 `result`，例如 `run_train = LG.result(); run_train.head(...)`。
6. **chain head**：`LG.result().head(...).head(...)`，断言两个 ResultAttr。
7. **helper 可达**：input 使用发生在 `forward()` 调用的 helper 中。
8. **negative case**：普通 `result.head(...)` 但 result 不是 `LG.result()`，不生成 ResultAttr。

---

## 9. 本阶段不执行的内容

本报告只补设计分析，不修改 scanner / ASTFrontend / DAG 主流程代码。原因：

- NaN 已明确指出 scanner 跳过设计直接编码流程不对，因此当前阶段应先恢复设计流程。
- bug 修复和新增测例都需要先给测例设计方案确认，不能直接写实现。
- ASTFrontend 拆文件属于较大重构，需要单独执行“基线备份、迁移、全量 test_dag_rules.py、HTML 产物打包”的流程。

