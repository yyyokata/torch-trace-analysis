# 5476790 节点缺口补充测例设计方案

> 范围：本文件只补充测例设计与实现方案，不修改代码。执行前需等待 NaN 确认。
>
> 约束：AST-only；禁止正则兜底；严禁 annotation fallback。任何无法沿真实 caller 链、显式实参、局部变量、字面量或可解析常量表追踪到的值，必须返回 `None`。

## 1. 当前边界确认

已读设计文档：`design/ast_refactor_5476790_node_gap_fix_plan.md` 与 `design/analyze_trace_ast_refactor_plan.lark.md`。

5476790 当前关注缺口从历史 `196 -> 229` 收敛到当前 `221 -> 229`，剩余差距 8 个节点。已知根因分三类：

1. **根因 A：SeqTrans.trans_blocks 展开链不完整**
   - 触发点：`SeqTrans.__init__` 中 `range(self.config.num_hidden_layers)`。
   - 允许路径：只能沿真实 caller 链追踪，例如 `C2kTrans/InsTrans -> TransformerConfig(**self.xxx_base_params) -> num_hidden_layers`。
   - 禁止路径：不得从 `config: TransformerConfig` 类型注解或 dataclass 字段默认值直接猜测。
2. **根因 B：`self.attr.helper(...)` 调用不被识别为 use**
   - 触发点：`self.seq_trans.dense_query(...)` / `self.ins_trans.dense_query(...)`。
   - 影响：dead-child 过滤可能误裁 `seq_trans` / `ins_trans` 子树。
3. **根因 C：DomainEmbedding 4 个 attr 注册/覆盖待单独定位**
   - 真实源码中至少有 4 处关键直接注册：
     - `ApgStar.gen_domain_embedding = DomainEmbedding(...)`
     - `SequenceModule.seq_ads_domain_embeddings = DomainEmbedding(...)`
     - `SeqBias.bias_domain = DomainEmbedding(...)`
     - `LogitAdaptive.la_domain = DomainEmbedding(...)`
   - `DomainEmbedding` 自身在 `model_common.py` 中注册 `self.domain_embedding = nn.Embedding(...)`。

现有文档已经覆盖 E1.1、E1.2、E1.3、E2 的基本 synthetic 设计，但需要补充两类关键保护：

- **禁止 annotation fallback 的负向测例**，防止 E1.2 修复再次走错路径；同时增加 dataclass default 正向测例，确认真实 `Config()` 构造语义被保留。
- **DomainEmbedding 真实模式的独立 synthetic 覆盖**，避免只靠 5476790 端到端节点数定位。

---

## 2. 测例分层总览

建议新增/补强三层测例：

1. **AST evaluator 单元测例**：验证常量解析的允许路径与禁止路径。
2. **静态树 synthetic 测例**：验证 `build_static_module_tree()` 的 attrs、first_call_loc、reachability。
3. **5476790 端到端防回退测例**：验证关键类/attr 覆盖与节点数下限。

<table header-row="true" header-col="false" col-widths="180,220,260,260">
  <tr>
    <td>编号</td>
    <td>目标根因</td>
    <td>Mock 数据核心</td>
    <td>核心断言</td>
  </tr>
  <tr>
    <td>A1-positive</td>
    <td>真实 caller 链解析 `range(self.config.num_hidden_layers)`</td>
    <td>`Parent.stack = Stack(Config(**self.params))`</td>
    <td>展开 `blocks[0..5]`；resolver 返回 6</td>
  </tr>
  <tr>
    <td>A2-negative</td>
    <td>禁止 annotation fallback</td>
    <td>`Stack.__init__(config: Config)`，Root 仅接收 runtime config</td>
    <td>resolver 返回 `None`；不展开具体 `blocks[i]`</td>
  </tr>
  <tr>
    <td>A3-positive</td>
    <td>dataclass default 是真实 `Config()` 构造语义</td>
    <td>`Config.num_hidden_layers=6`，caller 显式传入 `Config()`</td>
    <td>resolver 返回 `6`；正确展开 `blocks[0..5]`</td>
  </tr>
  <tr>
    <td>B1-helper</td>
    <td>`self.attr.helper(...)` 作为 use</td>
    <td>`Wrapper.bb.custom_query(x)`</td>
    <td>`first_call_loc['bb']` 存在；`Backbone` 不被裁剪</td>
  </tr>
  <tr>
    <td>C1-domain-direct</td>
    <td>DomainEmbedding 直接注册</td>
    <td>4 个父类分别 `self.xxx = DomainEmbedding(...)`</td>
    <td>4 个父 attr 均存在；`DomainEmbedding.domain_embedding` 存在</td>
  </tr>
  <tr>
    <td>C2-domain-e2e</td>
    <td>5476790 真实 DomainEmbedding 覆盖</td>
    <td>真实 `testset/extracted/5476790`</td>
    <td>关键 attr 路径均出现在 DAG JSON group/node 映射中</td>
  </tr>
</table>

---

## 3. 详细测例设计

### 3.1 A1-positive：真实 caller 链 + `**dict` 显式值解析

**目的**：覆盖 5476790 中 `TransformerConfig(**self.ins_trans_base_params)` / `TransformerConfig(**self.seq_trans_base_params)` 这类真实 caller 链。解析成功必须来自显式 dict 值，不来自注解或 dataclass 默认值。

**Mock 数据**：

```python
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Config:
    num_hidden_layers: int = 999  # 故意设置为错误默认值，防止误用 default
    hidden_size: int = 8

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)

class Stack(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            Block() for _ in range(self.config.num_hidden_layers)
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = {"num_hidden_layers": 6, "hidden_size": 8}
        self.stack = Stack(Config(**self.params))

    def forward(self, x):
        return self.stack(x)
```

**断言**：

1. ConstantTable 层：`instance_const_chain[("Root", "stack")]["config"]["num_hidden_layers"] == 6`。
2. ConstantResolver 层：对 `Stack.__init__` 中 `self.config.num_hidden_layers`，在 `Scope(parent_cls="Root", parent_attr="stack")` 下返回 `6`。
3. Static tree 层：`tree['Stack']['attrs']` 包含 `blocks[0]` 到 `blocks[5]`，不包含 `blocks[6]`。
4. 优先级断言：这里 caller 通过 `Config(**self.params)` 显式传入 `num_hidden_layers=6`，所以必须返回 6；若实现忽略显式参数而错误使用 default 999，本测例必须失败。

### 3.2 A2-negative：只有类型注解，无 caller 实参，必须返回 None

**目的**：专门防止 annotation fallback。即使 `config: Config` 有明确类型，也不得凭类型注解读取默认值。

**Mock 数据**：

```python
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Config:
    num_hidden_layers: int = 6

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

class Stack(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block() for _ in range(self.config.num_hidden_layers)])

class Root(nn.Module):
    def __init__(self, runtime_config):
        super().__init__()
        self.stack = Stack(runtime_config)
```

**断言**：

1. 在 `Scope(cls="Stack", method="__init__", parent_cls="Root", parent_attr="stack")` 下，`resolver.eval_int(self.config.num_hidden_layers)` 必须返回 `None`。
2. `table.instance_const_chain[("Root", "stack")]` 不应包含可用的 `config.num_hidden_layers`。
3. `build_static_module_tree()` 不应展开具体 `blocks[0..5]`；可接受保持 wildcard 或不展开，但不得出现 6 个具体 Block。

### 3.3 A3-positive：caller 显式构造 Config 时，dataclass default 是真实传入值

**目的**：确认 `Config()` 的 dataclass default 按 Python 构造语义参与常量传播。它不是 annotation fallback：annotation fallback 是仅凭 `config: Config` 类型注解去猜；而本测例中 caller 明确执行了 `Config()`，默认字段值是构造结果的一部分，属于真实 caller 链。

**Mock 数据**：

```python
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Config:
    num_hidden_layers: int = 6

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

class Stack(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block() for _ in range(self.config.num_hidden_layers)])

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = Stack(Config())  # 未显式传 num_hidden_layers
```

**断言**：

1. `resolver.eval_int(self.config.num_hidden_layers, Scope(...Root.stack...))` 必须返回 `6`。
2. `tree['Stack']['attrs']` 必须包含完整 `blocks[0]` 到 `blocks[5]` 展开。
3. 不应包含 `blocks[6]`。
4. 本测例必须与 A2-negative 成对存在：A2 证明“只有 annotation 不可猜”，A3 证明“真实 `Config()` 构造 default 可用”。

### 3.4 B1-helper：helper-method call 保留子树

现有设计文档 5.4 已有基本 mock。建议补强断言到 DAG JSON 层，避免只看 tree。

**Mock 数据**：沿用 `Backbone.custom_query()` / `Wrapper.bb.custom_query(x)`。

**补充断言**：

1. `tree['Wrapper']['first_call_loc']['bb']` 存在，line 指向 `return self.bb.custom_query(x)`。
2. 生成 DAG JSON 后，`class_name == 'Backbone'` 至少出现 1 个节点或 group。
3. `Wrapper` 的 `attr_id_map` 中包含 `bb`，且映射目标不是被裁剪后的 dangling 占位。
4. Rule1c 不应因此改写 dep-edge evidence 到 `__init__`；若产生边，evidence 必须仍指 forward/helper 调用行。

### 3.5 C1-domain-direct：DomainEmbedding 直接注册 synthetic

**目的**：把 DomainEmbedding 的 4 个直接注册场景从 5476790 中抽出来，确认 analyzer 对“跨文件用户模块 + 内部 `nn.Embedding` 子模块 + 多父类 attr”稳定。

**Mock 数据**：建议放在一个 synthetic modelcode 目录，包含 `model_common.py` 与 `main_model.py` 两个文件。

`model_common.py`：

```python
import torch
from torch import nn

class DomainEmbedding(nn.Module):
    def __init__(self, val_cnt, name_space="domain_embedding", dim_size=8):
        super().__init__()
        self.domain_embedding = nn.Embedding(val_cnt, dim_size)
        self.name_space = name_space

    def forward(self, domain):
        idx = torch.zeros_like(domain)
        return self.domain_embedding(idx.long()).squeeze(1)
```

`main_model.py`：

```python
import torch
from torch import nn
from model_common import DomainEmbedding

class ApgStar(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_domain_embedding = DomainEmbedding(7, "apg_star_domain")
    def forward(self, x):
        return self.gen_domain_embedding(x)

class SequenceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_ads_domain_embeddings = DomainEmbedding(7, "seq_ads_domain")
    def forward(self, x):
        return self.seq_ads_domain_embeddings(x)

class SeqBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias_domain = DomainEmbedding(7, "bias_domain", 1)
    def forward(self, x):
        return self.bias_domain(x)

class LogitAdaptive(nn.Module):
    def __init__(self):
        super().__init__()
        self.la_domain = DomainEmbedding(7, "la_domain")
    def forward(self, x):
        return self.la_domain(x)

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.apg = ApgStar()
        self.seq = SequenceModule()
        self.bias = SeqBias()
        self.la = LogitAdaptive()
    def forward(self, x):
        a = self.apg(x)
        b = self.seq(x)
        c = self.bias(x)
        d = self.la(x)
        return a + b + c + d
```

**断言**：

1. `tree` 必须包含 `DomainEmbedding`。
2. `tree['DomainEmbedding']['attrs']['domain_embedding']` 为 `nn.Embedding` 或 `Embedding`。
3. 以下父类 attr 必须存在：
   - `ApgStar.gen_domain_embedding -> DomainEmbedding`
   - `SequenceModule.seq_ads_domain_embeddings -> DomainEmbedding`
   - `SeqBias.bias_domain -> DomainEmbedding`
   - `LogitAdaptive.la_domain -> DomainEmbedding`
4. DAG JSON 层必须能找到 4 个父类下对应 attr；不能只在 class-level tree 里存在、最终被 reachability 裁掉。
5. 这 4 个 attr 的 first_call_loc 均应指向各自 forward 中的 `self.xxx(x)` 调用行。

### 3.6 C2-domain-e2e：5476790 真实覆盖断言

**目的**：直接保护 5476790 的 4 个 DomainEmbedding attr，防止端到端节点数过关但关键模块仍缺失。

**输入**：`testset/extracted/5476790`。

**断言函数建议**：新增 `check_5476790_domain_embedding_coverage(data)`，检查 DAG JSON 的 groups/nodes/attr 映射。

**必查路径**：

```python
required = {
    "ApgStar": ["gen_domain_embedding"],
    "SequenceModule": ["seq_ads_domain_embeddings"],
    "SeqBias": ["bias_domain"],
    "LogitAdaptive": ["la_domain"],
    "DomainEmbedding": ["domain_embedding"],
}
```

**断言细节**：

1. 对每个 parent class，至少存在一个 `label == parent class` 的 group。
2. 对每个 required attr，至少一个对应 parent group 的 `attr_id_map` 包含该 attr。
3. `DomainEmbedding` 的 group 或 node 至少出现一次。
4. `domain_embedding` 子 attr 至少出现一次，并映射到 `Embedding`/`nn.Embedding`。
5. 如果当前 DAG JSON 对 node/group class 字段命名不统一，断言实现应只做结构查询，不依赖 HTML 展示文案。

---

## 4. 测例落地位置建议

在 NaN 确认后再执行，建议分两处落地：

1. `develop/ast/storage/testset/test_dag_rules.py`
   - 保留当前 inline synthetic 风格，新增 A1/A2/A3/B1/C1 的 `_SYNTH_*` 源码字符串与测试函数。
   - 在 `run_ast_synthetic_tests()` 中纳入新增测试。
   - 新增 `check_5476790_domain_embedding_coverage(data)` 并在 `run_model()` 针对 5476790 时调用。
2. `develop/ast/storage/testset/synthetic_cases/run_synthetic_tests.py`
   - 对 DomainEmbedding 多文件 synthetic，也可放在 `synthetic_cases/domain_embedding_direct/modelcode/`，由 runner 统一执行。
   - 若希望减少文件数量，优先放在 `test_dag_rules.py` inline source；多文件 case 可以用临时目录写两份文件。

---

## 5. 实施顺序建议

1. **先补 A2-negative + A3-positive 成对测例**：A2 防止 annotation fallback；A3 确认 dataclass default 作为真实 `Config()` 构造语义可解析。
2. **补 A1 positive**：确保真实 caller 链 + `**dict` 显式值可解析。
3. **补 B1 helper DAG JSON 断言**：保护 dead-child 过滤行为。
4. **补 C1 DomainEmbedding synthetic**：隔离根因 C。
5. **补 C2 5476790 e2e coverage**：最后把真实模型覆盖纳入防回退。

---

## 6. 验证命令设计

确认并实现测例后，每次修改必须使用当前 AST 任务目录脚本：

```bash
cd develop/ast/storage
ANALYZE_SCRIPT_OVERRIDE=/workspace/iris_a493ea8b-53dc-4055-b5a7-cca66d850ce9/develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py \
  python3 testset/test_dag_rules.py
```

若同步使用 synthetic runner：

```bash
cd develop/ast/storage
ANALYZE_SCRIPT_OVERRIDE=/workspace/iris_a493ea8b-53dc-4055-b5a7-cca66d850ce9/develop/ast/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py \
  python3 testset/synthetic_cases/run_synthetic_tests.py
```

验收标准：

1. 新增 synthetic 测例全部 PASS。
2. 5476790 domain coverage PASS。
3. 全量 `test_dag_rules.py` 保持 7/7 PASS；修复后 5476790 nodes 目标回到 ≥229。
4. 不出现任何通过 annotation fallback 兜底解析 `num_hidden_layers` 的路径；dataclass default 仅在 caller 真实构造 `Config()` 时允许参与解析。

---

## 7. 风险与防护

<table header-row="true" header-col="false" col-widths="220,300,300">
  <tr>
    <td>风险</td>
    <td>表现</td>
    <td>防护测例</td>
  </tr>
  <tr>
    <td>误用 annotation fallback</td>
    <td>A2 中 runtime_config 被错误展开成 6 层</td>
    <td>A2-negative 必须返回 None</td>
  </tr>
  <tr>
    <td>dataclass default 构造语义丢失</td>
    <td>A3 中 `Config()` 未解析出默认字段，导致无法展开 6 层</td>
    <td>A3-positive 必须返回 6 并展开 `blocks[0..5]`</td>
  </tr>
  <tr>
    <td>`**dict` 显式值链断裂</td>
    <td>A1 无法从 `self.params` 解析到 6</td>
    <td>A1-positive 要求展开 6 层且不能误取 999</td>
  </tr>
  <tr>
    <td>helper call 仍被 dead-child 裁剪</td>
    <td>`self.bb.custom_query` 不保留 Backbone</td>
    <td>B1-helper 检查 first_call_loc 与 DAG JSON</td>
  </tr>
  <tr>
    <td>DomainEmbedding 只在 tree 存在但 DAG 被裁</td>
    <td>真实 5476790 仍缺 4 个 attr</td>
    <td>C1 synthetic + C2 e2e coverage</td>
  </tr>
</table>

---

## 8. 已确认点

1. A3 按 positive 执行：`Stack(Config())` 中 dataclass default 是真实构造语义，`num_hidden_layers=6` 必须可解析，应该展开 `blocks[0..5]`。
2. A2 按两阶段执行：无 runtime 配置时返回 `None` 并 warning/error；用户补充 `runtime_overrides` 后必须可解析并展开。
3. DomainEmbedding 的 4 个真实 attr 覆盖按本文 C2 列表作为 5476790 防回退硬断言。